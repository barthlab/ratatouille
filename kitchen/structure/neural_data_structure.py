import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional, Any, Self, Tuple, Iterable
from functools import cached_property

from kitchen.settings.timeline import SUPPORTED_TIMELINE_EVENT
from kitchen.settings.fluorescence import DETREND_BASELINE_WINDOW, DETREND_BASELINE_PERCENTILE, TRIAL_DF_F0_WINDOW
from kitchen.utils.numpy_kit import numpy_percentile_filter


@dataclass
class TimeSeries:
    """
    Base class for time series data with timestamps.

    Stores temporal data (neural activity, behavior, events) with validation
    for consistent lengths and ascending time order.

    the last frame of attribute v should be the time dimension.
    """
    v: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        """Validate consistent lengths and ascending time order."""
        assert self.t.ndim == 1, f"times should be 1-d array, got {self.t.shape} with {self.t.ndim} dimensions"
        assert self.v.shape[-1] == len(self.t), f"values, times have different length: {self.v.shape} vs {self.t.shape}"
        assert np.all(np.diff(self.t) >= 0), f"times should be in ascending order, got {self.t}"

    def __len__(self):
        """Return number of time points."""
        return len(self.t)
    
    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"TimeSeries with {len(self.t)} frames from {self.t[0]:.2f} to {self.t[-1]:.2f}"

    def segment(self, start_t: float, end_t: float) -> "TimeSeries":
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        assert end_t >= start_t, f"start time {start_t} should be earlier than end time {end_t}"
        segment_start_index = np.searchsorted(self.t, start_t)
        segment_end_index = np.searchsorted(self.t, end_t)
        return TimeSeries(
            v=self.v[..., segment_start_index: segment_end_index],
            t=self.t[segment_start_index: segment_end_index]
        )
    
    @cached_property
    def fs(self) -> float:
        """Return sampling frequency."""
        return float(1 / np.mean(np.diff(self.t)))
    
    def aligned_to(self, align_time: float) -> "TimeSeries":
        """Align time series to a specific time point."""
        return TimeSeries(v=self.v.copy(), t=self.t - align_time)



@dataclass
class Events:
    v: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        """Validate consistent lengths and ascending time order."""
        assert self.t.ndim == 1 == self.v.ndim, \
            f"values, times should be 1-d array, got {self.v.shape} and {self.t.shape}"
        assert self.v.shape == self.t.shape, \
            f"values, times have different length, got {self.v.shape} and {self.t.shape}"
        if len(self.t) > 0:
            assert np.all(np.diff(self.t) >= 0), "times should be in ascending order"
    
    def __len__(self):
        """Return number of time points."""
        return len(self.t)
    
    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation. If value is float, round to 2 decimal places."""
        example_events = [str(round(event, 2)) if isinstance(event, float) else str(event) for event in self.v[:3]]
        return f"{len(self)} Events: [{', '.join(example_events)} ...]"
    
    def __iter__(self) -> Generator[Tuple[Any, float], None, None]:
        """Iterate over events."""
        for t, v in zip(self.t, self.v):
            yield t, v

    def __add__(self, other: "Events") -> "Events":
        """Concatenate two events with same value type."""
        assert self.v.dtype == other.v.dtype, f"Cannot concatenate events with different value types {self.v.dtype} and {other.v.dtype}"
        new_t = np.concatenate([self.t, other.t])
        new_v = np.concatenate([self.v, other.v])
        # sort t and v together
        sort_idx = np.argsort(new_t)
        return Events(v=new_v[sort_idx], t=new_t[sort_idx])
    
    def __radd__(self, other: "Events") -> "Events":
        """Concatenate two events with same value type."""
        if other == 0:
            return self
        return self.__add__(other)

    def segment(self, start_t: float, end_t: float) -> Self:
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        assert end_t >= start_t, f"start time {start_t} should be earlier than end time {end_t}"
        segment_start_index = np.searchsorted(self.t, start_t)
        segment_end_index = np.searchsorted(self.t, end_t)
        return self.__class__(
            v=self.v[segment_start_index: segment_end_index],
            t=self.t[segment_start_index: segment_end_index]
        )        

    def inactivation_window_filter(self, inactivation_window: float) -> "Events":
        """Filter out events within inactivation_window seconds of previous events."""
        assert inactivation_window > 0, "inactivation window should be positive"
        if len(self.t) == 0:
            return Events(v=np.array([]), t=np.array([]))

        new_v, new_t = [self.v[0]], [self.t[0]]
        for i in range(1, len(self.t)):
            if self.t[i] - self.t[i-1] >= inactivation_window:
                new_v.append(self.v[i])
                new_t.append(self.t[i])
        return Events(v=np.array(new_v), t=np.array(new_t))

    def frequency(self, bin_size: float) -> TimeSeries:
        """Compute event frequency (events/sec) within bin_size seconds."""
        assert bin_size > 0, "bin size should be positive"
        assert len(self) > 0, "Cannot compute frequency for empty events"
        bins = np.arange(self.t[0]-bin_size, self.t[-1] + bin_size, bin_size)
        counts, bin_edges = np.histogram(self.t, bins=bins)
        return TimeSeries(v=counts / bin_size, t=bin_edges[:-1] + bin_size/2)

    def rate(self, bin_size: float) -> TimeSeries:
        """If value is number like, compute value rate (values / sec) within bin_size seconds."""
        assert np.issubdtype(self.v.dtype, np.number), f"Cannot compute rate for non-number value type: {self.v.dtype}"
        assert bin_size > 0, "bin size should be positive"
        assert len(self) > 0, "Cannot compute rate for empty events"
        bins = np.arange(self.t[0]-bin_size, self.t[-1] + bin_size, bin_size)
        sum_in_bin, bin_edges = np.histogram(self.t, bins=bins, weights=self.v)
        return TimeSeries(v=sum_in_bin / bin_size, t=bin_edges[:-1] + bin_size/2)

    def aligned_to(self, align_time: float) -> "Events":
        """Align events to a specific time point."""
        return Events(v=self.v.copy(), t=self.t - align_time)


@dataclass
class Timeline(Events):
    def __post_init__(self):
        super().__post_init__()
        assert all(isinstance(x, str) for x in self.v), f"timeline should be list of str, got {self.v.dtype}"
        for x in self.v:
            assert x in SUPPORTED_TIMELINE_EVENT, f"Found unsupported event {x} in timeline"
    
    def filter(self, event_types: str | Iterable[str]) -> "Timeline":
        """Extract events of specific types."""
        if isinstance(event_types, str):
            event_types = [event_types]
        for event_type in event_types:
            assert event_type in SUPPORTED_TIMELINE_EVENT, f"Found unsupported event {event_type} in event_types"
        return Timeline(
            v=np.array([x for x in self.v if x in event_types]),
            t=np.array([t for x, t in zip(self.v, self.t) if x in event_types])
        )

    def task_time(self) -> Tuple[float, float]:
        """Return start and end time of task."""
        assert "task start" in self.v, f"Cannot find task start in {self.v}"
        assert "task end" in self.v, f"Cannot find task end in {self.v}"
        task_start = self.filter("task start").t[0]
        task_end = self.filter("task end").t[0]
        return task_start, task_end

    def aligned_to(self, align_time: float) -> "Timeline":
        """Align events to a specific time point."""
        return Timeline(v=self.v.copy(), t=self.t - align_time)


@dataclass
class Fluorescence:
    raw_f: TimeSeries
    fov_motion: TimeSeries
    cell_position: np.ndarray
    cell_idx: np.ndarray
    cell_order: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        assert self.raw_f.v.shape == (self.num_cell, self.num_timepoint), \
            f"raw_f: Expected shape {(self.num_cell, self.num_timepoint)}, but got {self.raw_f.v.shape}"
        assert self.fov_motion.v.shape == (2, self.num_timepoint), \
            f"fov_motion: Expected shape {(2, self.num_timepoint)}, but got {self.fov_motion.v.shape}"
        assert self.cell_position.shape == (self.num_cell, 2), \
            f"cell_position: Expected shape {(self.num_cell, 2)}, but got {self.cell_position.shape}"
        assert len(self.cell_idx) == self.num_cell, \
            f"cell_idx: Expected length {self.num_cell}, but got {len(self.cell_idx)}"
        
        if self.cell_order is not None:
            assert len(self.cell_order) == self.num_cell, \
                f"cell_order: Expected length {self.num_cell}, but got {len(self.cell_order)}"
        else:
            self.cell_order = np.arange(self.num_cell)

    @cached_property
    def num_cell(self):
        return len(self.cell_idx)
    
    @cached_property
    def num_timepoint(self):
        return len(self.raw_f.t)

    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"Fluorescence data with {self.num_cell} cells and {self.num_timepoint} timepoints"
    
    def segment(self, start_t: float, end_t: float) -> "Fluorescence":
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        assert end_t >= start_t, f"start time {start_t} should be earlier than end time {end_t}"
        return Fluorescence(
            raw_f=self.raw_f.segment(start_t, end_t),
            fov_motion=self.fov_motion.segment(start_t, end_t),
            cell_position=self.cell_position,
            cell_idx=self.cell_idx,
            cell_order=self.cell_order
        )
    
    def extract_cell(self, cell_idx: int) -> "Fluorescence":
        """Extract fluorescence of a single cell."""
        assert self.cell_order is not None, f"cell_order is not set: {self}"
        return Fluorescence(
            raw_f=TimeSeries(v=self.raw_f.v[cell_idx:cell_idx+1], t=self.raw_f.t),
            fov_motion=self.fov_motion,
            cell_position=self.cell_position[cell_idx:cell_idx+1],
            cell_idx=np.array([cell_idx]),
            cell_order=np.array([self.cell_order[cell_idx]])
        )
    
    @cached_property
    def detrend_f(self) -> TimeSeries:
        """Detrend fluorescence by subtracting the 5th percentile of each cell."""
        baseline = numpy_percentile_filter(
            self.raw_f.v, s=int(self.raw_f.fs * DETREND_BASELINE_WINDOW), q=DETREND_BASELINE_PERCENTILE)
        return TimeSeries(v=np.divide(self.raw_f.v - baseline, baseline, where=baseline!=0) , t=self.raw_f.t)
    
    @cached_property
    def df_f0(self) -> TimeSeries:
        """Compute dF/F0."""
        baseline = np.mean(self.raw_f.segment(*TRIAL_DF_F0_WINDOW).v, axis=-1, keepdims=True)
        return TimeSeries(v=(self.raw_f.v - baseline) / baseline , t=self.raw_f.t)
    
    def aligned_to(self, align_time: float) -> "Fluorescence":
        """Align fluorescence to a specific time point."""
        return Fluorescence(
            raw_f=self.raw_f.aligned_to(align_time),
            fov_motion=self.fov_motion.aligned_to(align_time),
            cell_position=self.cell_position,
            cell_idx=self.cell_idx,
            cell_order=self.cell_order
        )
    

@dataclass
class NeuralData:
    """
    Container for neural and behavioral data.

    Stores fluorescence data, timeline events, and various behavioral modalities

    Be cautious to modify attribute directly in NeuralData, 
    array data should be immutable and are shared across nodes at different hierarchy levels.
    """
    # Behavior modalities
    position: Optional[Events | TimeSeries] = None
    locomotion: Optional[Events | TimeSeries] = None
    lick: Optional[Events | TimeSeries] = None
    pupil: Optional[Events | TimeSeries] = None
    tongue: Optional[Events | TimeSeries] = None
    whisker: Optional[Events | TimeSeries] = None

    # Fluorescence
    fluorescence: Optional[Fluorescence] = None

    # Timeline
    timeline: Optional[Timeline] = None

    def __post_init__(self):
        """Validate data types."""
        assert all(isinstance(getattr(self, name), Events | type(None)) for name in ("position", "locomotion", "lick")), \
            f"position, locomotion, lick should be Events or None, got {self.position}, {self.locomotion}, {self.lick}"
        assert all(isinstance(getattr(self, name), TimeSeries | type(None)) for name in ("pupil", "tongue", "whisker")), \
            f"pupil, tongue, whisker should be TimeSeries or None, got {self.pupil}, {self.tongue}, {self.whisker}"
    
    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation."""
        summary_str = "NeuralData:\n"
        empty_flag = True
        for name, value in self.__dict__.items():
            if value is not None:
                summary_str += f"  {name}: {value}\n"
                empty_flag = False
        if empty_flag:
            summary_str += "  EMPTY\n"
        return summary_str
    
    def shadow_clone(self, **kwargs) -> "NeuralData":
        """Create a shadow clone with new data."""
        assert all(name in self.__dict__ for name in kwargs.keys()), \
            f"NeuralData expects keys in {self.__dict__.keys()}, got {kwargs.keys()}"
        new_neural_data_dict = {name: value for name, value in self.__dict__.items() if name not in kwargs}
        new_neural_data_dict.update(kwargs)
        return NeuralData(**new_neural_data_dict)    

    def segment(self, start_t: float, end_t: float) -> "NeuralData":
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        assert end_t >= start_t, f"start time {start_t} should be earlier than end time {end_t}"
        new_neural_data_dict = {}
        for name, value in self.__dict__.items():
            if value is not None:
                if hasattr(value, 'segment'):
                    new_neural_data_dict[name] = value.segment(start_t, end_t)
                else:
                    raise NotImplementedError(f"Cannot segment {name} of type {type(value)}")
        return NeuralData(**new_neural_data_dict)

    def status(self) -> Dict[str, bool]:
        """Return status of neural data."""
        status_dict = {}
        for name, value in self.__dict__.items():
            if value is not None:
                status_dict[name] = True
            else:
                status_dict[name] = False
        return status_dict

    def aligned_to(self, align_time: float) -> "NeuralData":
        """Align all data to a specific time point."""
        new_neural_data_dict = {}
        for name, value in self.__dict__.items():
            if value is not None:
                if hasattr(value, 'aligned_to'):
                    new_neural_data_dict[name] = value.aligned_to(align_time)
                else:
                    raise NotImplementedError(f"Cannot align {name} of type {type(value)}")
        return NeuralData(**new_neural_data_dict)
    