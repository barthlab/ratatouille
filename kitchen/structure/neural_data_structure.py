import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterator, List, Optional, Any, Self, Tuple, Iterable
from functools import cached_property
from collections import defaultdict
from scipy import signal
import logging

from kitchen.settings.potential import COMPONENTS_BANDWIDTH, MAXIMAL_BANDWIDTH, SPIKE_MAX_WINDOW_LEN, SPIKE_MIN_DISTANCE, SPIKE_MIN_HEIGHT_STD_RATIO, SPIKE_MIN_PROMINENCE_STD_RATIO, SPIKES_BANDWIDTH, STD_SLIDING_WINDOW, GCaMP_kernel
from kitchen.settings.timeline import SUPPORTED_TIMELINE_EVENT, TRIAL_ALIGN_EVENT_DEFAULT
from kitchen.settings.fluorescence import DEFAULT_RECORDING_DURATION, DF_F0_RANGE, TRIAL_DF_F0_WINDOW
from kitchen.settings.trials import BOUT_FILTER_PARAMETERS
from kitchen.utils.numpy_kit import sliding_std, smart_interp, zscore
from kitchen.utils.pass_filter import high_pass, low_pass

logger = logging.getLogger(__name__)


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
    
    def __bool__(self):
        """Return True if non-empty."""
        return len(self) > 0
    
    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation."""
        if len(self.t) == 0:
            return "Empty TimeSeries"
        return f"TimeSeries with {len(self.t)} frames from {self.t[0]:.2f} to {self.t[-1]:.2f}"

    def __add__(self, other: "TimeSeries") -> "TimeSeries":
        """Add two time series."""
        if not isinstance(other, self.__class__):
            return NotImplemented

        new_t = np.union1d(self.t, other.t)
        self_v = smart_interp(new_t, self.t, self.v)
        other_v = smart_interp(new_t, other.t, other.v)

        return TimeSeries(t=new_t, v=self_v + other_v)
    
    def __radd__(self, other: "TimeSeries") -> "TimeSeries":
        """Add two time series."""
        if other == 0:
            return self
        return self.__add__(other)
    
    def __sub__(self, other: "TimeSeries") -> "TimeSeries":
        """Subtract two time series."""
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self + (-other)

    def __neg__(self) -> Self:
        """Return a new TimeSeries with negated values."""
        return self.__class__(v=-self.v, t=self.t)

    def segment(self, start_t: float, end_t: float) -> Self:
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        assert end_t >= start_t, f"start time {start_t} should be earlier than end time {end_t}"
        segment_start_index = np.searchsorted(self.t, start_t)
        segment_end_index = np.searchsorted(self.t, end_t)
        return self.__class__(
            v=self.v[..., segment_start_index: segment_end_index],
            t=self.t[segment_start_index: segment_end_index]
        )
    
    def batch_segment(self, ts: Iterable[float], segment_range: Tuple[float, float], _auto_align: bool = True) -> List[Self]:
        """Extract temporal segment for each align time in ts."""        
        return [self.segment(t + segment_range[0], t + segment_range[1]).aligned_to(t) if _auto_align else \
                self.segment(t + segment_range[0], t + segment_range[1]) 
                for t in ts]

    @cached_property
    def fs(self) -> float:
        """Return sampling frequency."""
        if len(self) < 2:
            return np.nan
        return float(1 / np.mean(np.diff(self.t)))
    
    def aligned_to(self, align_time: float) -> Self:
        """Align time series to a specific time point."""
        return self.__class__(v=self.v.copy(), t=self.t - align_time)
    
    def threshold(self, threshold: float, up_crossing_event: Any = None, down_crossing_event: Any = None) -> "Events":
        """
        Convert time series to events based on threshold crossing.
        
        Detects threshold crossing events and assigns up_crossing_event when crossing above threshold
        and down_crossing_event when crossing below threshold.
        """
        assert self.v.ndim == 1, f"values should be 1-d array, got {self.v.shape}"
        
        if len(self.v) < 2:
            return Events(v=np.array([]), t=np.array([]))
        
        sign_diff = np.diff(np.sign(self.v - threshold))
        crossing_indices = np.where(sign_diff != 0)[0]
        
        if len(crossing_indices) == 0:
            return Events(v=np.array([]), t=np.array([]))
        
        events_v, events_t = [], []
        for i in crossing_indices:
            if sign_diff[i] > 0 and up_crossing_event is not None:
                events_v.append(up_crossing_event)
                events_t.append(self.t[i + 1])
            elif sign_diff[i] < 0 and down_crossing_event is not None:
                events_v.append(down_crossing_event)
                events_t.append(self.t[i + 1])
        return Events(v=np.array(events_v), t=np.array(events_t))
    

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
            assert np.all(np.diff(self.t) >= 0), f"times should be in ascending order, got {self.t}"
    
    def __len__(self):
        """Return number of time points."""
        return len(self.t)

    def __bool__(self):
        """Return True if non-empty."""
        return len(self) > 0
    
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

    def __add__(self, other: "Events") -> Self:
        """Concatenate two events with same value type."""
        assert np.can_cast(self.v.dtype, other.v.dtype, 'same_kind') or \
            np.can_cast(other.v.dtype, self.v.dtype, 'same_kind'), \
                f"Cannot concatenate events with incompatible value types {self.v.dtype} and {other.v.dtype}"
        new_t = np.concatenate([self.t, other.t])
        new_v = np.concatenate([self.v, other.v])
        # sort t and v together
        sort_idx = np.argsort(new_t)
        return self.__class__(v=new_v[sort_idx], t=new_t[sort_idx])
    
    def __radd__(self, other: "Events") -> Self:
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

    def bout_filter(self, bout_sticky_window: float, minimal_bout_size: int = 1) -> Tuple[Self, Self]:  # bout_start, bout_end
        """Filter out events within bout_stick_window seconds of previous events."""
        assert bout_sticky_window > 0, "sticky bout window should be positive"
        if len(self.t) == 0:
            return self.__class__(v=np.array([]), t=np.array([])), self.__class__(v=np.array([]), t=np.array([]))

        time_diffs = np.diff(self.t)
        split_indices = np.where(time_diffs >= bout_sticky_window)[0] + 1
        start_indices = np.concatenate(([0], split_indices))
        end_indices = np.concatenate((split_indices - 1, [len(self.t) - 1]))

        # filter out small bouts
        if minimal_bout_size > 1:
            bout_sizes = end_indices - start_indices + 1
            mask = bout_sizes >= minimal_bout_size
            start_indices = start_indices[mask]
            end_indices = end_indices[mask]

        bout_start = self.__class__(v=self.v[start_indices], t=self.t[start_indices])
        bout_end = self.__class__(v=self.v[end_indices], t=self.t[end_indices])
        return bout_start, bout_end

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

    def aligned_to(self, align_time: float) -> Self:
        """Align events to a specific time point."""
        return self.__class__(v=self.v.copy(), t=self.t - align_time)

    def to_timeline(self) -> "Timeline":
        """
        Convert Events to Timeline with dtype validation.
        """
        if len(self) == 0:
            return Timeline(v=np.array([]), t=np.array([]))
        return Timeline(v=self.v.copy(), t=self.t.copy())

    def update_value(self, supplementary_events: "Events"):
        """
        Update values of events at specified time points using another Events object.
        """
        # Validate that all times in supplementary_events exist in self.t
        indices = np.searchsorted(self.t, supplementary_events.t)
        valid_mask = (indices < len(self.t)) & (self.t[indices] == supplementary_events.t)
        if not np.all(valid_mask):
            raise ValueError(f"Cannot update values, got {self.t=} and {supplementary_events.t=}")
    
        # Update values
        update_map = {t: v for t, v in supplementary_events}
        new_v = np.array([update_map.get(t, v) for t, v in zip(self.t, self.v)])
        self.v = new_v

    def mask(self, mask: np.ndarray) -> Self:
        """Apply boolean mask to events."""
        assert mask.shape == self.t.shape, f"mask shape {mask.shape} should match event shape {self.t.shape}"
        return self.__class__(v=self.v[mask], t=self.t[mask])
    
    def groupby(self) -> Generator[Tuple[Any, np.ndarray], None, None]:
        """Group by value."""
        groupby_dict = defaultdict(list)
        for t, v in zip(self.t, self.v):
            groupby_dict[v].append(t)
        for v, ts in groupby_dict.items():
            yield v, np.array(ts)
    
    def filter(self, event_types: Any | Iterable[Any]) -> Self:
        """Extract events of specific types."""
        if not isinstance(event_types, list):
            event_types = [event_types]
        return self.__class__(
            v=np.array([x for x in self.v if x in event_types]),
            t=np.array([t for x, t in zip(self.v, self.t) if x in event_types])
        )


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

    def advanced_filter(self, list_of_event_types: Iterable[str | Iterable[str]]) -> "Timeline":
        """Extract events of specific types."""
        for event_types in list_of_event_types:
            if len(self.filter(event_types)) > 0:
                return self.filter(event_types)
        return Timeline(v=np.array([]), t=np.array([]))

    def task_time(self) -> Tuple[float, float]:
        """Return start and end time of task."""            
        task_start = self.filter("task start").t[0] if "task start" in self.v else 0
        task_end = self.filter("task end").t[0] if "task end" in self.v else DEFAULT_RECORDING_DURATION
        return task_start, task_end



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
    
    def __len__(self) -> int:
        """Return number of time points."""
        return self.num_timepoint
    
    def __bool__(self) -> bool:
        """Return True if non-empty."""
        return len(self) > 0

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
    def z_score(self) -> TimeSeries:
        """ Normalize fluorescence to z-score. """
        return TimeSeries(v=zscore(self.raw_f.v, axis=-1), t=self.raw_f.t)
    
    @cached_property
    def df_f0(self, clip: bool = True) -> TimeSeries:
        """Compute dF/F0."""
        baseline = np.mean(self.raw_f.segment(*TRIAL_DF_F0_WINDOW).v, axis=-1, keepdims=True)
        df_f0 = (self.raw_f.v - baseline) / baseline
        if clip:
            df_f0[(df_f0 < DF_F0_RANGE[0]) | (df_f0 > DF_F0_RANGE[1])] = np.nan
        return TimeSeries(v= df_f0, t=self.raw_f.t)
    
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
class Potential:
    vm: TimeSeries
    spikes: Events = field(default_factory=lambda: Events(v=np.array([]), t=np.array([])))
    _components: Dict[float | str, TimeSeries] = field(default_factory=dict)
    is_prime: bool = False

    def __post_init__(self):
        assert self.vm.v.ndim == 1, f"vm should be 1-d array for one single cell, got {self.vm.v.shape} shape"
    
    @classmethod
    def create_master(cls, vm: TimeSeries) -> "Potential":
        """Create a master potential, will initiate component and spike computation."""
        dummy_potential = cls(vm=vm, is_prime=True)
        for cutoff in COMPONENTS_BANDWIDTH:
            dummy_potential.hp_component(cutoff)  
        dummy_potential._compute_spikes()
        return dummy_potential
    
    @classmethod
    def create_slave(cls, vm: TimeSeries, spikes: Events, components: Dict[float | str, TimeSeries]) -> "Potential":
        """Create a slave potential, will not initiate component and spike computation."""
        return cls(vm=vm, spikes=spikes, _components=components, is_prime=False)
  
    def _compute_spikes(self) -> Events:
        """Compute spikes from membrane potential."""
        if not self.spikes and self.is_prime:
            spike_component = self.hp_component(SPIKES_BANDWIDTH)
            detrend_vm = sliding_std(spike_component.v, window_len=int(STD_SLIDING_WINDOW * self.vm.fs))
            peak_indices, _ = signal.find_peaks(
                x=spike_component.v,
                distance=SPIKE_MIN_DISTANCE * self.vm.fs,
                height=(detrend_vm * SPIKE_MIN_HEIGHT_STD_RATIO, None),
                prominence=(detrend_vm * SPIKE_MIN_PROMINENCE_STD_RATIO, None),
                wlen=int(SPIKE_MAX_WINDOW_LEN * self.vm.fs)
            )
            self.spikes = Events(v=np.array(['spike'] * len(peak_indices)), t=self.vm.t[peak_indices])
        if not self.spikes:
            raise ValueError(f"spikes not computed: {self}")
        return self.spikes

    def _high_pass_vm(self, cutoff: float) -> TimeSeries:
        """Apply high-pass filter to membrane potential."""
        hp_v = high_pass(self.vm.t, self.vm.v, cutoff=cutoff, fs=self.vm.fs)
        hp_v = low_pass(self.vm.t, hp_v, cutoff=MAXIMAL_BANDWIDTH, fs=self.vm.fs)
        return TimeSeries(v=hp_v, t=self.vm.t)

    def hp_component(self, cutoff: float) -> TimeSeries:
        """Get high-pass filtered component."""
        if cutoff not in self._components and self.is_prime:
            self._components[cutoff] = self._high_pass_vm(cutoff)
        if cutoff not in self._components:
            raise ValueError(f"Cutoff {cutoff} not found in components: {self._components.keys()}")
        return self._components[cutoff]
    
    def _conv_GCaMP6f(self, downsample_ratio: int = 100) -> TimeSeries:
        """Convolve with GCaMP6f kernel."""
        if "conv" not in self._components and self.is_prime:                  
            downsampled_t = self.vm.t[::downsample_ratio]
            spike_indices = np.searchsorted(downsampled_t, self.spikes.t)
            spike_1hot = np.zeros_like(downsampled_t)
            spike_1hot[spike_indices] += 1
            gcamp_kernel, kernel_t = GCaMP_kernel(self.vm.fs / downsample_ratio)
            putative_df_f0 = np.convolve(spike_1hot, gcamp_kernel, mode='full')[:len(downsampled_t)]
            self._components["conv"] = TimeSeries(v=putative_df_f0, t=downsampled_t)
        if "conv" not in self._components:
            raise ValueError(f"conv not found in components: {self._components.keys()}")        
        return self._components["conv"]

    def aspect(self, keyword: Optional[Any] = None) -> TimeSeries:
        """Get potential component based on keyword."""
        if keyword is None:
            return self.hp_component(SPIKES_BANDWIDTH)
        elif isinstance(keyword, (int, float)):
            return self.hp_component(keyword)
        elif keyword == "conv":
            return self._conv_GCaMP6f()
        else:
            return self.vm
        
    @cached_property
    def num_timepoint(self):
        return len(self.vm.t)
    
    @cached_property
    def num_spikes(self):
        return len(self.spikes)

    @cached_property
    def num_components(self):
        return len(self._components)

    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"Potential with {self.num_spikes} spikes and {self.num_components} components: {self._components.keys()}"

    def __len__(self) -> int:
        """Return number of time points."""
        return self.num_timepoint
    
    def __bool__(self) -> bool:
        """Return True if non-empty."""
        return len(self) > 0
    
    def segment(self, start_t: float, end_t: float) -> "Potential":
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        return Potential.create_slave(
            vm=self.vm.segment(start_t, end_t),
            spikes=self.spikes.segment(start_t, end_t),
            components={Keyword: _comp.segment(start_t, end_t) for Keyword, _comp in self._components.items()},
        )
    
    def aligned_to(self, align_time: float) -> "Potential":
        """Align potential to a specific time point."""
        return Potential.create_slave(
            vm=self.vm.aligned_to(align_time),
            spikes=self.spikes.aligned_to(align_time),
            components={Keyword: _comp.aligned_to(align_time) for Keyword, _comp in self._components.items()},
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
    position: Optional[Events] = None
    locomotion: Optional[Events] = None
    lick: Optional[Events] = None
    pupil: Optional[TimeSeries] = None
    tongue: Optional[TimeSeries] = None
    whisker: Optional[TimeSeries] = None

    # Potential
    potential: Optional[Potential] = None

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
    
    def iterablize(self, data_name: str, *args) -> Iterator[float]:
        assert getattr(self, data_name) is not None, f"Cannot iterate over None {data_name}"
        if data_name == "timeline":
            assert self.timeline is not None, "Cannot iterate over None timeline"
            for trial_t in self.timeline.advanced_filter(TRIAL_ALIGN_EVENT_DEFAULT).t:
                yield trial_t
        elif data_name in ["locomotion", "lick"]:
            bout_filter_parameters = BOUT_FILTER_PARAMETERS[data_name]
            bout_start, bout_end = getattr(self, data_name).bout_filter(**bout_filter_parameters)
            trial_times = bout_start if "onset" in args else bout_end
            for bout_t in trial_times.t:
                yield bout_t
        else:
            raise NotImplementedError(f"Cannot iterate over {data_name}")