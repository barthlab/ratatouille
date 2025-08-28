import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional, Any, Self, Tuple, Iterable
from functools import cached_property
from scipy import signal

from kitchen.settings.potential import COMPONENTS_BANDWIDTH, SPIKE_MAX_WINDOW_LEN, SPIKE_MIN_DISTANCE, SPIKE_MIN_HEIGHT_STD_RATIO, SPIKES_BANDWIDTH, STD_SLIDING_WINDOW
from kitchen.settings.timeline import SUPPORTED_TIMELINE_EVENT
from kitchen.settings.fluorescence import DEFAULT_RECORDING_DURATION, DF_F0_RANGE, TRIAL_DF_F0_WINDOW
from kitchen.utils.numpy_kit import sliding_std, smart_interp
from kitchen.utils.pass_filter import high_pass


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

    def segment(self, start_t: float, end_t: float) -> "TimeSeries":
        """Extract temporal segment between start_t (inclusive) and end_t (exclusive)."""
        assert end_t >= start_t, f"start time {start_t} should be earlier than end time {end_t}"
        segment_start_index = np.searchsorted(self.t, start_t)
        segment_end_index = np.searchsorted(self.t, end_t)
        return self.__class__(
            v=self.v[..., segment_start_index: segment_end_index],
            t=self.t[segment_start_index: segment_end_index]
        )
    
    @cached_property
    def fs(self) -> float:
        """Return sampling frequency."""
        if len(self) < 2:
            return np.nan
        return float(1 / np.mean(np.diff(self.t)))
    
    def aligned_to(self, align_time: float) -> "TimeSeries":
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

    def inactivation_window_filter(self, inactivation_window: float) -> Self:
        """Filter out events within inactivation_window seconds of previous events."""
        assert inactivation_window > 0, "inactivation window should be positive"
        if len(self.t) == 0:
            return self.__class__(v=np.array([]), t=np.array([]))

        new_v, new_t = [self.v[0]], [self.t[0]]
        for i in range(1, len(self.t)):
            if self.t[i] - self.t[i-1] >= inactivation_window:
                new_v.append(self.v[i])
                new_t.append(self.t[i])
        return self.__class__(v=np.array(new_v), t=np.array(new_t))

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
        baseline = np.mean(self.raw_f.v, axis=-1, keepdims=True)
        std = np.std(self.raw_f.v, axis=-1, keepdims=True)
        return TimeSeries(v=(self.raw_f.v - baseline) / std , t=self.raw_f.t)
    
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
    _hp_components: Dict[float, TimeSeries] = field(default_factory=dict)
    is_prime: bool = False

    def __post_init__(self):
        assert self.vm.v.ndim == 1, f"vm should be 1-d array for one single cell, got {self.vm.v.shape} shape"
    
    @classmethod
    def create_master(cls, vm: TimeSeries) -> "Potential":
        """Create a master potential, will initiate component and spike computation."""
        dummy_potential = cls(vm=vm, is_prime=True)
        dummy_hp_components = {cutoff: dummy_potential._high_pass_vm(cutoff) for cutoff in COMPONENTS_BANDWIDTH}
        dummy_spikes = dummy_potential._compute_spikes()
        return cls(vm=vm, spikes=dummy_spikes, _hp_components=dummy_hp_components, is_prime=True)
    
    @classmethod
    def create_slave(cls, vm: TimeSeries, spikes: Events, hp_components: Dict[float, TimeSeries]) -> "Potential":
        """Create a slave potential, will not initiate component and spike computation."""
        return cls(vm=vm, spikes=spikes, _hp_components=hp_components, is_prime=False)
  
    def _compute_spikes(self) -> Events:
        """Compute spikes from membrane potential."""
        if self.spikes:
            return self.spikes
        spike_component = self.hp_component(SPIKES_BANDWIDTH)
        detrend_vm = sliding_std(spike_component.v, window_len=int(STD_SLIDING_WINDOW * self.vm.fs))
        peak_indices, _ = signal.find_peaks(
             x=spike_component.v,
             distance=SPIKE_MIN_DISTANCE * self.vm.fs,
             height=(detrend_vm * SPIKE_MIN_HEIGHT_STD_RATIO, None),
             wlen=int(SPIKE_MAX_WINDOW_LEN * self.vm.fs)
        )
        return Events(v=np.array(['spike'] * len(peak_indices)), t=self.vm.t[peak_indices])

    def _high_pass_vm(self, cutoff: float) -> TimeSeries:
        """Apply high-pass filter to membrane potential."""
        hp_v = high_pass(self.vm.t, self.vm.v, fs=self.vm.fs, cutoff=cutoff)
        return TimeSeries(v=hp_v, t=self.vm.t)

    def hp_component(self, cutoff: float) -> TimeSeries:
        """Get high-pass filtered component."""
        if cutoff not in self._hp_components and self.is_prime:
            self._hp_components[cutoff] = self._high_pass_vm(cutoff)
        if cutoff not in self._hp_components:
            raise ValueError(f"Cutoff {cutoff} not found in hp_components: {self._hp_components.keys()}")
        return self._hp_components[cutoff]

    @cached_property
    def num_timepoint(self):
        return len(self.vm.t)
    
    @cached_property
    def num_spikes(self):
        return len(self.spikes)

    @cached_property
    def num_hp_components(self):
        return len(self._hp_components)

    def __repr__(self):
        """Return string representation."""
        return self.__str__()
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"Potential with {self.num_spikes} spikes and {self.num_hp_components} high-pass components"

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
            hp_components={cutoff: hp_comp.segment(start_t, end_t) for cutoff, hp_comp in self._hp_components.items()},
        )
    
    def aligned_to(self, align_time: float) -> "Potential":
        """Align potential to a specific time point."""
        return Potential.create_slave(
            vm=self.vm.aligned_to(align_time),
            spikes=self.spikes.aligned_to(align_time),
            hp_components={cutoff: hp_comp.aligned_to(align_time) for cutoff, hp_comp in self._hp_components.items()},
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
    