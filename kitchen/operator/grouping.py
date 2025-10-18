from dataclasses import dataclass
from typing import List, Self, Tuple
import numpy as np
from scipy import stats

from kitchen.structure.neural_data_structure import Events, TimeSeries
from kitchen.utils.numpy_kit import smart_interp



@dataclass
class AdvancedTimeSeries(TimeSeries):
    """
    A TimeSeries that models a value as a distribution with a mean and variance.

    It inherits from TimeSeries, using the `v` attribute to store the mean value
    and adding a `variance` attribute. All operations are updated to correctly
    """

    variance: np.ndarray
    raw_array: np.ndarray

    @classmethod
    def create_empty(cls) -> Self:
        return cls(v=np.array([]), t=np.array([]), variance=np.array([]), raw_array=np.empty((0, 0)))
    
    @property
    def mean(self, default = 0.) -> np.ndarray:
        return self.v.copy() if len(self) > 0 else np.array([default])
    
    @property
    def var(self) -> np.ndarray:
        return self.variance.copy()

    @property
    def raw(self) -> np.ndarray:
        return self.raw_array.copy()
    
    @property
    def data_num(self) -> int:
        return len(self.raw)
    
    @property
    def data_shape(self) -> tuple:
        return self.raw.shape[1:]

    def __post_init__(self):
        """Validate data integrity, including the variance."""
        super().__post_init__()
        if self.variance.shape != self.v.shape:
            raise ValueError(
                f"Shape of variance {self.variance.shape} must match "
                f"shape of mean (v) {self.v.shape}."
            )
        if np.any(self.variance < 0):
            raise ValueError("Variance cannot be negative.")
        if self.data_shape != self.v.shape:
            raise ValueError(
                f"Shape of raw array {self.raw.shape} should match "
                f"shape of mean (v) {self.v.shape}."
            )
    
    def __neg__(self) -> Self:
        return self.__class__(
            t=self.t.copy(), v=-self.v.copy(), variance=self.variance.copy(), raw_array=-self.raw.copy()
        )

    def segment(self, start_t: float, end_t: float) -> Self:
        raise NotImplementedError("segment is not allowed for AdvancedTimeSeries")

    def batch_segment(self, ts: np.ndarray, segment_range: Tuple[float, float]):
        raise NotImplementedError("batch_segment is not allowed for AdvancedTimeSeries")
    
    def aligned_to(self, align_time: float) -> Self:
        return self.__class__(
            v=self.v.copy(),
            t=self.t - align_time,
            variance=self.variance.copy(),
            raw_array=self.raw.copy()
        )

    def squeeze(self, axis: int) -> Self:
        return self.__class__(
            v=np.squeeze(self.v, axis=axis),
            t=self.t,
            variance=np.squeeze(self.variance, axis=axis),
            raw_array=np.squeeze(self.raw, axis=axis+1)
        )

    @property
    def mean_ts(self) -> TimeSeries:
        return TimeSeries(v=self.v.copy(), t=self.t)


def calculate_group_tuple(arrs: List[np.ndarray], t: np.ndarray) -> AdvancedTimeSeries:
    """Group a list of arrays in mean and variance."""
    assert len(arrs) > 0, "Cannot calculate on empty list"
    assert all(arr.shape == arrs[0].shape for arr in arrs), "All arrays should have same shape"
    assert len(t) == arrs[0].shape[-1], f"Time array should have same length as array shape (last dim), got {len(t)} vs {arrs[0].shape}"
    means = np.nanmean(arrs, axis=0)
    variances = stats.sem(arrs, axis=0, nan_policy="omit") if len(arrs) > 1 else np.zeros_like(means)
    return AdvancedTimeSeries(v=means, t=t, variance=variances, raw_array=np.array(arrs))


def grouping_events_rate(events: List[Events], bin_size: float, use_event_value_as_weight: bool = True) -> AdvancedTimeSeries:
    """Group a list of events in rate."""
    assert bin_size > 0, "bin size should be positive"
    if len(events) == 0:
        return AdvancedTimeSeries.create_empty()
    group_t = np.sort(np.concatenate([event.t for event in events]))
    # Handle case where all events are empty (no spikes across all trials)
    if len(group_t) == 0:
        return AdvancedTimeSeries.create_empty()
    bins = np.arange(group_t[0] - bin_size, group_t[-1] + bin_size, bin_size)
    all_rates = [np.histogram(event.t, bins=bins, weights=event.v
                              if use_event_value_as_weight else None)[0] / bin_size for event in events]
    return calculate_group_tuple(all_rates, bins[:-1] + bin_size/2)


def grouping_events_histogram(events: List[Events], bins: np.ndarray, use_event_value_as_weight: bool = True) -> AdvancedTimeSeries:
    all_rates = [np.histogram(event.t, bins=bins, weights=event.v 
                              if use_event_value_as_weight else None)[0] / (bins[1:] - bins[:-1]) for event in events]
    return calculate_group_tuple(all_rates, (bins[:-1] + bins[1:]) / 2)


def grouping_timeseries(timeseries: List[TimeSeries], scale_factor: float = 2, interp_method: str = "previous") -> AdvancedTimeSeries:
    """Group a list of timeseries."""
    min_t, max_t =max(timeseries, key=lambda x: x.t[0]).t[0], min(timeseries, key=lambda x: x.t[-1]).t[-1]
    max_fs = max(timeseries, key=lambda x: x.fs).fs
    group_t = np.linspace(min_t, max_t, int((max_t - min_t) * max_fs * scale_factor))
    all_values = [smart_interp(group_t, ts.t, ts.v, interp_method) for ts in timeseries]
    return calculate_group_tuple(all_values, group_t)

