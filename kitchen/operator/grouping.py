from dataclasses import dataclass
from typing import List, Self
import numpy as np
from scipy import stats

from kitchen.structure.neural_data_structure import Events, TimeSeries
from kitchen.utils.numpy_kit import smart_interp



@dataclass
class AdvacedTimeSeries(TimeSeries):
    """
    A TimeSeries that models a value as a distribution with a mean and variance.

    It inherits from TimeSeries, using the `v` attribute to store the mean value
    and adding a `variance` attribute. All operations are updated to correctly
    """

    variance: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.v
    
    @property
    def var(self) -> np.ndarray:
        return self.variance

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

    def __neg__(self) -> Self:
        return self.__class__(
            t=self.t.copy(), v=-self.mean, variance=self.variance.copy()
        )

    def segment(self, start_t: float, end_t: float) -> Self:
        raise NotImplementedError("segment is not allowed for AdvacedTimeSeries")

    def aligned_to(self, align_time: float) -> Self:
        return self.__class__(
            v=self.mean.copy(),
            t=self.t - align_time,
            variance=self.variance.copy(),
        )

    def squeeze(self, axis: int) -> Self:
        return self.__class__(
            v=np.squeeze(self.mean, axis=axis),
            t=self.t,
            variance=np.squeeze(self.variance, axis=axis),
        )


def calculate_group_tuple(arrs: List[np.ndarray], t: np.ndarray) -> AdvacedTimeSeries:
    """Group a list of arrays in mean and variance."""
    assert len(arrs) > 0, "Cannot calculate on empty list"
    assert all(arr.shape == arrs[0].shape for arr in arrs), "All arrays should have same shape"
    assert len(t) == arrs[0].shape[-1], f"Time array should have same length as array shape (last dim), got {len(t)} vs {arrs[0].shape}"
    means = np.nanmean(arrs, axis=0)
    variances = stats.sem(arrs, axis=0, nan_policy="omit")
    return AdvacedTimeSeries(v=means, t=t, variance=variances)


def grouping_events_rate(events: List[Events], bin_size: float) -> AdvacedTimeSeries:
    """Group a list of events in rate."""
    assert bin_size > 0, "bin size should be positive"
    group_t = np.sort(np.concatenate([event.t for event in events]))
    bins = np.arange(group_t[0]-bin_size, group_t[-1] + bin_size, bin_size)
    all_rates = [np.histogram(event.t, bins=bins, weights=event.v)[0] / bin_size for event in events]
    return calculate_group_tuple(all_rates, bins[:-1] + bin_size/2)


def grouping_timeseries(timeseries: List[TimeSeries], scale_factor: float = 2, interp_method: str = "previous") -> AdvacedTimeSeries:
    """Group a list of timeseries."""
    min_t, max_t =max(timeseries, key=lambda x: x.t[0]).t[0], min(timeseries, key=lambda x: x.t[-1]).t[-1]
    max_fs = max(timeseries, key=lambda x: x.fs).fs
    group_t = np.linspace(min_t, max_t, int((max_t - min_t) * max_fs * scale_factor))
    all_values = [smart_interp(group_t, ts.t, ts.v, interp_method) for ts in timeseries]
    return calculate_group_tuple(all_values, group_t)

