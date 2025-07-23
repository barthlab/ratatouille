from collections import namedtuple
from typing import List
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

from kitchen.structure.neural_data_structure import Events, TimeSeries


GROUP_TUPLE = namedtuple("GROUP_TUPLE", ["t", "mean", "variance"])


def calculate_group_tuple(arrs: List[np.ndarray], t: np.ndarray) -> GROUP_TUPLE:
    """Group a list of arrays in mean and variance."""
    assert len(arrs) > 0, "Cannot calculate on empty list"
    assert all(arr.shape == arrs[0].shape for arr in arrs), "All arrays should have same shape"
    assert len(t) == arrs[0].shape[-1], f"Time array should have same length as array shape (last dim), got {len(t)} vs {arrs[0].shape}"
    means = np.nanmean(arrs, axis=0)
    variances = stats.sem(arrs, axis=0, nan_policy="omit")
    return GROUP_TUPLE(t=t, mean=means, variance=variances)


def grouping_events_rate(events: List[Events], bin_size: float) -> GROUP_TUPLE:
    """Group a list of events in rate."""
    assert bin_size > 0, "bin size should be positive"
    group_t = np.sort(np.concatenate([event.t for event in events]))
    bins = np.arange(group_t[0]-bin_size, group_t[-1] + bin_size, bin_size)
    all_rates = [np.histogram(event.t, bins=bins, weights=event.v)[0] / bin_size for event in events]
    return calculate_group_tuple(all_rates, bins[:-1] + bin_size/2)


def smart_interp(x_new, xp, fp, method: str = "previous"):
    if method == "previous":
        f_new = interp1d(xp, fp, kind='previous', axis=-1, bounds_error=False)
    elif method == "nearest":
        f_new = interp1d(xp, fp, kind='nearest', axis=-1, bounds_error=False)
    elif method == "linear":
        f_new = interp1d(xp, fp, kind='linear', axis=-1, bounds_error=False)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    return f_new(x_new)


def grouping_timeseries(timeseries: List[TimeSeries], scale_factor: float = 2, interp_method: str = "previous") -> GROUP_TUPLE:
    """Group a list of timeseries."""
    min_t, max_t =max(timeseries, key=lambda x: x.t[0]).t[0], min(timeseries, key=lambda x: x.t[-1]).t[-1]
    max_fs = max(timeseries, key=lambda x: x.fs).fs
    group_t = np.linspace(min_t, max_t, int((max_t - min_t) * max_fs * scale_factor))
    all_values = [smart_interp(group_t, ts.t, ts.v, interp_method) for ts in timeseries]
    return calculate_group_tuple(all_values, group_t)

