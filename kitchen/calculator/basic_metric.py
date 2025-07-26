"""
Basic metric calculation functions.
"""
from typing import Tuple
import numpy as np

from kitchen.structure.neural_data_structure import TimeSeries, Events

def AVERAGE_VALUE(data: TimeSeries | Events, segment_period: Tuple[float, float]) -> float:
    """
    Calculate average value over a specified time segment.

    For TimeSeries data, computes the mean of values in the segment.
    For Events data, computes rate (count/duration) for numeric values
    or frequency (count/duration) for non-numeric values.

    Args:
        data: TimeSeries or Events data to analyze
        segment_period: Tuple of (start_time, end_time) defining the analysis window

    Returns:
        Average value or rate over the specified segment
    """
    segment_duration = segment_period[1] - segment_period[0]
    assert segment_duration > 0, f"segment_duration {segment_duration} should be positive"
    
    if isinstance(data, TimeSeries):
        return float(np.nanmean(data.segment(*segment_period).v))
    elif np.issubdtype(data.v.dtype, np.number):
        return float(np.nansum(data.segment(*segment_period).v) / segment_duration)
    else:
        return float(len(data.segment(*segment_period)) / segment_duration)


def PEAK_VALUE(data: TimeSeries | Events, segment_period: Tuple[float, float]) -> float:
    """
    Calculate peak (maximum) value over a specified time segment.

    Computes the maximum value within the specified time window.
    Currently only implemented for TimeSeries data.

    Args:
        data: TimeSeries or Events data to analyze
        segment_period: Tuple of (start_time, end_time) defining the analysis window

    Returns:
        Maximum value in the specified segment
    """
    if isinstance(data, TimeSeries):
        return float(np.nanmax(data.segment(*segment_period).v))
    else:
        raise NotImplementedError(f"Cannot calculate peak value for {type(data)}")