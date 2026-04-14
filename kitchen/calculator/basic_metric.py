"""
Basic metric calculation functions.
"""
from typing import Optional, Tuple
import numpy as np

from kitchen.operator import grouping
from kitchen.structure.neural_data_structure import TimeSeries, Events
from kitchen.utils.numpy_kit import smart_interp

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
        interp_t = np.linspace(segment_period[0], segment_period[1], num=int(2*data.fs))
        interp_v = smart_interp(interp_t, data.t, data.v, method="previous")
        return float(np.nanmean(interp_v))
    elif np.issubdtype(data.v.dtype, np.number):
        return float(np.nansum(data.segment(*segment_period).v) / segment_duration)
    else:
        return float(len(data.segment(*segment_period)) / segment_duration)


def AUC_VALUE(data: TimeSeries | Events, segment_period: Tuple[float, float]) -> float:
    """
    Calculate area under the curve (AUC) over a specified time segment.

    For TimeSeries data, computes the integral of values in the segment.
    For Events data, computes total count for numeric values or total occurrences for non-numeric values.

    Args:
        data: TimeSeries or Events data to analyze
        segment_period: Tuple of (start_time, end_time) defining the analysis window
    Returns:
        AUC value or total count over the specified segment
    """
    if isinstance(data, TimeSeries):
        interp_t = np.linspace(segment_period[0], segment_period[1], num=int(2*data.fs))
        interp_v = smart_interp(interp_t, data.t, data.v, method="previous")
        return np.trapezoid(interp_v, interp_t)
    else:
        raise NotImplementedError(f"Cannot calculate AUC for {type(data)}")


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
        interp_t = np.linspace(segment_period[0], segment_period[1], num=int(2*data.fs))
        interp_v = smart_interp(interp_t, data.t, data.v, method="previous")
        return float(np.nanmax(interp_v))
    else:
        raise NotImplementedError(f"Cannot calculate peak value for {type(data)}")
    

def DIFFERENCE(data: TimeSeries | Events, segment_period1: Tuple[float, float], segment_period2: Tuple[float, float]) -> float:
    """
    Calculate difference between two time segments.

    Computes the difference between the average value of two segments.

    Args:
        data: TimeSeries or Events data to analyze
        segment_period1: Tuple of (start_time, end_time) defining the first segment
        segment_period2: Tuple of (start_time, end_time) defining the second segment

    Returns:
        Difference between the average value of the two segments
    """
    return AVERAGE_VALUE(data, segment_period1) - AVERAGE_VALUE(data, segment_period2)


def PEARSON_CORRELATION(data1: TimeSeries, data2: TimeSeries, segment_period: Optional[Tuple[float, float]]=None) -> float:
    """
    Calculate correlation value over a specified time segment.

    Computes the correlation coefficient between two TimeSeries within the specified time window.

    Args:
        data1: First TimeSeries data to analyze
        data2: Second TimeSeries data to analyze
        segment_period: Tuple of (start_time, end_time) defining the analysis window

    Returns:
        Correlation coefficient between the two data series in the specified segment
    """
    segment1 = data1.segment(*segment_period) if segment_period is not None else data1
    segment2 = data2.segment(*segment_period) if segment_period is not None else data2
    corr_t = grouping.get_ts_combine_t([data1, data2])
    data1_interp = smart_interp(corr_t, segment1.t, segment1.v, "linear")
    data2_interp = smart_interp(corr_t, segment2.t, segment2.v, "linear")
    return float(np.corrcoef(data1_interp, data2_interp)[0, 1])