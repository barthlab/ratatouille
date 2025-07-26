import matplotlib.pyplot as plt

from kitchen.operator.grouping import AdvacedTimeSeries
from kitchen.structure.neural_data_structure import TimeSeries


def oreo_plot(ax: plt.Axes, group_tuple: AdvacedTimeSeries, y_offset: float, ratio: float, trace_style: dict, fill_between_style: dict, **kwargs):
    """plot oreo: trace + fill_between"""
    ax.plot(group_tuple.t, group_tuple.mean * ratio + y_offset, **trace_style, **kwargs)
    # fucking stupid pylance stuck here
    ax.fill_between(group_tuple.t,  
                    (group_tuple.mean - group_tuple.variance) * ratio + y_offset,  # type: ignore
                    (group_tuple.mean + group_tuple.variance) * ratio + y_offset, color=trace_style["color"], **fill_between_style)  # type: ignore


def sushi_plot(ax: plt.Axes, f1: TimeSeries | AdvacedTimeSeries, f2: TimeSeries | AdvacedTimeSeries, 
               y_offset: float, ratio: float, fill_between_style: dict):
    cell_trace_diff = f2 - f1
    # fucking stupid pylance stuck here
    ax.fill_between(x=cell_trace_diff.t, y1=cell_trace_diff.v * ratio + y_offset,  # type: ignore 
                    y2=y_offset, **fill_between_style)