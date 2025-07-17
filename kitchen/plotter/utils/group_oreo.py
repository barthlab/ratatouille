import matplotlib.pyplot as plt

from kitchen.operator.grouping import GROUP_TUPLE


def oreo_plot(ax: plt.Axes, group_tuple: GROUP_TUPLE, y_offset: float, ratio: float, trace_style: dict, fill_between_style: dict):
    """plot oreo: trace + fill_between"""
    ax.plot(group_tuple.t, group_tuple.mean * ratio + y_offset, **trace_style)
    ax.fill_between(group_tuple.t, 
                    (group_tuple.mean - group_tuple.variance) * ratio + y_offset, 
                    (group_tuple.mean + group_tuple.variance) * ratio + y_offset, color=trace_style["color"], **fill_between_style)


