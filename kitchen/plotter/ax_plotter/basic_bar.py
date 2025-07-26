import matplotlib.pyplot as plt
import numpy as np
from typing import Generator, List, Callable

from kitchen.plotter.plotting_params import ANNOTATION_BBOX_HEIGHT_FACTOR, ANNOTATION_OFFSET_FACTOR, RATIO_DICT
from kitchen.plotter.unit_plotter.unit_bar import unit_bar, unit_comparison_annotation_ttest_ind, unit_comparison_annotation_ttest_rel, unit_swarm
from kitchen.plotter.unit_plotter.unit_yticks import yticks_combo
from kitchen.structure.hierarchical_data_structure import DataSet

def pairwise_comparison_bar(
        ax: plt.Axes,
        datasets: List[DataSet],

        metric_funcs: List[Callable[[DataSet], np.ndarray]],
        bar_styles: List[dict],
        comparison_method: str,
        data_name: str,
        offset_order: int = 1,
) -> Generator[float, float, None]:
    assert len(datasets) == len(metric_funcs) == len(bar_styles), \
        f"Number of datasets, metric functions and bar styles must match, got {len(datasets)}, {len(metric_funcs)}, {len(bar_styles)}"
    
    # skip the previous offset_order of yield
    y_offset = 0
    for _ in range(offset_order):
        y_offset = yield 0

    # calculate metric
    ratio = RATIO_DICT[data_name]
    bar_data = [metric_func(dataset) for dataset, metric_func in zip(datasets, metric_funcs)]

    # plot bar
    bar_means = []
    for i, (bar_values, bar_style) in enumerate(zip(bar_data, bar_styles)):
        bar_mean = unit_bar(bar_data=(i, bar_values), ax=ax, bar_styles={"label": f"Group {i}"} | bar_style, 
                            y_offset=y_offset, ratio=ratio)
        unit_swarm(bar_data=(i, bar_values), ax=ax, y_offset=y_offset, ratio=ratio)
        bar_means.append(bar_mean)

    # conduct stats test and annotate
    y_max = float(ax.get_ylim()[1])
    annotation_y_offset = (y_max - np.min(bar_means)) * ANNOTATION_OFFSET_FACTOR

    for i, bar1 in enumerate(bar_data):
        for j, bar2 in enumerate(bar_data):
            if i >= j:
                continue
            y_bbox = (y_max, y_max + annotation_y_offset * ANNOTATION_BBOX_HEIGHT_FACTOR)
            if comparison_method == "ttest_ind":
                unit_comparison_annotation_ttest_ind((i, bar1), (j, bar2), ax, y_bbox, y_offset, ratio)
            elif comparison_method == "ttest_rel":
                unit_comparison_annotation_ttest_rel((i, bar1), (j, bar2), ax, y_bbox, y_offset, ratio)
            else:
                raise ValueError(f"Unknown comparison method: {comparison_method}")
            y_max += annotation_y_offset

    # set x ticks
    ax.set_xticks(np.arange(len(datasets)), 
                  [bar_style['label'] if 'label' in bar_style else f"Group {i}" 
                   for i, bar_style in enumerate(bar_styles)])
    
    # set y ticks
    yticks_combo(data_name, ax, y_offset, ratio)
    
    yield 0