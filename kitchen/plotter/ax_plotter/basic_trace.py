
from collections import defaultdict
from typing import Generator, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import seaborn as sns

from kitchen.plotter import style_dicts
from kitchen.plotter.color_scheme import num_to_color
from kitchen.structure.hierarchical_data_structure import DataSet, Node
from kitchen.utils import sequence_kit


logger = logging.getLogger(__name__)


def trace_view(
        ax: plt.Axes,
        group_of_datasets: dict[str, DataSet],
        y_axis_func: Callable[[Node], float],
        x_axis_func: Callable[[Node], float],

        plotting_settings: Optional[dict[str, dict]] = None,
        _break_at_int: bool = False,
        _remove_topright_spines: bool = True,
        _yerr_bar: bool = True,
        _box: Optional[float] = None,
        _box_color: Callable = lambda x: "gray",
        _legend: bool = True,
):
    """Trace view of all nodes in the dataset"""

    if _remove_topright_spines:
        ax.spines[['right', 'top']].set_visible(False)
    
    if plotting_settings is None:
        plotting_settings = {}

    if _yerr_bar:
        # err_kw = {"errorbar": "se", "err_style": "bars", }
        err_kw = {"errorbar": "se", "err_style": "band", "err_kws": {"lw": 0,}}
    else:
        err_kw = {"errorbar": None,}
    
    all_data = defaultdict(list)
    for group_idx, (group_name, dataset) in enumerate(group_of_datasets.items()):
        group_settings = {"color": num_to_color(group_idx),} | plotting_settings.get(group_name, {})
        x_values = np.array([x_axis_func(node) for node in dataset])
        y_values = np.array([y_axis_func(node) for node in dataset])
        df = pd.DataFrame({"x": x_values, "y": y_values})
        for x_coord, xy_items in sequence_kit.group_by(zip(x_values, y_values), key_func=lambda x: x[0]).items():
            all_data[x_coord].append(np.nanmean([xy_item[1] for xy_item in xy_items]))
            
        df['segments'] = np.floor(df['x']) - 1e-6
        if _break_at_int:
            for segment_idx, segment_part in enumerate(df['segments'].unique()):
                segment_df = df[df['segments'] == segment_part]
                sns.lineplot(data=segment_df, x='x', y='y', ax=ax, 
                             label=group_name if _legend and (segment_idx == 0) else None,
                             **err_kw, **group_settings,
                            )
        else:
            sns.lineplot(data=df, x='x', y='y', ax=ax, 
                        label=group_name if _legend else None,
                        **err_kw, **group_settings, )
    
    if _box and len(group_of_datasets) > 1:
        for x_coord, y_values in all_data.items():
            if len(y_values) < 2:
                continue
            ax.boxplot(
                y_values,
                positions=[x_coord],
                widths=_box,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "none", "edgecolor": _box_color(x_coord), "alpha": 0.5},
                whiskerprops={"color": _box_color(x_coord), "alpha": 0.5},
                capprops={"color": _box_color(x_coord), "alpha": 0.5},
                medianprops={"color": _box_color(x_coord), "alpha": 0.5},
                zorder=10,
            )
    if _legend:
        ax.legend(frameon=False, loc='best', ncol=2, fontsize="x-small",)
    
