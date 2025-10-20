
from typing import Generator, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import seaborn as sns

from kitchen.structure.hierarchical_data_structure import DataSet, Node


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
        _legend: bool = True,
):
    """Trace view of all nodes in the dataset"""

    if _remove_topright_spines:
        ax.spines[['right', 'top']].set_visible(False)
    
    if plotting_settings is None:
        plotting_settings = {}

    if _yerr_bar:
        err_kw = {"errorbar": "se", "err_style": "band", "err_kws": {"lw": 0,}}
    else:
        err_kw = {"errorbar": None,}
        
    for group_idx, (group_name, dataset) in enumerate(group_of_datasets.items()):
        group_settings = {"color": f"C{group_idx}",} | plotting_settings.get(group_name, {})
        x_values = np.array([x_axis_func(node) for node in dataset])
        y_values = np.array([y_axis_func(node) for node in dataset])
        df = pd.DataFrame({"x": x_values, "y": y_values})
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

    if _legend:
        ax.legend(frameon=False, loc='best')
    