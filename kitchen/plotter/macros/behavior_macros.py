
from functools import partial
from typing import Optional, Tuple, Type, TypeVar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import logging
import traceback

from kitchen.configs import routing
from kitchen.configs.naming import get_node_name
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.ax_plotter.advance_plot import subtract_view
from kitchen.plotter.ax_plotter.basic_plot import stack_view
from kitchen.plotter.ax_plotter.basic_trace import trace_view
from kitchen.plotter.color_scheme import num_to_color, string_to_hex_color
from kitchen.plotter.decorators.default_decorators import default_exit_save, default_plt_param, default_style
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL, PlotManual
from kitchen.plotter.plotting_params import STACK_X_INCHES, STACK_Y_INCHES
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Mice, Node, NodeTypeVar, Session, Trial
from kitchen.utils.sequence_kit import find_only_one, select_from_value

logger = logging.getLogger(__name__)


def trace_view_delta_behavior_macro(
        dataset: DataSet,
        behavior_name: str,
        
        left_day_trial_types: set[str],
        right_day_trial_types: set[str],

        range_compare: tuple[float, float],
        range_baseline: tuple[float, float],

        yaxis_formattor,
        yaxis_label: str,

        prefix_keyword: Optional[str] = None,
        
        _aligment_style: str = "Aligned2Trial",

        _day_offset: float = 0.8,

        _right_day_cover: str = "gray",
):
    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    plot_manual = PlotManual(**{behavior_name: True})
    save_name = f"{dataset.name}_{behavior_name}_delta"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    all_y_group_nodes = dataset.select("mice", _empty_warning=False)
    group_of_dataset = {}
    plotting_settings = {}

    session_id2x_coord_dict = {}
    for y_group_idx, y_group_node in enumerate(all_y_group_nodes):
        print(y_group_node.mice_id)
        # if y_group_node.mice_id in ("QYV5M", "SCE6F"):
        #     continue
        y_group_subtree = dataset.subtree(y_group_node)
        
        left_day2_plot = []
        right_day2_plot = []
        for day_node in y_group_subtree.select("day"):
            type2dataset = split_dataset_by_trial_type(
                y_group_subtree.subtree(day_node), 
                plot_manual=plot_manual,
                _element_trial_level = "trial",)
            if set(type2dataset.keys()) == left_day_trial_types:
                left_day2_plot.append(day_node)
            elif set(type2dataset.keys()) == right_day_trial_types:
                right_day2_plot.append(day_node)

        if len(left_day2_plot) == 0 or len(right_day2_plot) == 0:
            continue
        for day_idx, day_node in enumerate(left_day2_plot):
            major_x_coord = day_idx - len(left_day2_plot)
            all_session_nodes = y_group_subtree.subtree(day_node).select("cellsession")
            for session_idx, session_node in enumerate(all_session_nodes):
                session_id2x_coord_dict[session_node.session_id] = major_x_coord + _day_offset * (session_idx / len(all_session_nodes))
        for day_idx, day_node in enumerate(right_day2_plot):
            major_x_coord = day_idx
            all_session_nodes = y_group_subtree.subtree(day_node).select("cellsession")
            for session_idx, session_node in enumerate(all_session_nodes):
                session_id2x_coord_dict[session_node.session_id] = major_x_coord + _day_offset * (session_idx / len(all_session_nodes))

        for trial_type in left_day_trial_types | right_day_trial_types:
            certain_type_dataset = y_group_subtree.select(
                "trial", _self=lambda x: x.info.get("trial_type") == trial_type)
            group_of_dataset[f"{y_group_node.mice_id}_{trial_type}"] = sync_nodes(certain_type_dataset, alignment_events, plot_manual)
            plotting_settings[f"{y_group_node.mice_id}_{trial_type}"] = {
                "color": num_to_color(y_group_idx),
                "marker": "o" if trial_type not in left_day_trial_types else ".",
                "ls": "-" if trial_type not in left_day_trial_types else "--",
                "alpha": 1 if trial_type not in left_day_trial_types else 0.5,
                "lw": 1.5 if trial_type not in left_day_trial_types else 0.8,
                "zorder": 10/len(certain_type_dataset),
                }


    def y_axis_func(node: Node) -> float:
        behavior_data = getattr(node.data, behavior_name)
        return np.abs(behavior_data.segment(*range_compare).v.mean() - behavior_data.segment(*range_baseline).v.mean())
    
    def x_axis_func(node: Node) -> float:
        return session_id2x_coord_dict.get(node.session_id, np.nan) 

    # plotting
    # default_plt_param()

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    group_of_dataset = {k: v for k, v in sorted(group_of_dataset.items(), key=lambda x: x[0])}
    trace_view(
        ax=ax, group_of_datasets=group_of_dataset, y_axis_func=y_axis_func, x_axis_func=x_axis_func,
        plotting_settings=plotting_settings,
        _yerr_bar=False,
        _break_at_int=True,
    )
    ax.set_xticks([-1.6, -0.6, 0.4, 1.4,], ["day1", "day2", "day1", "day2"])
    # ax.set_ylim(-0.2, 0.3)
    ax.set_ylim(0., 0.5)
    ax.axvspan(-0.2, 2, alpha=0.1, color=_right_day_cover, lw=0, zorder=-10)
    # ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=-10)
    ax.set_xlabel("")
    ax.set_ylabel(yaxis_label)
    ax.yaxis.set_major_formatter(FuncFormatter(yaxis_formattor))

    save_path = routing.default_fig_path(dataset, prefix_str + f"_{_aligment_style}.png")
    default_exit_save(fig, save_path)


def half_stack_view_default_macro(
        dataset: DataSet,
        node_level: str,
        plot_manual: PlotManual,
        prefix_keyword: Optional[str] = None,
        unit_shape: Tuple[float, float] = (STACK_X_INCHES, STACK_Y_INCHES),

        _element_trial_level: str = "trial",
        _append_subtract_view: bool = True,  # for two types node only
        _aligment_style: str = "Aligned2Trial",

        **kwargs,
) -> None:
    """
    Generate stack view overview plots for all nodes within a dataset.    
    Creates a multi-panel figure with each row per node and each column per selected type.
    """
    nodes2plot = dataset.select(node_level)
    save_name = f"{dataset.name}_{_element_trial_level}@{node_level}_half_stackview"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    try:
        n_col = 0
        n_row = 0
        mosaic_style, content_dict = [], {}
        for node in nodes2plot:
            all_trial_nodes = dataset.subtree(node).select(_element_trial_level, _empty_warning=False)
            all_type2dataset = select_from_value(
                all_trial_nodes.rule_based_group_by(lambda x: x.info.get("trial_type")),
                _self = partial(CHECK_PLOT_MANUAL, plot_manual=plot_manual)
            )
            for half_idx in range(2):
                type2dataset = {k: DataSet(f"{k}_{half_idx}", v.nodes[half_idx * len(v) // 2: (half_idx + 1) * len(v) // 2]) 
                                for k, v in sorted(all_type2dataset.items(), key=lambda x: len(x[1]))}
                if len(type2dataset) == 1:
                    continue
                n_row += 1
                # update mosaic_style and content_dict
                mosaic_style.append([f"{get_node_name(node)}_{half_idx}\n{selected_type}" 
                                        for selected_type in type2dataset.keys()])
                content_dict.update({
                    f"{get_node_name(node)}_{half_idx}\n{selected_type}": (
                        partial(stack_view, plot_manual=plot_manual, sync_events=alignment_events),
                        type2dataset[selected_type])
                    for selected_type in type2dataset.keys()
                    })
                
                if _append_subtract_view and len(type2dataset) == 2:
                    selected_type1, selected_type2 = type2dataset.keys()
                    mosaic_style[-1].append(f"{get_node_name(node)}_{half_idx}\nSubtraction")
                    content_dict[f"{get_node_name(node)}_{half_idx}\nSubtraction"] = (
                        partial(subtract_view, plot_manual=plot_manual, sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1=selected_type1, name2=selected_type2)),
                        list(type2dataset.values())
                    )
                
                n_col = max(n_col, len(mosaic_style[-1]))
        for i in range(len(mosaic_style)):
            mosaic_style[i] += ["."] * (n_col - len(mosaic_style[i]))

        default_style(
            mosaic_style=mosaic_style,
            content_dict=content_dict,
            figsize=(unit_shape[0] * n_col, unit_shape[1] * n_row),
            save_path=routing.default_fig_path(dataset, prefix_str + f"_{{}}_{_aligment_style}.png"),
            **kwargs,
        )
    except Exception as e:
        logger.debug(f"Cannot plot stack view default macro for {dataset.name} with {_aligment_style}: {e}")
        logger.debug(traceback.format_exc())
