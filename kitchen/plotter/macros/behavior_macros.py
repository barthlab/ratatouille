
from functools import partial
from typing import Optional, Type, TypeVar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from kitchen.configs import routing
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.ax_plotter.basic_trace import trace_view
from kitchen.plotter.decorators.default_decorators import default_exit_save, default_plt_param
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL, PlotManual
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Node, NodeTypeVar, Session
from kitchen.utils.sequence_kit import find_only_one, select_from_value


def trace_view_delta_behavior_macro(
        dataset: DataSet,
        behavior_name: str,
        transition_trial_type: str,
        range_compare: tuple[float, float],
        range_baseline: tuple[float, float],
        prefix_keyword: Optional[str] = None,
        
        _aligment_style: str = "Aligned2Trial",
        _y_group_level: str = "mice",
        _x_group_type: Type[NodeTypeVar] = Session,
        _to_percent: bool = False,
        _day_offset: float = 0.8,
):
    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    plot_manual = PlotManual(**{behavior_name: True})
    save_name = f"{dataset.name}_{behavior_name}_@{_y_group_level}@{_x_group_type.__name__}_delta"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name
    
    all_trial_nodes = sync_nodes(dataset.select("trial", _empty_warning=False), alignment_events, plot_manual)

    all_y_group_nodes = dataset.select(_y_group_level, _empty_warning=False)
    type2dataset = select_from_value(
        all_trial_nodes.rule_based_group_by(lambda x: x.info.get("trial_type")),
        _self = partial(CHECK_PLOT_MANUAL, plot_manual=plot_manual)
    )
    group_of_dataset = {
        selected_type + " " + y_group_node.object_uid.edge: 
        type2dataset[selected_type].select("trial", coordinate=lambda x: y_group_node.coordinate.contains(x)) 
        for selected_type in type2dataset.keys() for y_group_node in all_y_group_nodes
    }
    
    def y_axis_func(node: Node) -> float:
        behavior_data = getattr(node.data, behavior_name)
        return behavior_data.segment(*range_compare).v.mean() - behavior_data.segment(*range_baseline).v.mean()

    mice_day_dict, mice_transition_day_dict, mice_x_group_dict = {}, {}, {}
    for y_group_node in all_y_group_nodes:
        all_day_nodes_per_y_group = dataset.subtree(y_group_node).select("day")
        mice_day_dict[y_group_node.object_uid.edge] = [day_node.day_id for day_node in all_day_nodes_per_y_group]
        mice_x_group_dict[y_group_node.object_uid.edge] = [[x_group_node.temporal_uid 
                                                  for x_group_node in dataset.subtree(day_node).select(_x_group_type.hash_key)] 
                                                 for day_node in all_day_nodes_per_y_group]
        for day_node in all_day_nodes_per_y_group:
            all_trial_nodes_per_day = dataset.subtree(day_node).select("trial", _empty_warning=False)
            all_trial_types_per_day = all_trial_nodes_per_day.rule_based_group_by(lambda x: x.info.get("trial_type"))
            if transition_trial_type in all_trial_types_per_day:
                mice_transition_day_dict[y_group_node.object_uid.edge] = day_node.day_id
                break
            
    x_ticks, x_tick_labels = [], []
    def x_axis_func(node: Node) -> float:
        node_y_group_id = node.object_uid.get_hier_value(_y_group_level)
        cur_day_index = mice_day_dict[node_y_group_id].index(node.day_id)
        transition_day_index = mice_day_dict[node_y_group_id].index(mice_transition_day_dict[node_y_group_id])
        cur_x_group_index = mice_x_group_dict[node_y_group_id][cur_day_index].index(
            node.temporal_uid.transit(_x_group_type._expected_temporal_uid_level))
        if cur_day_index - transition_day_index not in x_ticks:
            x_ticks.append(cur_day_index - transition_day_index)
            x_tick_labels.append(f"Day {cur_day_index - transition_day_index}")
        return cur_day_index - transition_day_index + _day_offset * (cur_x_group_index / len(mice_x_group_dict[node_y_group_id][cur_day_index]))
    

    # plotting
    # default_plt_param()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    trace_view(
        ax=ax, group_of_datasets=group_of_dataset, y_axis_func=y_axis_func, x_axis_func=x_axis_func,
        plotting_settings={selected_type: {"marker": "o",} for selected_type in group_of_dataset.keys()},
        _break_at_int=True,
    )
    ax.set_xticks(np.array(x_ticks) + _day_offset / 2, x_tick_labels)
    ax.set_xlabel("Day From Transition")
    unit = "%" if _to_percent else "a.u."
    ax.set_ylabel(r"$\Delta$ " + f"{behavior_name} [{unit}]")

    if _to_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=-10)
    
    save_path = routing.default_fig_path(dataset, prefix_str + f"_{_aligment_style}.png")
    default_exit_save(fig, save_path)
