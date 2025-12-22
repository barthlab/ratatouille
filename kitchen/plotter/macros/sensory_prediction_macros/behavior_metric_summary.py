
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
from kitchen.plotter.ax_plotter.basic_plot import heatmap_view, stack_view
from kitchen.plotter.ax_plotter.basic_trace import trace_view
from kitchen.plotter.color_scheme import num_to_color, string_to_hex_color
from kitchen.plotter.decorators.default_decorators import default_exit_save, default_plt_param, default_style
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL, PlotManual
from kitchen.plotter.plotting_params import HEATMAP_X_INCHES, HEATMAP_Y_INCHES, STACK_X_INCHES, STACK_Y_INCHES
from kitchen.plotter.stats_tests.basic_ttest import stats_ks_2samp
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Mice, Node, NodeTypeVar, Session, Trial
from kitchen.utils.sequence_kit import find_only_one, select_from_value

logger = logging.getLogger(__name__)

# plotting
default_plt_param()

def sensory_prediction_summary_behavior_macro(
        dataset: DataSet,
        behavior_name: str,
        
        left_day_trial_types: set[str],
        right_day_trial_types: set[str],

        range_baseline: tuple[float, float],
        range_compare: tuple[float, float],
        baseline_setup: tuple[float, float, bool],
        amplitude_setup: tuple[float, float, str],

        yaxis_formattor,
        yaxis_label: str,
        yaxis_lim: tuple[float, float],

        prefix_keyword: Optional[str] = None,

        
        _additional_trial_types: Optional[dict[str, callable]] = None,
        
        _aligment_style: str = "Aligned2Trial",

        _day_offset: float = 0.8,

        _right_day_cover: str = "gray",


):
    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    plot_manual = PlotManual(**{behavior_name: True}, baseline_subtraction=baseline_setup, amplitude_sorting=amplitude_setup)
    save_name = f"{dataset.name}_{behavior_name}_delta"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    all_y_group_nodes = dataset.select("mice", _empty_warning=False)
    group_of_dataset = {}
    plotting_settings = {}


    def y_axis_func(node: Node) -> float:
        behavior_data = getattr(node.data, behavior_name)
        assert behavior_data is not None, f"{behavior_name} data is None for {node}"
        compare_array = behavior_data.segment(*range_compare).v
        baseline_array = behavior_data.segment(*range_baseline).v
        compare_value = 0 if len(compare_array) == 0 else np.nanmean(compare_array)
        baseline_value = 0 if len(baseline_array) == 0 else np.nanmean(baseline_array)
        return compare_value - baseline_value
    



    session_id2x_coord_dict = {}
    for y_group_idx, y_group_node in enumerate(all_y_group_nodes):
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
        print(f"\n{y_group_node.mice_id}:")


        for day_idx, day_node in enumerate(left_day2_plot):
            major_x_coord = day_idx - len(left_day2_plot)
            all_session_nodes = y_group_subtree.subtree(day_node).select("cellsession")
            for session_idx, session_node in enumerate(all_session_nodes):
                session_id2x_coord_dict[session_node.session_id] = (major_x_coord + _day_offset * (session_idx / len(all_session_nodes)), 
                                                                    major_x_coord, day_idx, len(left_day2_plot), session_idx, len(all_session_nodes), "left")
        for day_idx, day_node in enumerate(right_day2_plot):
            major_x_coord = day_idx
            all_session_nodes = y_group_subtree.subtree(day_node).select("cellsession")
            for session_idx, session_node in enumerate(all_session_nodes):
                session_id2x_coord_dict[session_node.session_id] = (major_x_coord + _day_offset * (session_idx / len(all_session_nodes)), 
                                                                    major_x_coord, day_idx, len(right_day2_plot), session_idx, len(all_session_nodes), "right")

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

        # plot mice summary
        all_right_days_subtree = sum([y_group_subtree.subtree(day_node) for day_node in right_day2_plot], 
                                     DataSet(name="all_right_days", nodes=[]))
        all_right_days_trial_types = split_dataset_by_trial_type(
            all_right_days_subtree, 
            plot_manual=plot_manual,
            _element_trial_level = "trial",
            _add_dummy=True,
        )
        if _additional_trial_types is not None:
            for trial_type, func in _additional_trial_types.items():
                all_right_days_trial_types[trial_type] = all_right_days_subtree.select("trial", _self=func)
        for key, value in all_right_days_trial_types.items():
            print(key, len(value))

        

        # heatmap summary
        mosaic_style = [
            f"{y_group_node.mice_id}\n{trial_type}" 
            for trial_type in all_right_days_trial_types.keys()
        ]
        content_dict = {}
        for trial_type, trial_dataset in all_right_days_trial_types.items():
            if trial_type in left_day_trial_types or trial_type == "Dummy":
                content_dict[f"{y_group_node.mice_id}\n{trial_type}"] = (
                    partial(heatmap_view, plot_manual=plot_manual, sync_events=alignment_events, 
                            modality_name=behavior_name, _sort_rows=True),
                    trial_dataset,
                )
            else:
                content_dict[f"{y_group_node.mice_id}\n{trial_type}"] = (
                    partial(heatmap_view, plot_manual=plot_manual, sync_events=alignment_events, 
                            modality_name=behavior_name, _sort_rows=False),
                    trial_dataset,
                )
        
        default_style(
            mosaic_style=[mosaic_style],
            content_dict=content_dict,
            figsize=(len(mosaic_style) * 2.2, 3.5),
            save_path=routing.default_fig_path(y_group_subtree, prefix_str + f"_heatmap_{_aligment_style}.png"),
            sharey=False,
        )
        


        # effective curve summary
        fig, axs = plt.subplots(len(right_day2_plot), len(all_right_days_trial_types), 
                                figsize=(9, 3.5), sharey='all', constrained_layout=True)
        for row_idx, day_node in enumerate(right_day2_plot):
            dummay_trials = all_right_days_trial_types["Dummy"].select("trial", day_id=day_node.day_id)
            dummay_ec_values = sorted([y_axis_func(trial) for trial in sync_nodes(dummay_trials, alignment_events, plot_manual)])
            for col_idx, trial_type in enumerate(all_right_days_trial_types.keys()):
                ax = axs[row_idx, col_idx]
                specific_trials = all_right_days_trial_types[trial_type].select("trial", day_id=day_node.day_id)
                ec_values = sorted([y_axis_func(trial) for trial in sync_nodes(specific_trials, alignment_events, plot_manual)])
              
                ax.plot(np.linspace(0, 1, len(ec_values)), ec_values, color="black")
                ax.plot(np.linspace(0, 1, len(dummay_ec_values)), dummay_ec_values, color="gray", alpha=0.5, ls='--')

                axs[0, col_idx].set_title(f"{y_group_node.mice_id}\n{trial_type}")
                axs[row_idx, 0].set_ylabel(f"day{row_idx+1}")
                ax.spines[['right', 'top',]].set_visible(False)
                ax.axhline(0, color='gray', lw=0.5, alpha=0.5, zorder=-10)
                ax.set_xticks([0, 1], ["min", "max"])

                ks_pvalue, annot_text = stats_ks_2samp(ec_values, dummay_ec_values)
                ax.text(0.5, 0.9, "K-S\n"+annot_text, alpha=0.7 if ks_pvalue < 0.05 else 0.2,
                        fontsize="x-small",
                        transform=ax.transAxes, ha='center', va='top')
                
        default_exit_save(fig, routing.default_fig_path(y_group_subtree, prefix_str + f"_effective_{_aligment_style}.png"))
        

        # trial average summary
        mosaic_style, content_dict = [], {}
        special_cols = ["Dummy",] + list(_additional_trial_types.keys() if _additional_trial_types is not None else [])
        first2trial_types = list(all_right_days_trial_types.keys())[:2]
        for col_idx, trial_type in enumerate(all_right_days_trial_types.keys()):
            if trial_type not in special_cols:
                day1_trials = all_right_days_trial_types[trial_type].select("trial", day_id=right_day2_plot[0].day_id)
                day2_trials = all_right_days_trial_types[trial_type].select("trial", day_id=right_day2_plot[1].day_id)

                mosaic_style.append(f"{y_group_node.mice_id}\n{trial_type}")

                subtract_manual = SUBTRACT_MANUAL(color1="C0", color2="C1", name1="day1", name2="day2")

                content_dict[f"{y_group_node.mice_id}\n{trial_type}"] = (
                    partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                    [day1_trials, day2_trials]
                )
            elif trial_type == "Dummy":
                mosaic_style.append(f"{y_group_node.mice_id}\n{first2trial_types[0]} vs {first2trial_types[1]}")
                subtract_manual = SUBTRACT_MANUAL(color1="C2", color2="C3", name1=first2trial_types[0], name2=first2trial_types[1])
                content_dict[f"{y_group_node.mice_id}\n{first2trial_types[0]} vs {first2trial_types[1]}"] = (
                    partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                    [all_right_days_trial_types[first2trial_types[0]], all_right_days_trial_types[first2trial_types[1]]]
                )
            else:
                mosaic_style.append(f"{first2trial_types[0]} vs\n{trial_type}")
                subtract_manual = SUBTRACT_MANUAL(color1="C2", color2=f"C{special_cols.index(trial_type) + 3}", 
                                                  name1=first2trial_types[0], name2=trial_type)
                content_dict[f"{first2trial_types[0]} vs\n{trial_type}"] = (
                    partial(subtract_view, subtract_manual=subtract_manual, plot_manual=plot_manual, sync_events=alignment_events),
                    [all_right_days_trial_types[first2trial_types[0]], all_right_days_trial_types[trial_type]]
                )
        default_style(
            mosaic_style=[mosaic_style],
            content_dict=content_dict,
            figsize=(10, 2.2),
            save_path=routing.default_fig_path(y_group_subtree, prefix_str + f"_trial_average_{_aligment_style}.png"),
            sharey=True,
            # sharex=False,
        )




    if len(group_of_dataset) == 0:
        return


    def x_axis_func(node: Node) -> float:
        return session_id2x_coord_dict.get(node.session_id, [np.nan, np.nan])[0]
    
    def x_axis_func_mini(node: Node) -> float:
        trial_type_offset = 0. if node.info.get("trial_type") in left_day_trial_types else 0.2
        return session_id2x_coord_dict.get(node.session_id, [np.nan, np.nan])[1] + trial_type_offset
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
    group_of_dataset = {k: v for k, v in sorted(group_of_dataset.items(), key=lambda x: x[0])}
    trace_view(
        ax=ax, group_of_datasets=group_of_dataset, y_axis_func=y_axis_func, x_axis_func=x_axis_func,
        plotting_settings=plotting_settings,
        _yerr_bar=False,
        _break_at_int=True,
    )
    ax.set_xticks([-1.6, -0.6, 0.4, 1.4,], ["day1", "day2", "day1", "day2"])
    ax.axvspan(-0.2, 1.8, alpha=0.1, color=_right_day_cover, lw=0, zorder=-10)
    ax.axhline(0, color='gray', lw=0.2, alpha=0.5, zorder=-10)
    ax.set_xlabel("")
    ax.set_ylabel(yaxis_label)
    ax.set_ylim(*yaxis_lim)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_formatter(FuncFormatter(yaxis_formattor))

    save_path = routing.default_fig_path(dataset, prefix_str + f"_{_aligment_style}.png")
    default_exit_save(fig, save_path)

    
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    group_of_dataset = {k: v for k, v in sorted(group_of_dataset.items(), key=lambda x: x[0])}
    trace_view(
        ax=ax, group_of_datasets=group_of_dataset, y_axis_func=y_axis_func, x_axis_func=x_axis_func_mini,
        plotting_settings=plotting_settings,
        _yerr_bar=False,
        _break_at_int=False,
    )
    ax.set_xticks([-2, -1, 0.1, 1.1,], ["day1", "day2", "day1", "day2"])
    ax.axvspan(-0.5, 1.5, alpha=0.1, color=_right_day_cover, lw=0, zorder=-10)
    ax.axhline(0, color='gray', lw=0.2, alpha=0.5, zorder=-10)
    ax.set_xlabel("")
    ax.set_ylabel(yaxis_label)
    ax.set_ylim(*yaxis_lim)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_formatter(FuncFormatter(yaxis_formattor))

    save_path = routing.default_fig_path(dataset, prefix_str + f"_{_aligment_style}_mini.png")
    default_exit_save(fig, save_path)

