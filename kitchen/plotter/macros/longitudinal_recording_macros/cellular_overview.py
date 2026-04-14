
from asyncio import coroutines
from collections import defaultdict
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from kitchen.calculator.basic_metric import AUC_VALUE, AVERAGE_VALUE, PEAK_VALUE, PEARSON_CORRELATION
from kitchen.calculator.curve_fitting import fit_dataset_trial_fluo
from kitchen.calculator.sorting_data import get_amplitude_diff_sorted_idxs, get_amplitude_sorted_idxs
from kitchen.configs import routing
from kitchen.operator.grouping import grouping_events_rate, grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.ax_plotter.advance_plot import subtract_view
from kitchen.plotter.ax_plotter.basic_plot import heatmap_view, stack_view
from kitchen.plotter.decorators.default_decorators import coroutine_cycle, default_exit_save
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import LICK_BIN_SIZE, LOCOMOTION_BIN_SIZE
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_timeline
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Node
from kitchen.structure.neural_data_structure import Events, Fluorescence, TimeSeries, TimeSeries_concat
from kitchen.utils.sequence_kit import filter_by, find_only_one, select_truthy_items


def get_session_index(cs_node: Node) -> int:
    session_index_str = cs_node.session_id.split("_")[3]
    if session_index_str == "P1":
        return 0
    elif session_index_str == "P2":
        return 6
    else:
        assert session_index_str.startswith("S"), f"Unexpected session id format: {cs_node.session_id}"
        return int(session_index_str[1:])

def get_session_index2(cs_node: Node) -> int:
    session_index_str = cs_node.session_id.split("_")[3]
    if session_index_str == "P1":
        return 0
    elif session_index_str == "P2":
        if ("A50" in cs_node.session_id) or ("A100" in cs_node.session_id):
            return 1
        else:
            return 6
    else:
        assert session_index_str.startswith("S"), f"Unexpected session id format: {cs_node.session_id}"
        return int(session_index_str[1:])


def get_day_index(cs_node: Node) -> Optional[int]:
    day_id_int = int(cs_node.day_id)
    if cs_node.mice_id == "SUS6F":
        if day_id_int == 9:
            return None
        elif day_id_int > 9:    
            return day_id_int - 1
    return day_id_int


def visualize_celluar_activity_with_behavior(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    # plot_manual_whisk = PlotManual(whisker=True, baseline_subtraction=None)
    plot_manual_lick = PlotManual(lick=True, baseline_subtraction=None)
    plot_manual_loco = PlotManual(locomotion=True, baseline_subtraction=None)
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9
    for mice_node in dataset.select("mice"):
        mice_subtree = dataset.subtree(mice_node)
    
        for cell_order, cell_node in enumerate(mice_subtree.select("cell")):
            cell_subtree = mice_subtree.subtree(cell_node)

            # get all days
            days = sorted(cell_subtree.get_temporal_hiers("day", unique=True))            
            print(f"Plotting {cell_node.coordinate} with {len(days)} days...")

            ACC_Days = days[:6]
            Training_Days = days[6:11] if mice_node.mice_id != "SUS6F" else days[6:8] + days[9:12]
            Extended_Days = days[11:]

            day2col_offset, day2col_num = {}, {}
            def count_cols(days, given_offset):
                col_num = 0
                for day_idx, cell_day_id in enumerate(days):
                    cell_day_node = find_only_one(cell_subtree.select("cellday", day_id=cell_day_id).nodes)
                    all_cell_sessions = cell_subtree.subtree(cell_day_node).select("cellsession")
                    type2dataset = split_dataset_by_trial_type(cell_subtree.subtree(cell_day_node), 
                                                            plot_manual=plot_manual_fluo,
                                                            _element_trial_level =_element_trial_level,)
                    day2col_offset[cell_day_id] = col_num + given_offset
                    day2col_num[cell_day_id] = list(type2dataset.keys())
                    day2col_num[cell_day_id].remove("PuffOnly")
                    day2col_num[cell_day_id].remove("BlankOnly")
                    col_num += len(day2col_num[cell_day_id])
                return col_num
            # n_col_ACC = count_cols(ACC_Days, 0)
            # n_col_Training = count_cols(Training_Days, n_col_ACC + 1)
            n_col_Training = count_cols(Training_Days, 0)
            # n_col_Extended = count_cols(Extended_Days, n_col_ACC + 1 + n_col_Training + 1)

            
            # total_col_num = n_col_ACC + 1 + n_col_Training + 1 + n_col_Extended

            def CelluarOverview_with_behavior_TrainingDays():
                fig, axs = plt.subplots(18, n_col_Training, figsize=(3 + 25/20 * n_col_Training, 15), sharex=False, sharey=False, 
                                        constrained_layout=True, 
                                        width_ratios=[1,] * n_col_Training,
                                        # width_ratios=[1,] * n_col_ACC + [0.5,] + [1,] * n_col_Training + [0.5,] + [1,] * n_col_Extended,
                                        height_ratios=[ 
                                                        1, 1, 1, 1, 1, 2,
                                                        ] * 3)
                fig.suptitle(f"Cell {cell_order} in {mice_node.mice_id}", fontsize=50)
                
                for ax in axs.flatten():
                    ax.tick_params(axis='y', labelleft=True)
                    ax.set_yticks([])
                    ax.set_xlim(-5, 5)  
                    ax.set_xticks([])
                for ax in axs[5::6, :].flatten():
                    ax.set_xticks([-4, 0, 4])

                # for ax in axs[:, n_col_ACC].flatten():
                #     ax.remove()
                # for ax in axs[:, n_col_ACC + 1 + n_col_Training].flatten():
                #     ax.remove()

                axs[2, 0].set_ylabel("Lick", fontsize=15)
                axs[8, 0].set_ylabel("Fluorescence", fontsize=15)
                axs[14, 0].set_ylabel("Locomotion", fontsize=15)

                coroutines = {} 
                for day_idx, cell_day in enumerate(cell_subtree.select("cellday")):
                    if not ((7 <= int(cell_day.day_id) <= 13) and (cell_day.day_id in Training_Days)):
                        continue
                    cd_subtree = cell_subtree.subtree(cell_day)
                    all_cell_sessions = cd_subtree.select("cellsession")
                    print(cell_day.coordinate, f"Found {len(all_cell_sessions)} sessions...")
                    assert len(all_cell_sessions) <= 7, f"Expected less than 7 session, got {len(all_cell_sessions)}"

                    col_index_start = day2col_offset[cell_day.day_id]
                    col_quota = len(day2col_num[cell_day.day_id])

                    def get_cs_title(cs_node):
                        return cs_node.session_id.split("_")[3]
                    
                    def match_cs_title(cs_node, match2str):
                        element_cs_title = get_cs_title(cs_node)
                        if match2str == "P1" or match2str == "P2":
                            return element_cs_title == match2str
                        elif match2str == "S":
                            return element_cs_title.startswith("S")
                        else:
                            raise ValueError(f"Unknown match2str {match2str}")
                        
                    # passive1_cs_node = filter_by(all_cell_sessions.nodes, _self=lambda cs_node: match_cs_title(cs_node, "P1"))
                    # passive2_cs_node = filter_by(all_cell_sessions.nodes, _self=lambda cs_node: match_cs_title(cs_node, "P2"))
                    active_cs_nodes = filter_by(all_cell_sessions.nodes, _self=lambda cs_node: match_cs_title(cs_node, "S"))
                    
                    alive_cs_nodes = {i+1: None for i in range(5)}
                    for cs_node in active_cs_nodes:
                        cs_title_index = int(get_cs_title(cs_node)[1:])
                        alive_cs_nodes[cs_title_index] = cs_node

                    for session_index, cs_node in alive_cs_nodes.items():

                        row_start_idx = session_index - 1
                        if cs_node is None:
                            for content_idx in range(3):
                                for ax in axs[row_start_idx+content_idx*6:row_start_idx+1+content_idx*6, 
                                                col_index_start: col_index_start + col_quota].flatten():
                                    ax.remove()
                            continue
                        cs_subtree = cd_subtree.subtree(cs_node)

                
                        type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                        assert 4 >= len(type2dataset) > 0, f"Expected 1-4 trial types, got {len(type2dataset)}: {type2dataset.keys()}"

                        for trial_type, raw_type_dataset in type2dataset.items():
                            type_col_idx = day2col_num[cell_day.day_id].index(trial_type) + col_index_start
                            type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                            for content_idx, (modality_name, specific_plot_manual) in enumerate(zip(
                                ["lick", "fluorescence", "locomotion", ], 
                                [plot_manual_lick, plot_manual_fluo, plot_manual_loco])):
                                specific_plotter = heatmap_view(
                                    ax=axs[row_start_idx+content_idx*6, type_col_idx], datasets=type_dataset, sync_events=alignment_events, 
                                    plot_manual=specific_plot_manual, modality_name=modality_name)
                                coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}"
                                axs[0+content_idx*6, type_col_idx].set_title(f"{trial_type}")
                    
                    all_active_nodes = sum([cd_subtree.subtree(active_cs_node) for active_cs_node in active_cs_nodes],
                                                        DataSet(name="active_dataset", nodes=[]))
                    type2dataset = split_dataset_by_trial_type(all_active_nodes, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                    for trial_type, raw_type_dataset in type2dataset.items():
                        type_col_idx = day2col_num[cell_day.day_id].index(trial_type) + col_index_start
                        type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                        for content_idx, (modality_name, specific_plot_manual) in enumerate(zip(
                            ["lick", "fluorescence", "locomotion", ], 
                            [plot_manual_lick, plot_manual_fluo, plot_manual_loco])):
                        
                            specific_plotter = stack_view(
                                ax=axs[5+content_idx*6, type_col_idx], datasets=type_dataset, sync_events=alignment_events, 
                                plot_manual=specific_plot_manual)
                            coroutines[specific_plotter] = f"{trial_type}_stack_view_{modality_name}"   
                            # print(day2col_num[cell_day.day_id], trial_type, col_index_start, type_col_idx)
                            axs[5+content_idx*6, type_col_idx].set_ylim(0, 4 if modality_name == "fluorescence" else 2)
                    


                    if len(type2dataset) < col_quota:
                        for content_idx in range(3):
                            for ax in axs[0+content_idx*6:6+content_idx*6, 
                                        col_index_start + len(type2dataset): col_index_start + col_quota].flatten():
                                ax.remove()
                coroutine_cycle(coroutines)
                
                
                for day_idx, cell_day in enumerate(cell_subtree.select("cellday")):
                    if not ((7 <= int(cell_day.day_id) <= 13) and (cell_day.day_id in Training_Days)):
                        continue
                    col_index_start = day2col_offset[cell_day.day_id]
                    col_quota = len(day2col_num[cell_day.day_id])
                    for ax in axs[5::6, col_index_start + 1: col_index_start + col_quota].flatten():
                        ax.set_yticks([])


                save_path = routing.default_fig_path(cell_node, "CelluarOverview_with_behavior_TrainingDays" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
                default_exit_save(fig, save_path)

            def CelluarSummary_TrainingDays():
                fig, axs = plt.subplots(len(Training_Days) + 1, 7, figsize=(8, 6), sharex=False, sharey=False,
                                                    constrained_layout=True)
                fig.suptitle(f"Cell {cell_order} in {mice_node.mice_id}", fontsize=15)
                
                for ax in axs.flatten():
                    ax.set_ylim(0, 4)
                    ax.set_yticks([])
                    ax.set_xlim(-5, 5)  
                    ax.set_xticks([])
                for ax in axs[-2:, :].flatten():
                    ax.set_xticks([-4, 0, 4])
                
                for ax_id, ax in enumerate(axs[:, 0].flatten()):
                    ax.set_ylabel(f"Day {1 + ax_id}", fontsize=10)
                axs[-1, 0].set_ylabel("All days", fontsize=10)


                coroutines = {} 
                day_idx = -1
                all_day_active_nodes = []
                for cell_day in cell_subtree.select("cellday"):
                    if not ((7 <= int(cell_day.day_id) <= 13) and (cell_day.day_id in Training_Days)):
                        continue
                    day_idx += 1
                    cd_subtree = cell_subtree.subtree(cell_day)
                    all_cell_sessions = cd_subtree.select("cellsession")
                    print(cell_day.coordinate, f"Found {len(all_cell_sessions)} sessions...")
                    assert len(all_cell_sessions) <= 7, f"Expected less than 7 session, got {len(all_cell_sessions)}"

                    def get_cs_title(cs_node):
                        return cs_node.session_id.split("_")[3]
                    def match_cs_title(cs_node, match2str):
                        element_cs_title = get_cs_title(cs_node)
                        if match2str == "P1" or match2str == "P2":
                            return element_cs_title == match2str
                        elif match2str == "S":
                            return element_cs_title.startswith("S")
                        else:
                            raise ValueError(f"Unknown match2str {match2str}")
                        
                    active_cs_nodes = filter_by(all_cell_sessions.nodes, _self=lambda cs_node: match_cs_title(cs_node, "S"))
                    all_active_nodes = sum([cd_subtree.subtree(active_cs_node) for active_cs_node in active_cs_nodes],
                                                        DataSet(name="active_dataset", nodes=[]))
                    all_day_active_nodes.append(all_active_nodes)
                    type2dataset = split_dataset_by_trial_type(all_active_nodes, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                    for col_id, type_name in enumerate(["CuePuffWater", "CuePuffNoWater", "CueBlankWater", "CueBlankNoWater"]):
                        if type_name not in type2dataset:
                            axs[day_idx, col_id].remove()
                            continue
                        raw_type_dataset = type2dataset[type_name]
                        type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                        specific_plotter = stack_view(
                            ax=axs[day_idx, col_id], datasets=type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo)
                        coroutines[specific_plotter] = f"{type_name}_stack_view_fluorescence_day{day_idx}"
                        axs[0, col_id].set_title(f"{type_name}")
                    puff_dataset = type2dataset.get("CuePuffWater", DataSet(name="", nodes=[])) + type2dataset.get("CuePuffNoWater", DataSet(name="", nodes=[]))
                    blank_dataset = type2dataset.get("CueBlankWater", DataSet(name="", nodes=[])) + type2dataset.get("CueBlankNoWater", DataSet(name="", nodes=[]))
                    specific_plotter = stack_view(
                        ax=axs[day_idx, -3], datasets=puff_dataset, sync_events=alignment_events, 
                        plot_manual=plot_manual_fluo)
                    coroutines[specific_plotter] = f"puff_stack_view_fluorescence_day{day_idx}"
                    specific_plotter = stack_view(
                        ax=axs[day_idx, -2], datasets=blank_dataset, sync_events=alignment_events, 
                        plot_manual=plot_manual_fluo)
                    coroutines[specific_plotter] = f"blank_stack_view_fluorescence_day{day_idx}"
                    axs[0, -3].set_title("Puff", color='C0')
                    axs[0, -2].set_title("Blank", color='C1')
                    specific_plotter = subtract_view(
                        ax=axs[day_idx, -1], datasets=[puff_dataset, blank_dataset], sync_events=alignment_events, 
                        subtract_manual=SUBTRACT_MANUAL(name1="Puff", name2="Blank"), plot_manual=plot_manual_fluo)
                    coroutines[specific_plotter] = f"subtract_view_fluorescence_day{day_idx}"
                    axs[0, -1].set_title("Puff vs Blank")
                    
                    coroutines[specific_plotter] = f"subtract_view_fluorescence_day{day_idx}"
                all_active_nodes_sumup = sum(all_day_active_nodes, DataSet(name="all_active_nodes_sumup", nodes=[]))
                type2dataset = split_dataset_by_trial_type(all_active_nodes_sumup, 
                                                            plot_manual=plot_manual_fluo,
                                                            _element_trial_level =_element_trial_level,)
                for col_id, type_name in enumerate(["CuePuffWater", "CuePuffNoWater", "CueBlankWater", "CueBlankNoWater"]):
                    if type_name not in type2dataset:
                        axs[-1, col_id].remove()
                        continue
                    raw_type_dataset = type2dataset[type_name]
                    type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                    specific_plotter = stack_view(
                        ax=axs[-1, col_id], datasets=type_dataset, sync_events=alignment_events, 
                        plot_manual=plot_manual_fluo)
                    coroutines[specific_plotter] = f"{type_name}_stack_view_fluorescence_sumup"
                puff_dataset = type2dataset.get("CuePuffWater", DataSet(name="", nodes=[])) + type2dataset.get("CuePuffNoWater", DataSet(name="", nodes=[]))
                blank_dataset = type2dataset.get("CueBlankWater", DataSet(name="", nodes=[])) + type2dataset.get("CueBlankNoWater", DataSet(name="", nodes=[]))
                specific_plotter = stack_view(
                    ax=axs[-1, -3], datasets=puff_dataset, sync_events=alignment_events, 
                    plot_manual=plot_manual_fluo)
                coroutines[specific_plotter] = f"puff_stack_view_fluorescence_sumup"
                specific_plotter = stack_view(
                    ax=axs[-1, -2], datasets=blank_dataset, sync_events=alignment_events, 
                    plot_manual=plot_manual_fluo)
                coroutines[specific_plotter] = f"blank_stack_view_fluorescence_sumup"
                specific_plotter = subtract_view(
                    ax=axs[-1, -1], datasets=[puff_dataset, blank_dataset], sync_events=alignment_events, 
                    subtract_manual=SUBTRACT_MANUAL(name1="Puff", name2="Blank"), plot_manual=plot_manual_fluo,)
                coroutines[specific_plotter] = f"subtract_view_fluorescence_sumup"
                coroutine_cycle(coroutines)

                for ax in axs[:, 1:4].flatten():
                    ax.set_yticks([])
                for ax in axs[:, -2].flatten():
                    ax.set_yticks([])

                save_path = routing.default_fig_path(cell_node, "CelluarSummary_TrainingDays" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
                default_exit_save(fig, save_path)

            def CelluarExamples_TrainingDays():
                fig, axs = plt.subplots(1, len(Training_Days), figsize=(5, 1.5), sharex=False, sharey=False,
                                                    constrained_layout=True)
                fig.suptitle(f"Cell {cell_order} in {mice_node.mice_id}", fontsize=15)
                
                for ax in axs.flatten():
                    ax.set_ylim(0, 2.5)
                    ax.set_yticks([])
                    ax.set_xlim(-4, 4)  
                    ax.set_xticks([-2, 0, 2])
                
                for ax_id, ax in enumerate(axs.flatten()):
                    ax.set_title(f"Day {ax_id + 1}", fontsize=7)


                coroutines = {} 
                day_idx = -1
                for cell_day in cell_subtree.select("cellday"):
                    if not ((7 <= int(cell_day.day_id) <= 13) and (cell_day.day_id in Training_Days)):
                        continue
                    day_idx += 1
                    cd_subtree = cell_subtree.subtree(cell_day)
                    
                    puff_trials = cd_subtree.select(
                        _element_trial_level, _self=lambda node: node.info.get("trial_type") in ["CuePuffWater", "CuePuffNoWater"])
                    blank_trials = cd_subtree.select(
                        _element_trial_level, _self=lambda node: node.info.get("trial_type") in ["CueBlankWater", "CueBlankNoWater"])
                    
                    specific_plotter = subtract_view(
                        ax=axs[day_idx], datasets=[puff_trials, blank_trials], sync_events=alignment_events, 
                        subtract_manual=SUBTRACT_MANUAL(name1="Puff", name2="Blank"), plot_manual=plot_manual_fluo)
                    coroutines[specific_plotter] = f"subtract_view_fluorescence_day{day_idx}"
                    # add_textonly_legend(axs[day_idx], {"Puff": {'color': 'C0'}, "Blank": {'color': 'C1'}})
                    
                coroutine_cycle(coroutines)
                for i in range(1, len(Training_Days)):
                    axs[i].set_yticks([])

                save_path = routing.default_fig_path(cell_node, "CelluarExamples_TrainingDays" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
                default_exit_save(fig, save_path)

           
            
            CelluarOverview_with_behavior_TrainingDays()
            
            CelluarSummary_TrainingDays()
            CelluarExamples_TrainingDays()


def visualize_celluar_activity_in_heatmap(
        dataset: DataSet,

        theme_color: str,
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def combine_trial_nodes(trial_nodes: DataSet) -> Node:
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes], 0).fusion()
        avg_node = sync_trial_nodes.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo.v + 1, t=group_fluo.t)
        avg_node.data.timeline = group_timeline
        return avg_node
    
    def combine_trial_node_delta(trial_nodes1: DataSet, trial_nodes2: DataSet) -> Node:
        sync_trial_nodes1 = sync_nodes(trial_nodes1, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo1 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes1], 
                                            baseline_subtraction=None).mean_ts
        sync_trial_nodes2 = sync_nodes(trial_nodes2, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo2 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes2], 
                                            baseline_subtraction=None,
                                            _predefined_t=group_fluo1.t).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes1] + 
                             [trial.data.timeline for trial in sync_trial_nodes2], 0).fusion()
        avg_node = sync_trial_nodes1.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo1.v - group_fluo2.v + 1, t=group_fluo1.t)
        avg_node.data.timeline = group_timeline
        return avg_node
    
    plotting_nodes, allday_plotting_nodes = [], []

    for mice_idx, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        training_day_range = list(range(7, 12)) if mice_node.mice_id != "SUS6F" else (7, 8, 10, 11, 12)

        for day_node in mice_subtree.select("day"):
            print(f"Processing {mice_node.mice_id} Day {day_node.day_id}")
            if not int(day_node.day_id) in training_day_range:
                continue
            training_day_index = training_day_range.index(int(day_node.day_id))
            day_subtree = mice_subtree.subtree(day_node)
            for cellday_node in day_subtree.select("cellday"):
                all_puff_trials = day_subtree.subtree(cellday_node).select(
                    _element_trial_level, 
                    _self=lambda x: x.info.get("trial_type") in ("CuePuffWater", "CuePuffNoWater"),
                )
                puff_avg_node = combine_trial_nodes(all_puff_trials)
                puff_avg_node.info["trial_type"] = "CuePuff"
                puff_avg_node.info["day_index"] = training_day_index
                
                all_blank_trials = day_subtree.subtree(cellday_node).select(
                    _element_trial_level, 
                    _self=lambda x: x.info.get("trial_type") in ("CueBlankWater", "CueBlankNoWater"),
                )
                blank_avg_node = combine_trial_nodes(all_blank_trials)
                blank_avg_node.info["trial_type"] = "CueBlank"
                blank_avg_node.info["day_index"] = training_day_index
                
                delta_avg_node = combine_trial_node_delta(all_puff_trials, all_blank_trials)
                delta_avg_node.info["trial_type"] = "Subtract"
                delta_avg_node.info["day_index"] = training_day_index

                plotting_nodes.extend([puff_avg_node, blank_avg_node, delta_avg_node])
                
                # puff_only_trials = day_subtree.subtree(cellday_node).select(
                #     _element_trial_level,
                #     _self=lambda x: x.info.get("trial_type") == "PuffOnly",
                # )
                # puff_only_avg_node = combine_trial_nodes(puff_only_trials)
                # puff_only_avg_node.info["trial_type"] = "PuffOnly"
                # puff_only_avg_node.info["day_index"] = training_day_index
                # blank_only_trials = day_subtree.subtree(cellday_node).select(
                #     _element_trial_level,
                #     _self=lambda x: x.info.get("trial_type") == "BlankOnly",
                # )
                # blank_only_avg_node = combine_trial_nodes(blank_only_trials)
                # blank_only_avg_node.info["trial_type"] = "BlankOnly"
                # blank_only_avg_node.info["day_index"] = training_day_index 
                # delta_only_avg_node = combine_trial_node_delta(puff_only_trials, blank_only_trials)
                # delta_only_avg_node.info["trial_type"] = "SubtractOnly" 
                # delta_only_avg_node.info["day_index"] = training_day_index

                # plotting_nodes.extend([puff_only_avg_node, blank_only_avg_node, delta_only_avg_node])
                
        for cell_nodes in mice_subtree.select("cell"):
            all_puff_trials = mice_subtree.subtree(cell_nodes).select(
                _element_trial_level, 
                _self=lambda x: x.info.get("trial_type") in ("CuePuffWater", "CuePuffNoWater"),
            )
            puff_avg_node = combine_trial_nodes(all_puff_trials)
            puff_avg_node.info["trial_type"] = "CuePuff"

            all_blank_trials = mice_subtree.subtree(cell_nodes).select(
                _element_trial_level, 
                _self=lambda x: x.info.get("trial_type") in ("CueBlankWater", "CueBlankNoWater"),
            )
            blank_avg_node = combine_trial_nodes(all_blank_trials)
            blank_avg_node.info["trial_type"] = "CueBlank"

            delta_avg_node = combine_trial_node_delta(all_puff_trials, all_blank_trials)
            delta_avg_node.info["trial_type"] = "Subtract"

            allday_plotting_nodes.extend([puff_avg_node, blank_avg_node, delta_avg_node])
        
    plotting_dataset = DataSet(name="plotting_dataset", nodes=plotting_nodes)
    allday_plotting_dataset = DataSet(name="allday_plotting_dataset", nodes=allday_plotting_nodes)

    def plot_daywise_summary_sort_by_basic():
        selected_trial_types = ["CuePuff", "CueBlank", "Subtract"]
        n_col = len(selected_trial_types)

        for sorting_day_index in (0, 4):
            for sort_by_flag in ("CuePuff", "Subtract", "CueBlank" ):
                for sort_setup in (((0, 0.5), False), ((0, 2), True)):
                    fig, axs = plt.subplots(2 * (len(dataset.select("mice")) + 1), 6*n_col, 
                                            figsize=(15, 8), sharex=False, sharey=False, constrained_layout=True,
                                            height_ratios=[1.5, 1,] * len(dataset.select("mice")) + [2, 1])
                    for ax in axs.flatten():
                        ax.tick_params(axis='y', labelleft=True)
                        ax.set_yticks([])
                        ax.set_xlim(-4, 4)  
                        ax.set_xticks([])
                    for ax in axs[1::2, :].flatten():
                        ax.set_xticks([-2, 0, 2])

                    coroutines = {} 
                    row_datasets = {mice_node.mice_id: (
                        plotting_dataset.select("trial", mice_id=mice_node.mice_id),
                        allday_plotting_dataset.select("trial", mice_id=mice_node.mice_id),
                        ) for mice_node in dataset.select("mice")}
                    row_datasets["All mice"] = (plotting_dataset, allday_plotting_dataset)
                    for row_idx, (row_name, (plotting_dataset_subtree, allday_plotting_dataset_subtree)) in enumerate(row_datasets.items()):
                        axs[row_idx*2, 0].set_ylabel(row_name, fontsize=10)

                        sorting_day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: (x.info.get("day_index") == sorting_day_index and 
                                                                                            x.info.get("trial_type") == sort_by_flag))
                        sync_sorting_day_subtree = sync_nodes(sorting_day_subtree, alignment_events, plot_manual=plot_manual_fluo)
                        fluo_adv_ts = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                            for single_trial in sync_sorting_day_subtree], 
                                                        baseline_subtraction=None)
                        sorting_order = get_amplitude_sorted_idxs(fluo_adv_ts, 
                                                                amplitude_range=sort_setup[0],
                                                                _mono_decreasing=sort_setup[1],)

                        for day_index in range(5):
                            day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: x.info.get("day_index") == day_index)
                            
                            type2dataset = split_dataset_by_trial_type(day_subtree, 
                                                                        plot_manual=plot_manual_fluo,
                                                                        _element_trial_level =_element_trial_level,)
                            for type_id, type_name in enumerate(selected_trial_types):
                                if type_name not in type2dataset:
                                    for ax in axs[2*row_idx:2*row_idx+2, day_index*n_col+type_id].flatten():
                                        ax.remove()
                                    continue
                                raw_type_dataset = type2dataset[type_name]
                                type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                            
                                assert [node.object_uid for node in type_dataset] == [node.object_uid for node in sorting_day_subtree], \
                                    f"Object uid mismatch between sorting day and other days, got \n" \
                                    f"{[node.object_uid.cell_id for node in type_dataset]} \n{[node.object_uid.cell_id for node in sorting_day_subtree]}"
                                

                                specific_plotter = heatmap_view(
                                    ax=axs[2*row_idx, day_index*n_col+type_id], datasets=type_dataset, sync_events=alignment_events, 
                                    plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=sorting_order
                                )
                                coroutines[specific_plotter] = f"{row_name}_day{day_index}_{type_name}_heatmap_view"
                                
                                specific_plotter = stack_view(
                                    ax=axs[2*row_idx+1, day_index*n_col+type_id], datasets=raw_type_dataset, sync_events=alignment_events, 
                                    plot_manual=plot_manual_fluo
                                ) if type_name != "Subtract" else subtract_view(
                                    ax=axs[2*row_idx+1, day_index*n_col+type_id], 
                                    datasets=[type2dataset["CuePuff"], type2dataset["CueBlank"]], sync_events=alignment_events, 
                                    subtract_manual=SUBTRACT_MANUAL(name1="CuePuff", name2="CueBlank"), plot_manual=plot_manual_fluo,
                                )
                                coroutines[specific_plotter] = f"{row_name}_day{day_index}_{type_name}_stack_view" \
                                    if type_name != "Subtract" else f"{row_name}_day{day_index}_{type_name}_subtract_view"

                                if type_name == "Subtract":
                                    display_name = "Diff"
                                elif type_name == "CuePuff":
                                    display_name = "Puff"
                                elif type_name == "CueBlank":
                                    display_name = "Blank"
                                else:
                                    display_name = type_name
                                axs[0, day_index*n_col+type_id].set_title(f"Day {day_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")
                                
                                axs[-2, day_index*n_col+type_id].set_title(f"Day {day_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")

                        type2dataset = split_dataset_by_trial_type(allday_plotting_dataset_subtree, 
                                                                    plot_manual=plot_manual_fluo,
                                                                    _element_trial_level =_element_trial_level,)
                        for type_id, type_name in enumerate(selected_trial_types):
                            if type_name not in type2dataset:
                                for ax in axs[2*row_idx:2*row_idx+2, -n_col+type_id].flatten():
                                    ax.remove()
                                continue
                            raw_type_dataset = type2dataset[type_name]
                            type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                            specific_plotter = heatmap_view(
                                ax=axs[2*row_idx, -n_col+type_id], datasets=type_dataset, sync_events=alignment_events, 
                                plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=sorting_order
                            )
                            coroutines[specific_plotter] = f"{row_name}_allday_{type_name}_heatmap_view"
                            
                            specific_plotter = stack_view(
                                ax=axs[2*row_idx+1, -n_col+type_id], datasets=raw_type_dataset, sync_events=alignment_events, 
                                plot_manual=plot_manual_fluo
                            ) if type_name != "Subtract" else subtract_view(
                                ax=axs[2*row_idx+1, -n_col+type_id], 
                                datasets=[type2dataset["CuePuff"], type2dataset["CueBlank"]], 
                                sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1="CuePuff", name2="CueBlank"), 
                                plot_manual=plot_manual_fluo,
                            )
                            coroutines[specific_plotter] = f"{row_name}_allday_{type_name}_stack_view" \
                                if type_name != "Subtract" else f"{row_name}_allday_{type_name}_subtract_view"
                            
                            if type_name == "Subtract":
                                display_name = "Diff"
                            elif type_name == "CuePuff":
                                display_name = "Puff"
                            elif type_name == "CueBlank":
                                display_name = "Blank"
                            else:
                                display_name = type_name
                            axs[0, -n_col+type_id].set_title(f"All days\n{display_name}" if type_id==1 else f"\n{display_name}")

                            axs[-2, -n_col+type_id].set_title(f"All days\n{display_name}" if type_id==1 else f"\n{display_name}")
                        
                    coroutine_cycle(coroutines)
                    for i in range(1, n_col):
                        for ax in axs[:, i::n_col].flatten():
                            ax.set_yticks([])
                    # for ax in axs[1::2, :].flatten():
                    #     ax.set_ylim(0, 2.)
                        
                    save_path = routing.default_fig_path(dataset, "HeatmapOverview_TrainingDays_" + f"_{{}}_{_aligment_style}_sortbyday{sorting_day_index + 1}_{sort_by_flag}_{'decrease' if sort_setup[1] else 'increase'}.png", fov_skip=True)
                    default_exit_save(fig, save_path)
    
    def plot_daywise_summary_trace_simple_only():
        selected_trial_types = ["Subtract", "CuePuff", "CueBlank", ]
        n_col = len(selected_trial_types)

        fig, axs = plt.subplots(n_col, 6, 
                                figsize=(8, 4), sharex=False, sharey=False, constrained_layout=True,
                                )
        # color_scheme.FLUORESCENCE_COLOR = theme_color
        style_dicts.FLUORESCENCE_TRACE_STYLE["color"] = theme_color
        for ax in axs.flatten():
            ax.tick_params(axis='y', labelleft=True)
            ax.set_yticks([])
            ax.set_xlim(-3, 3)  
            ax.set_xticks([-2, 0, 2])
            ax.set_ylim(0, 1.1)

        coroutines = {} 

        for day_index in range(5):
            day_subtree = plotting_dataset.select("trial", _self=lambda x: x.info.get("day_index") == day_index)
            
            type2dataset = split_dataset_by_trial_type(day_subtree, 
                                                        plot_manual=plot_manual_fluo,
                                                        _element_trial_level =_element_trial_level,)
            for type_id, type_name in enumerate(selected_trial_types):
                if type_name not in type2dataset:
                    axs[type_id, day_index].remove()
                    continue
                raw_type_dataset = type2dataset[type_name]
                type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                specific_plotter = stack_view(
                    ax=axs[type_id, day_index], datasets=raw_type_dataset, sync_events=alignment_events, 
                    plot_manual=plot_manual_fluo
                ) if type_name != "Subtract" else subtract_view(
                    ax=axs[type_id, day_index], 
                    datasets=[type2dataset["CuePuff"], type2dataset["CueBlank"]], sync_events=alignment_events, 
                    subtract_manual=SUBTRACT_MANUAL(name1="CuePuff", name2="CueBlank"), plot_manual=plot_manual_fluo,
                )
                coroutines[specific_plotter] = f"day{day_index}_{type_name}_stack_view" \
                    if type_name != "Subtract" else f"day{day_index}_{type_name}_subtract_view"

                if type_name == "Subtract":
                    display_name = "Diff"
                elif type_name == "CuePuff":
                    display_name = "Puff"
                elif type_name == "CueBlank":
                    display_name = "Blank"
                else:
                    display_name = type_name
                axs[0, day_index].set_title(f"Day {day_index+1}")
                axs[type_id, 0].set_ylabel(display_name)
                

        type2dataset = split_dataset_by_trial_type(allday_plotting_dataset, 
                                                    plot_manual=plot_manual_fluo,
                                                    _element_trial_level =_element_trial_level,)
        for type_id, type_name in enumerate(selected_trial_types):
            if type_name not in type2dataset:
                axs[type_id, -1].remove()
                continue
            raw_type_dataset = type2dataset[type_name]
            type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

            specific_plotter = stack_view(
                ax=axs[type_id, -1], datasets=raw_type_dataset, sync_events=alignment_events, 
                plot_manual=plot_manual_fluo
            ) if type_name != "Subtract" else subtract_view(
                ax=axs[type_id, -1], 
                datasets=[type2dataset["CuePuff"], type2dataset["CueBlank"]], 
                sync_events=alignment_events, 
                subtract_manual=SUBTRACT_MANUAL(name1="CuePuff", name2="CueBlank"), 
                plot_manual=plot_manual_fluo,
            )
            coroutines[specific_plotter] = f"allday_{type_name}_stack_view" \
                if type_name != "Subtract" else f"allday_{type_name}_subtract_view"
            
            if type_name == "Subtract":
                display_name = "Diff"
            elif type_name == "CuePuff":
                display_name = "Puff"
            elif type_name == "CueBlank":
                display_name = "Blank"
            else:
                display_name = type_name
            axs[0, -1].set_title(f"All days")
        
        coroutine_cycle(coroutines)
        for i in range(1, 5):
            for ax in axs[:, i].flatten():
                ax.set_yticks([])

        save_path = routing.default_fig_path(dataset, "HeatmapOverviewTraceSimple_TrainingDays_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
        default_exit_save(fig, save_path, _transparent=True)
    
    def plot_daywise_summary_trace_simple_only_passive():
        selected_trial_types = ["SubtractOnly", "PuffOnly", "BlankOnly", ]
        n_col = len(selected_trial_types)

        fig, axs = plt.subplots(n_col, 11, 
                                figsize=(10, 4), sharex=False, sharey=False, constrained_layout=True,
                                )
        # color_scheme.FLUORESCENCE_COLOR = theme_color
        style_dicts.FLUORESCENCE_TRACE_STYLE["color"] = theme_color
        for ax in axs.flatten():
            ax.tick_params(axis='y', labelleft=True)
            ax.set_yticks([])
            ax.set_xlim(-3, 3)  
            ax.set_xticks([-2, 0, 2])
            ax.set_ylim(0, 4)

        coroutines = {} 

        for day_index in range(11):
            day_subtree = plotting_dataset.select("trial", _self=lambda x: x.info.get("day_index") == day_index)
            
            type2dataset = split_dataset_by_trial_type(day_subtree, 
                                                        plot_manual=plot_manual_fluo,
                                                        _element_trial_level =_element_trial_level,)
            for type_id, type_name in enumerate(selected_trial_types):
                if type_name not in type2dataset:
                    axs[type_id, day_index].remove()
                    continue
                raw_type_dataset = type2dataset[type_name]
                type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                specific_plotter = stack_view(
                    ax=axs[type_id, day_index], datasets=raw_type_dataset, sync_events=alignment_events, 
                    plot_manual=plot_manual_fluo
                ) if type_name != "SubtractOnly" else subtract_view(
                    ax=axs[type_id, day_index], 
                    datasets=[type2dataset["PuffOnly"], type2dataset["BlankOnly"]], sync_events=alignment_events, 
                    subtract_manual=SUBTRACT_MANUAL(name1="PuffOnly", name2="BlankOnly"), plot_manual=plot_manual_fluo,
                )
                coroutines[specific_plotter] = f"day{day_index}_{type_name}_stack_view" \
                    if type_name != "Subtract" else f"day{day_index}_{type_name}_subtract_view"

                if type_name == "SubtractOnly":
                    display_name = "Diff"
                elif type_name == "PuffOnly":
                    display_name = "Puff"
                elif type_name == "BlankOnly":
                    display_name = "Blank"
                else:
                    display_name = type_name
                axs[0, day_index].set_title(f"Day {day_index+1}")
                axs[type_id, 0].set_ylabel(display_name)
                
        
        coroutine_cycle(coroutines)
        for i in range(1, 11):
            for ax in axs[:, i].flatten():
                ax.set_yticks([])

        save_path = routing.default_fig_path(dataset, "HeatmapOverviewTraceSimple_PassiveDays_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
        default_exit_save(fig, save_path, _transparent=True)
    
    # plot_daywise_summary_sort_by_basic()
    # plot_daywise_summary_trace_simple_only()
    # plot_daywise_summary_trace_simple_only_passive()


def visualize_passive_reproduce_mo_figures(
        dataset: DataSet,

        theme_color: str,
        theme_name: str,
        theme_scale: float,
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def combine_trial_nodes(trial_nodes: DataSet) -> Node:
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes], 0).fusion()
        avg_node = sync_trial_nodes.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo.v + 1, t=group_fluo.t)
        avg_node.data.timeline = group_timeline
        return avg_node
    
    def combine_trial_node_delta(trial_nodes1: DataSet, trial_nodes2: DataSet) -> Node:
        sync_trial_nodes1 = sync_nodes(trial_nodes1, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo1 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes1], 
                                            baseline_subtraction=None).mean_ts
        sync_trial_nodes2 = sync_nodes(trial_nodes2, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo2 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes2], 
                                            baseline_subtraction=None,
                                            _predefined_t=group_fluo1.t).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes1] + 
                             [trial.data.timeline for trial in sync_trial_nodes2], 0).fusion()
        avg_node = sync_trial_nodes1.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo1.v - group_fluo2.v + 1, t=group_fluo1.t)
        avg_node.data.timeline = group_timeline
        return avg_node
    
    # fig, axs = plt.subplots(6, 5, figsize=(5.5, 7), constrained_layout=True)
    # passive1_trials = dataset.select(_element_trial_level, _self= lambda x: "P1" in x.session_id)
    # passive2_trials = dataset.select(_element_trial_level, _self= lambda x: "P2" in x.session_id)
    # passive_all_trials = dataset.select(_element_trial_level, _self= lambda x: "P1" in x.session_id or "P2" in x.session_id)

    # for ax in axs.flatten():
    #     ax.set_yticks([])
    #     ax.set_xlim(-1, 2)  
    #     ax.set_xticks([0, 1])
    #     ax.spines[["top", "right"]].set_visible(False)

    # coroutines = {}
    # for row_id, (passive_trials, row_name) in enumerate(zip([passive1_trials, passive2_trials, passive_all_trials],
    #                                                         ["Passive1", "Passive2", "Passve1+2"])):
    #     acc456 = passive_trials.select("trial", _self=lambda x: get_day_index(x) in (4, 5, 6))
    #     training12345 = passive_trials.select("trial", _self=lambda x: get_day_index(x) in (7, 8, 9, 10, 11))

    #     for col_id, training_dayindex in enumerate(range(7, 12)):
    #         training_single_day = training12345.select("trial", _self=lambda x: get_day_index(x) == training_dayindex)
    #         if len(training_single_day) == 0:
    #             for trial_id in range(2):
    #                 axs[row_id + trial_id * 3, col_id].remove()
    #             continue
    #         for trial_id, (trial_type, trial_setup) in enumerate(zip(["PuffOnly", "BlankOnly"], [{"ls": "-", }, {"ls": "--", }])):
    #             specific_plotter = subtract_view(
    #                 ax = axs[row_id + trial_id * 3, col_id],
    #                 datasets = [acc456.select("trial", _self=lambda x: x.info.get("trial_type") == trial_type),
    #                             training_single_day.select("trial", _self=lambda x: x.info.get("trial_type") == trial_type),],
    #                 subtract_manual=SUBTRACT_MANUAL(name1="ACC", color1="gray", name2="Training", color2=theme_color, settings1=trial_setup, settings2=trial_setup),
    #                 plot_manual=plot_manual_fluo, sync_events=alignment_events,
    #             )
    #             coroutines[specific_plotter] = f"{row_name}_day{training_dayindex}_{trial_type}_subtract_view"
    #             axs[row_id + trial_id * 3, col_id].set_ylabel(f"{row_name}", fontsize=8)
    #             axs[row_id + trial_id * 3, col_id].set_title(f"Day {training_dayindex-6}")
    # coroutine_cycle(coroutines)
    # for ax in axs[:, 1:].flatten():
    #     ax.set_yticks([])
    # for ax in axs.flatten():
    #     ax.set_ylim(0, theme_scale)
    # save_path = routing.default_fig_path(dataset, "Puff_evoked_trace_passive_in_MO_style_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    # default_exit_save(fig, save_path, _transparent=True)


    # fig, axs = plt.subplots(2, 5, figsize=(5.5, 7/3), constrained_layout=True)
    # training_trials = dataset.select(_element_trial_level, _self= lambda x: "P1" not in x.session_id and "P2" not in x.session_id)

    # for ax in axs.flatten():
    #     ax.set_yticks([])
    #     ax.set_xlim(-1, 2)  
    #     ax.set_xticks([0, 1])
    #     ax.spines[["top", "right"]].set_visible(False)

    # coroutines = {}
    # acc456 = passive_all_trials.select("trial", _self=lambda x: get_day_index(x) in (4, 5, 6))
    # training12345 = training_trials.select("trial", _self=lambda x: get_day_index(x) in (7, 8, 9, 10, 11))
    # for col_id, training_dayindex in enumerate(range(7, 12)):
    #     training_single_day = training12345.select("trial", _self=lambda x: get_day_index(x) == training_dayindex)
    #     for trial_id, (acc_trial_type, training_trial_types, trial_setup) in enumerate(zip(
    #         ["PuffOnly", "BlankOnly"],
    #         [("CuePuffWater", "CuePuffNoWater"), ("CueBlankWater", "CueBlankNoWater")], 
    #         [{"ls": "-", }, {"ls": "--", }])):
    #         specific_plotter = subtract_view(
    #             ax = axs[trial_id, col_id],
    #             datasets = [acc456.select("trial", _self=lambda x: x.info.get("trial_type") == acc_trial_type),
    #                         training_single_day.select("trial", _self=lambda x: x.info.get("trial_type") in training_trial_types),],
    #             subtract_manual=SUBTRACT_MANUAL(name1="ACC", color1="gray", name2="Training", color2=theme_color, settings1=trial_setup, settings2=trial_setup),
    #             plot_manual=plot_manual_fluo, sync_events=alignment_events,
    #         )
    #         coroutines[specific_plotter] = f"training_day{training_dayindex}_{acc_trial_type}_subtract_view"
    #         axs[trial_id, col_id].set_ylabel(f"Training", fontsize=8)
    #         axs[trial_id, col_id].set_title(f"Day {training_dayindex-6}")
    # coroutine_cycle(coroutines)
    # for ax in axs[:, 1:].flatten():
    #     ax.set_yticks([])
    # for ax in axs.flatten():
    #     ax.set_ylim(0, theme_scale)
    # save_path = routing.default_fig_path(dataset, "Puff_evoked_trace_training_in_MO_style_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    # default_exit_save(fig, save_path, _transparent=True)



    
    for metric_func, metric_name in zip([PEAK_VALUE, ], ["Peak", ]):
            
        fig, axs = plt.subplots(8, 2, figsize=(6, 20), constrained_layout=True)
        passive_all_trials = dataset.select(_element_trial_level, _self= lambda x: "P1" in x.session_id or "P2" in x.session_id)
        training_trials = dataset.select(_element_trial_level, _self= lambda x: "P1" not in x.session_id and "P2" not in x.session_id)

        acc456 = sync_nodes(passive_all_trials.select("trial", _self=lambda x: get_day_index(x) in (4, 5, 6)),
                            alignment_events, plot_manual=plot_manual_fluo)
        passive12345 = sync_nodes(passive_all_trials.select("trial", _self=lambda x: get_day_index(x) in (7, 8, 9, 10, 11)),
                            alignment_events, plot_manual=plot_manual_fluo)
        training12345 = sync_nodes(training_trials.select("trial", _self=lambda x: get_day_index(x) in (7, 8, 9, 10, 11)),
                            alignment_events, plot_manual=plot_manual_fluo)

        
        def get_fluo_metric(trial_nodes: DataSet, range_sec: Tuple[float, float]) -> np.ndarray:
            total_metrics = [metric_func(trial_node.fluorescence.df_f0, segment_period=range_sec) for trial_node in trial_nodes]
            return np.array(total_metrics)
        
        def get_fluo_metric_tuple(trial_nodes: DataSet, range_sec: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
            total_metrics = get_fluo_metric(trial_nodes, range_sec)
            metrics_per_mice = [np.nanmean(get_fluo_metric(trial_nodes.select("trial", 
                            _self=lambda x: x.mice_id == mice_node.mice_id), range_sec=range_sec)) 
                            for mice_node in dataset.select("mice")]
            return total_metrics, np.array(metrics_per_mice)

        
        def nan_sem(data: np.ndarray) -> float:
            if len(data) == 0:
                return np.nan
            return np.nanstd(data) / np.sqrt(np.sum(~np.isnan(data)))
        
        def plot_single_bar(ax, x_position: float, metric_data: np.ndarray, metric_data_per_mice: np.ndarray, 
                            bar_kwargs: dict, scatter_kwargs: dict):
            ax.bar(
                x_position, np.nanmean(metric_data), yerr=nan_sem(metric_data),
                **bar_kwargs
            )
            ax.scatter(
                [x_position for _ in range(len(metric_data_per_mice))],
                metric_data_per_mice,
                **scatter_kwargs
            )
        early_split = 0.5
        universal_scatter_kwargs = {"edgecolor": 'white', "alpha": 0.7, "s": 10, "linewidths": 0.5, "zorder": 5}
        for ax in axs[:, 0].flatten():
            ax.set_title("Passive", fontsize=20)
        for ax in axs[:, 1].flatten():
            ax.set_title("Training", fontsize=20)
        for col_id, (later_dataset12345, later_trial_types) in enumerate(zip(
            [passive12345, training12345], 
            [[("PuffOnly",), ("BlankOnly",)], 
            [("CuePuffWater", "CuePuffNoWater"), 
            ("CueBlankWater", "CueBlankNoWater")]])):
            for row_big_id, (early_trial_names, late_trial_names) in enumerate(zip(
                ("PuffOnly", "BlankOnly"),
                later_trial_types
            )):
                early_trials = acc456.select("trial", _self=lambda x: x.info.get("trial_type") == early_trial_names)
                later_trials = later_dataset12345.select("trial", _self=lambda x: x.info.get("trial_type") in late_trial_names)
                row_offset = 4 * row_big_id
                acc456_baseline = np.nanmean(get_fluo_metric(early_trials, range_sec=(0, 2)))
                acc456_baseline_components = [np.nanmean(get_fluo_metric(early_trials, range_sec=(0, early_split))),
                                              np.nanmean(get_fluo_metric(early_trials, range_sec=(early_split, 2)))]

                for i, row_name_str in enumerate(("", " Normalized", " Early vs Late", "E. vs L. normalized")):
                    axs[row_offset + i, col_id].set_ylabel(early_trial_names[:-4] + row_name_str, fontsize=16)
                xticks, xtick_labels = [], []

                for i in (4, 5, 6):
                    peaks, peaks_per_mice = get_fluo_metric_tuple(early_trials.select("trial", 
                            _self=lambda x: get_day_index(x) == i), range_sec=(0, 2))
                    plot_single_bar(
                        ax=axs[row_offset, col_id], x_position=i, metric_data=peaks, metric_data_per_mice=peaks_per_mice,
                        bar_kwargs={"color": "gray", "width": 0.8, "edgecolor": 'white',},
                        scatter_kwargs={"color": 'gray'} | universal_scatter_kwargs,)
                    
                    axs[row_offset, col_id].axhline(acc456_baseline, color="gray", ls="--", lw=0.5, zorder=10)
                    xticks.append(i)
                    xtick_labels.append(f"ACC{i}")

                    peaks_normalized = peaks / acc456_baseline
                    peaks_per_mice_normalized = peaks_per_mice / acc456_baseline
                    plot_single_bar(
                        ax=axs[row_offset+1, col_id], x_position=i, metric_data=peaks_normalized, 
                        metric_data_per_mice=peaks_per_mice_normalized,
                        bar_kwargs={"color": "gray", "width": 0.8, "edgecolor": 'white',},
                        scatter_kwargs={"color": 'gray'} | universal_scatter_kwargs,)
                    
                    axs[row_offset + 1, col_id].axhline(1, color="gray", ls="--", lw=0.5, zorder=10)

                    for component_idx, component_range in enumerate(((0, early_split), (early_split, 2))):
                        component_peaks, component_peaks_per_mice = get_fluo_metric_tuple(early_trials.select("trial", 
                            _self=lambda x: get_day_index(x) == i), range_sec=component_range)
                        plot_single_bar(
                            ax=axs[row_offset + 2, col_id], 
                            x_position=i + (component_idx - 0.5) * 0.4, 
                            metric_data=component_peaks, metric_data_per_mice=component_peaks_per_mice,
                            bar_kwargs={"color": "gray", "width": 0.35, "edgecolor": 'white', "alpha": 0.5+component_idx*0.4},
                            scatter_kwargs={"color": 'gray'} | universal_scatter_kwargs,)
                        
                        axs[row_offset + 2, col_id].axhline(acc456_baseline_components[component_idx], 
                                                            alpha=0.5 + component_idx*0.4, color="gray", ls="--", lw=0.5, zorder=10)
                        
                        componenet_peaks_normalized = component_peaks / acc456_baseline_components[component_idx]
                        component_peaks_per_mice_normalized = component_peaks_per_mice / acc456_baseline_components[component_idx]
                        plot_single_bar(
                            ax=axs[row_offset + 3, col_id],
                            x_position=i + (component_idx - 0.5) * 0.4,
                            metric_data=componenet_peaks_normalized, metric_data_per_mice=component_peaks_per_mice_normalized,
                            bar_kwargs={"color": "gray", "width": 0.35, "edgecolor": 'white', "alpha": 0.5+component_idx*0.4},
                            scatter_kwargs={"color": 'gray'} | universal_scatter_kwargs,)
                        
                        axs[row_offset + 3, col_id].axhline(1, alpha=0.5 + component_idx*0.4, color="gray", ls="--", lw=0.5, zorder=10)

                for i in (7, 8, 9, 10, 11):
                    peaks, peaks_per_mice = get_fluo_metric_tuple(later_trials.select("trial", 
                            _self=lambda x: get_day_index(x) == i), range_sec=(0, 2))
                    plot_single_bar(
                        ax=axs[row_offset, col_id], x_position=i, metric_data=peaks, metric_data_per_mice=peaks_per_mice,
                        bar_kwargs={"color": "black", "width": 0.8, "edgecolor": 'white',},
                        scatter_kwargs={"color": "black"} | universal_scatter_kwargs,)
                    
                    xticks.append(i)
                    xtick_labels.append(f"{theme_name}{i-6}")

                    peaks_normalized = peaks / acc456_baseline
                    peaks_per_mice_normalized = peaks_per_mice / acc456_baseline
                    plot_single_bar(
                        ax=axs[row_offset+1, col_id], x_position=i, metric_data=peaks_normalized, 
                        metric_data_per_mice=peaks_per_mice_normalized,
                        bar_kwargs={"color": "black", "width": 0.8, "edgecolor": 'white',},
                        scatter_kwargs={"color": "black"} | universal_scatter_kwargs,)

                    for component_idx, component_range in enumerate(((0, early_split), (early_split, 2))):
                        component_peaks, component_peaks_per_mice = get_fluo_metric_tuple(later_trials.select("trial", 
                            _self=lambda x: get_day_index(x) == i), range_sec=component_range)
                        plot_single_bar(
                            ax=axs[row_offset + 2, col_id], 
                            x_position=i + (component_idx - 0.5) * 0.4, 
                            metric_data=component_peaks, metric_data_per_mice=component_peaks_per_mice,
                            bar_kwargs={"color": "black", "width": 0.35, "edgecolor": 'white', "alpha": 0.5+component_idx*0.4},
                            scatter_kwargs={"color": "black"} | universal_scatter_kwargs,)
                        
                        componenet_peaks_normalized = component_peaks / acc456_baseline_components[component_idx]
                        component_peaks_per_mice_normalized = component_peaks_per_mice / acc456_baseline_components[component_idx]
                        plot_single_bar(
                            ax=axs[row_offset + 3, col_id], 
                            x_position=i + (component_idx - 0.5) * 0.4, 
                            metric_data=componenet_peaks_normalized, metric_data_per_mice=component_peaks_per_mice_normalized,
                            bar_kwargs={"color": "black", "width": 0.35, "edgecolor": 'white', "alpha": 0.5+component_idx*0.4},
                            scatter_kwargs={"color": "black"} | universal_scatter_kwargs,)

                
                for ax in axs[row_offset:row_offset+4, col_id]:
                    ax.set_xticks(xticks, xtick_labels, fontsize=6)
        for ax in axs.flatten():
            # ax.set_ylim(0, 0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axvspan(6.5, 11.5, color="lightgray", alpha=0.05, zorder=-20, lw=0)
            ax.set_ylim(0, 1.5)
        for ax in axs[1::2, :].flatten():
            ax.set_ylim(0, 3.5)
        save_path = routing.default_fig_path(dataset, f"Puff_evoked_bar_{metric_name}_training_in_MO_style_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
        default_exit_save(fig, save_path, _transparent=False)





def visualize_celluar_activity_session_wise(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)
    plot_manual_all = PlotManual(fluorescence=True, lick=True, locomotion=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def combine_trial_nodes(trial_nodes: DataSet) -> Node:
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes], 0).fusion()
        group_lick = grouping_events_rate([single_trial.data.lick for single_trial in sync_trial_nodes], bin_size=LICK_BIN_SIZE).mean_ts.to_events()
        group_locomotion = grouping_events_rate([single_trial.data.locomotion for single_trial in sync_trial_nodes], bin_size=LOCOMOTION_BIN_SIZE).mean_ts.to_events()
        avg_node = sync_trial_nodes.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo.v + 1, t=group_fluo.t)
        avg_node.data.timeline = group_timeline
        avg_node.data.lick = group_lick
        avg_node.data.locomotion = group_locomotion
        return avg_node
    
    def combine_trial_node_delta(trial_nodes1: DataSet, trial_nodes2: DataSet) -> Node:
        sync_trial_nodes1 = sync_nodes(trial_nodes1, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo1 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes1], 
                                            baseline_subtraction=None).mean_ts
        group_lick1 = grouping_events_rate([single_trial.data.lick for single_trial in sync_trial_nodes1], bin_size=LICK_BIN_SIZE).mean_ts
        group_locomotion1 = grouping_events_rate([single_trial.data.locomotion for single_trial in sync_trial_nodes1], bin_size=LOCOMOTION_BIN_SIZE).mean_ts
        sync_trial_nodes2 = sync_nodes(trial_nodes2, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo2 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes2], 
                                            baseline_subtraction=None,
                                            _predefined_t=group_fluo1.t).mean_ts
        group_lick2 = grouping_events_rate([single_trial.data.lick for single_trial in sync_trial_nodes2], bin_size=LICK_BIN_SIZE,
                                           _predefined_bin_centers=group_lick1.t).mean_ts
        group_locomotion2 = grouping_events_rate([single_trial.data.locomotion for single_trial in sync_trial_nodes2], bin_size=LOCOMOTION_BIN_SIZE,
                                                _predefined_bin_centers=group_locomotion1.t).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes1] + 
                             [trial.data.timeline for trial in sync_trial_nodes2], 0).fusion()
        
        avg_node = sync_trial_nodes1.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo1.v - group_fluo2.v + 1, t=group_fluo1.t)
        avg_node.data.timeline = group_timeline
        avg_node.data.lick = TimeSeries(v=group_lick1.v - group_lick2.v, t=group_lick1.t).to_events()
        avg_node.data.locomotion = TimeSeries(v=group_locomotion1.v - group_locomotion2.v, t=group_locomotion1.t).to_events()
        return avg_node
    
    def get_session_index(cs_node: Node) -> int:
        session_index_str = cs_node.session_id.split("_")[3]
        if session_index_str == "P1":
            return 0
        elif session_index_str == "P2":
            return 6
        else:
            assert session_index_str.startswith("S"), f"Unexpected session id format: {cs_node.session_id}"
            return int(session_index_str[1:])
    

    plotting_nodes = []

    for mice_idx, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        training_day_range = list(range(5, 12)) if mice_node.mice_id != "SUS6F" else (5, 6, 7, 8, 10, 11, 12)

        for day_node in mice_subtree.select("day"):
            print(f"Processing {mice_node.mice_id} Day {day_node.day_id}")
            if not int(day_node.day_id) in training_day_range:
                continue
            training_day_index = training_day_range.index(int(day_node.day_id))
            day_subtree = mice_subtree.subtree(day_node)
            for cell_session_node in day_subtree.select("cellsession"):
                if "P1" in cell_session_node.session_id or "P2" in cell_session_node.session_id:
                    all_puff_only_trials = day_subtree.subtree(cell_session_node).select(
                        _element_trial_level,
                        _self=lambda x: x.info.get("trial_type") in ("PuffOnly",),
                    )
                    puff_only_avg_node = combine_trial_nodes(all_puff_only_trials)
                    puff_only_avg_node.info["trial_type"] = "PuffOnly"
                    puff_only_avg_node.info["day_index"] = training_day_index
                    puff_only_avg_node.info["session_index"] = get_session_index(cell_session_node)

                    all_blank_only_trials = day_subtree.subtree(cell_session_node).select(
                        _element_trial_level,
                        _self=lambda x: x.info.get("trial_type") in ("BlankOnly",),
                    )   
                    blank_only_avg_node = combine_trial_nodes(all_blank_only_trials)
                    blank_only_avg_node.info["trial_type"] = "BlankOnly"
                    blank_only_avg_node.info["day_index"] = training_day_index
                    blank_only_avg_node.info["session_index"] = get_session_index(cell_session_node)
                    
                    delta_only_avg_node = combine_trial_node_delta(all_puff_only_trials, all_blank_only_trials)
                    delta_only_avg_node.info["trial_type"] = "SubtractOnly"
                    delta_only_avg_node.info["day_index"] = training_day_index
                    delta_only_avg_node.info["session_index"] = get_session_index(cell_session_node)
                    plotting_nodes.extend([puff_only_avg_node, blank_only_avg_node, delta_only_avg_node])
                
                elif training_day_index >= 2:  
                    all_puff_trials = day_subtree.subtree(cell_session_node).select(
                        _element_trial_level, 
                        _self=lambda x: x.info.get("trial_type") in ("CuePuffWater", "CuePuffNoWater"),
                    )
                    puff_avg_node = combine_trial_nodes(all_puff_trials)
                    puff_avg_node.info["trial_type"] = "CuePuff"
                    puff_avg_node.info["day_index"] = training_day_index
                    puff_avg_node.info["session_index"] = get_session_index(cell_session_node)
                    
                    all_blank_trials = day_subtree.subtree(cell_session_node).select(
                        _element_trial_level, 
                        _self=lambda x: x.info.get("trial_type") in ("CueBlankWater", "CueBlankNoWater"),
                    )
                    blank_avg_node = combine_trial_nodes(all_blank_trials)
                    blank_avg_node.info["trial_type"] = "CueBlank"
                    blank_avg_node.info["day_index"] = training_day_index
                    blank_avg_node.info["session_index"] = get_session_index(cell_session_node)

                    delta_avg_node = combine_trial_node_delta(all_puff_trials, all_blank_trials)
                    delta_avg_node.info["trial_type"] = "Subtract"
                    delta_avg_node.info["day_index"] = training_day_index
                    delta_avg_node.info["session_index"] = get_session_index(cell_session_node)
                    plotting_nodes.extend([puff_avg_node, blank_avg_node, delta_avg_node])
                else:                
                    all_cue_water_trials = day_subtree.subtree(cell_session_node).select(
                        _element_trial_level,
                        _self=lambda x: x.info.get("trial_type") in ("CueWater",),
                    )
                    cue_water_avg_node = combine_trial_nodes(all_cue_water_trials)
                    cue_water_avg_node.info["trial_type"] = "CueWater"
                    cue_water_avg_node.info["day_index"] = training_day_index
                    cue_water_avg_node.info["session_index"] = get_session_index(cell_session_node)

                    all_cue_nowater_trials = day_subtree.subtree(cell_session_node).select(
                        _element_trial_level,
                        _self=lambda x: x.info.get("trial_type") in ("CueNoWater",),
                    )
                    cue_nowater_avg_node = combine_trial_nodes(all_cue_nowater_trials)
                    cue_nowater_avg_node.info["trial_type"] = "CueNoWater"
                    cue_nowater_avg_node.info["day_index"] = training_day_index
                    cue_nowater_avg_node.info["session_index"] = get_session_index(cell_session_node)
                    
                    delta_cue_avg_node = combine_trial_node_delta(all_cue_water_trials, all_cue_nowater_trials)
                    delta_cue_avg_node.info["trial_type"] = "SubtractCue"
                    delta_cue_avg_node.info["day_index"] = training_day_index
                    delta_cue_avg_node.info["session_index"] = get_session_index(cell_session_node)
                    plotting_nodes.extend([cue_water_avg_node, cue_nowater_avg_node, delta_cue_avg_node])
          
        
    plotting_dataset = DataSet(name="plotting_dataset", nodes=plotting_nodes)

    def plotting1():
        # selected_trial_types = ["CuePuff", "CueBlank", "Subtract"]
        n_col = 3
        n_days = 4
        fig, axs = plt.subplots(2 * n_days * (len(dataset.select("mice")) + 1), 7 * n_col, 
                                    figsize=(20, 25), sharex=False, sharey=False, constrained_layout=True,
                                    height_ratios=[1.5, 1,] * n_days * len(dataset.select("mice")) + [2, 1] * n_days)
        for ax in axs.flatten():
            ax.tick_params(axis='y', labelleft=True)
            ax.set_yticks([])
            ax.set_xlim(-4, 4)  
            ax.set_xticks([])
        for ax in axs[1::2, :].flatten():
            ax.set_xticks([-2, 0, 2])
        
        coroutines = {} 
        row_datasets = {mice_node.mice_id: plotting_dataset.select("trial", mice_id=mice_node.mice_id) for mice_node in dataset.select("mice")}
        row_datasets["All mice"] = plotting_dataset
        
        for row_idx, (row_name, plotting_dataset_subtree) in enumerate(row_datasets.items()):
            

            sorting_day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: (x.info.get("day_index") == n_days - 1 and 
                                                                                            x.info.get("session_index") == 5 and
                                                                                x.info.get("trial_type") == "CueBlank"))
            sorting_cell_obj_uids = [node.object_uid for node in sorting_day_subtree]
            sync_sorting_day_subtree = sync_nodes(sorting_day_subtree, alignment_events, plot_manual=plot_manual_fluo)
            fluo_adv_ts = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_sorting_day_subtree], 
                                            baseline_subtraction=None)
            sorting_order = get_amplitude_sorted_idxs(fluo_adv_ts, 
                                                    amplitude_range=(0, 0.5),
                                                    _mono_decreasing=True)
            
            for day_index in range(n_days):

                row_id_offset = 2*n_days*row_idx + 2*day_index
                axs[row_id_offset + 1, 0].set_ylabel(f"{row_name}\nDay {day_index+1}" if day_index == 0 else f"\nDay {day_index+1}", fontsize=10)

                for session_index in range(7):
                    session_subtree = plotting_dataset_subtree.select(
                        "trial", _self=lambda x: (x.info.get("day_index") == day_index) and (x.info.get("session_index") == session_index)
                    )

                    col_id_offset = n_col *session_index
                    if session_index in (0, 6):
                        selected_trial_types = ["PuffOnly", "BlankOnly", "SubtractOnly"]
                    elif day_index >= 2:
                        selected_trial_types = ["CuePuff", "CueBlank", "Subtract"]
                    else:
                        selected_trial_types = ["CueWater", "CueNoWater", "SubtractCue"]
                    type2dataset = split_dataset_by_trial_type(session_subtree, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                    for type_id, type_name in enumerate(selected_trial_types):
                        if type_name not in type2dataset:
                            for ax in axs[row_id_offset: row_id_offset+2, col_id_offset+type_id].flatten():
                                ax.remove()
                            continue

                        raw_type_dataset = type2dataset[type_name]
                        type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                    
                        assert set(node.object_uid for node in type_dataset).issubset(set(node.object_uid for node in sorting_day_subtree)), \
                            f"Object uid mismatch between sorting day and other days, got \n" \
                            f"{[node.object_uid.cell_id for node in type_dataset]} \n{[node.object_uid.cell_id for node in sorting_day_subtree]}" 
                        raw_speicifc_sorting_order = [sorting_order[sorting_cell_obj_uids.index(node.object_uid)] for node in type_dataset]
                        raw_rank = {x: i for i, x in enumerate(sorted(raw_speicifc_sorting_order))}
                        speicifc_sorting_order = [raw_rank[rank] for rank in raw_speicifc_sorting_order]

                        specific_plotter = heatmap_view(
                            ax=axs[row_id_offset, col_id_offset+type_id], datasets=type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=speicifc_sorting_order
                        )
                        coroutines[specific_plotter] = f"{row_name}_day{day_index}_s{session_index}_{type_name}_heatmap_view"
                        
                        specific_plotter = stack_view(
                            ax=axs[row_id_offset+1, col_id_offset+type_id], datasets=raw_type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo
                        ) if "Subtract" not in type_name else subtract_view(
                            ax=axs[row_id_offset+1, col_id_offset+type_id], 
                            datasets=[type2dataset[selected_trial_types[0]], type2dataset[selected_trial_types[1]]], sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1=selected_trial_types[0], name2=selected_trial_types[1]), plot_manual=plot_manual_fluo,
                        )
                        coroutines[specific_plotter] = f"{row_name}_day{day_index}_s{session_index}_{type_name}_stack_view" \
                            if "Subtract" not in type_name else f"{row_name}_day{day_index}_s{session_index}_{type_name}_subtract_view"

                        display_name = type_name
                        axs[row_id_offset, col_id_offset+type_id].set_title(f"Session {session_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")
            
        coroutine_cycle(coroutines)
        for i in range(1, n_col):
            for ax in axs[:, i::n_col].flatten():
                ax.set_yticks([])
        # for ax in axs[1::2, :].flatten():
        #     ax.set_ylim(0, 1.6)
            
        save_path = routing.default_fig_path(dataset, "HeatmapOverview_SessionsFirsttwoday" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
        default_exit_save(fig, save_path)


    def plotting2():
        n_days = 7
        n_col = 3
        n_sessions = 7
        for mice_idx, mice_node in enumerate(dataset.select("mice")):
            mice_subtree = dataset.subtree(mice_node)
            plotting_dataset_subtree = plotting_dataset.select("trial", mice_id=mice_node.mice_id)
            training_day_range = list(range(5, 12)) if mice_node.mice_id != "SUS6F" else (5, 6, 7, 8, 10, 11, 12)

            sorting_day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: (x.info.get("day_index") == n_days - 1 and 
                                                                                            x.info.get("session_index") == 5 and
                                                                                x.info.get("trial_type") == "CueBlank"))
            sorting_cell_obj_uids = [node.object_uid for node in sorting_day_subtree]
            sync_sorting_day_subtree = sync_nodes(sorting_day_subtree, alignment_events, plot_manual=plot_manual_fluo)
            fluo_adv_ts = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_sorting_day_subtree], 
                                            baseline_subtraction=None)
            sorting_order = get_amplitude_sorted_idxs(fluo_adv_ts, 
                                                    amplitude_range=(0, 0.5),
                                                    _mono_decreasing=True)

            for day_index in range(n_days):
                absolute_day = training_day_range[day_index]
                print(f"Processing {mice_node.mice_id} Day {absolute_day}")
                day_node = find_only_one(mice_subtree.select("day", day_id=lambda x: int(x) == absolute_day))
                day_subtree = mice_subtree.subtree(day_node)
                plotting_day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: x.info.get("day_index") == day_index)

                max_trial_nums = max([len(day_subtree.select("fovtrial", session_id=plot_node.session_id)) for plot_node in plotting_day_subtree])

                coroutines = {}
                fig, axs = plt.subplots(n_sessions*2, max_trial_nums + n_col, figsize=(35, 20), sharex=False, sharey=False, constrained_layout=True,
                                        height_ratios=[1.5, 1.2] * n_sessions)
                fig.suptitle(f"{mice_node.mice_id} Day {absolute_day}", fontsize=32)
                for ax in axs.flatten():
                    ax.tick_params(axis='y', labelleft=True)
                    ax.set_yticks([])
                    ax.set_xlim(-5, 5)  
                    ax.set_xticks([])
                for ax in axs[1::2, :].flatten():
                    ax.set_xticks([-3, 0, 3])

                for session_index in range(n_sessions):
                    plotting_session_subtree = plotting_day_subtree.select("trial", _self=lambda x: x.info.get("session_index") == session_index)
                    if len(plotting_session_subtree) == 0:
                        for ax in axs[2*session_index: 2*session_index+2, :].flatten():
                            ax.remove()
                        continue
                    session_id_str = plotting_session_subtree.nodes[0].session_id
                    session_subtree = day_subtree.select("trial", session_id=session_id_str)
                    axs[session_index*2, 0].set_ylabel(f"Session {session_index+1}", fontsize=10)

                    if session_index in (0, 6):
                        selected_trial_types = ["PuffOnly", "BlankOnly", "SubtractOnly"]
                    elif day_index >= 2:
                        selected_trial_types = ["CuePuff", "CueBlank", "Subtract"]
                    else:
                        selected_trial_types = ["CueWater", "CueNoWater", "SubtractCue"]
                    
                    fov_trial_subtrees = day_subtree.select("fovtrial", session_id=session_id_str)
                    print(f"Session {session_index+1} has {len(fov_trial_subtrees)} fov trials, max trial nums is {max_trial_nums}")
                    for trial_index, fovtrial_node in enumerate(fov_trial_subtrees):
                        trial_dataset = session_subtree.select("trial", chunk_id=fovtrial_node.chunk_id)

                        assert set(node.object_uid for node in trial_dataset).issubset(set(node.object_uid for node in sorting_day_subtree)), \
                            f"Object uid mismatch between sorting day and other days, got \n" \
                            f"{[node.object_uid.cell_id for node in trial_dataset]} \n{[node.object_uid.cell_id for node in sorting_day_subtree]}" 
                        raw_speicifc_sorting_order = [sorting_order[sorting_cell_obj_uids.index(node.object_uid)] for node in trial_dataset]
                        raw_rank = {x: i for i, x in enumerate(sorted(raw_speicifc_sorting_order))}
                        speicifc_sorting_order = [raw_rank[rank] for rank in raw_speicifc_sorting_order]

                        specific_plotter = heatmap_view(
                            ax=axs[2*session_index, trial_index], datasets=trial_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=speicifc_sorting_order
                        )
                        coroutines[specific_plotter] = f"Day{day_index}_Session{session_index}_Trial{trial_index}_heatmap_view"
                        
                        specific_plotter = stack_view(
                            ax=axs[2*session_index+1, trial_index], datasets=trial_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_all
                        ) 
                        coroutines[specific_plotter] = f"Day{day_index}_Session{session_index}_Trial{trial_index}_stack_view"

                        display_name = fovtrial_node.info.get('trial_type').replace("Cue", "").replace("Water", "W")
                        display_color = "Green" if fovtrial_node.info.get('trial_type') in ("CuePuffWater", "CuePuffNoWater", "PuffOnly", "CueWater") else "black"
                        axs[2*session_index, trial_index].set_title(f"Trial {trial_index+1}\n{display_name}", color = display_color)

                    for ax in axs[2*session_index: 2*session_index+2, len(fov_trial_subtrees):-n_col].flatten():
                        ax.remove()
                    type2dataset = split_dataset_by_trial_type(plotting_session_subtree, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                    for type_id, type_name in enumerate(selected_trial_types):
                        if type_name not in type2dataset:
                            for ax in axs[2*session_index:2*session_index+2, -n_col + type_id].flatten():
                                ax.remove()
                            continue

                        raw_type_dataset = type2dataset[type_name]
                        type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                    
                        assert set(node.object_uid for node in type_dataset).issubset(set(node.object_uid for node in sorting_day_subtree)), \
                            f"Object uid mismatch between sorting day and other days, got \n" \
                            f"{[node.object_uid.cell_id for node in type_dataset]} \n{[node.object_uid.cell_id for node in sorting_day_subtree]}" 
                        raw_speicifc_sorting_order = [sorting_order[sorting_cell_obj_uids.index(node.object_uid)] for node in type_dataset]
                        raw_rank = {x: i for i, x in enumerate(sorted(raw_speicifc_sorting_order))}
                        speicifc_sorting_order = [raw_rank[rank] for rank in raw_speicifc_sorting_order]

                        specific_plotter = heatmap_view(
                            ax=axs[2*session_index, -n_col + type_id], datasets=type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=speicifc_sorting_order
                        )
                        coroutines[specific_plotter] = f"Day{day_index}_Session{session_index}_{type_name}_heatmap_view"
                        
                        specific_plotter = stack_view(
                            ax=axs[2*session_index+1, -n_col + type_id], datasets=raw_type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_all
                        ) if "Subtract" not in type_name else subtract_view(
                            ax=axs[2*session_index+1, -n_col + type_id], 
                            datasets=[type2dataset[selected_trial_types[0]], type2dataset[selected_trial_types[1]]], sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1=selected_trial_types[0], name2=selected_trial_types[1]), plot_manual=plot_manual_all,
                        )
                        coroutines[specific_plotter] = f"Day{day_index}_Session{session_index}_{type_name}_stack_view" \
                            if "Subtract" not in type_name else f"Day{day_index}_Session{session_index}_{type_name}_subtract_view"

                        display_name = type_name.replace("Cue", "").replace("Water", "W")
                        axs[2*session_index, -n_col + type_id].set_title(f"{display_name}")
            
                progress = coroutine_cycle(coroutines)
                for ax in axs[:, 1:max_trial_nums].flatten():
                    ax.set_yticks([])
                for ax in axs[:, -n_col+1:].flatten():
                    ax.set_yticks([])
                for ax in axs[1::2, :].flatten():
                    ax.set_ylim(-1, progress)
                    
                save_path = routing.default_fig_path(day_node, "TrialLevelCellularActivity_SessionWise" + f"_{{}}_{_aligment_style}_Day{absolute_day}.png", fov_skip=True)
                default_exit_save(fig, save_path)
    plotting1()
    plotting2()



def visualize_celluar_activity_in_heatmap_PSE_Expanded(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def combine_trial_nodes(trial_nodes: DataSet) -> Node:
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes], 0).fusion()
        avg_node = sync_trial_nodes.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo.v + 1, t=group_fluo.t)
        avg_node.data.timeline = group_timeline
        return avg_node
    
    def combine_trial_node_delta(trial_nodes1: DataSet, trial_nodes2: DataSet) -> Node:
        sync_trial_nodes1 = sync_nodes(trial_nodes1, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo1 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes1], 
                                            baseline_subtraction=None).mean_ts
        sync_trial_nodes2 = sync_nodes(trial_nodes2, alignment_events, plot_manual=plot_manual_fluo)
        group_fluo2 = grouping_timeseries([single_trial.data.fluorescence.df_f0
                                                for single_trial in sync_trial_nodes2], 
                                            baseline_subtraction=None,
                                            _predefined_t=group_fluo1.t).mean_ts
        group_timeline = sum([trial.data.timeline for trial in sync_trial_nodes1] + 
                             [trial.data.timeline for trial in sync_trial_nodes2], 0).fusion()
        avg_node = sync_trial_nodes1.nodes[0].shadow_clone()
        avg_node.data.fluorescence.raw_f = TimeSeries(v=group_fluo1.v - group_fluo2.v + 1, t=group_fluo1.t)
        avg_node.data.timeline = group_timeline
        return avg_node
    
    plotting_nodes, allday_plotting_nodes = [], []

    for mice_idx, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        training_day_range = list(range(7, 12)) if mice_node.mice_id != "SUS6F" else (7, 8, 10, 11, 12)

        for day_node in mice_subtree.select("day"):
            print(f"Processing {mice_node.mice_id} Day {day_node.day_id}")
            if not int(day_node.day_id) in training_day_range:
                continue
            training_day_index = training_day_range.index(int(day_node.day_id))
            day_subtree = mice_subtree.subtree(day_node)
            for cellday_node in day_subtree.select("cellday"):
                puff_water_trials = day_subtree.subtree(cellday_node).select(
                    _element_trial_level, 
                    _self=lambda x: x.info.get("trial_type") == "CuePuffWater",
                )
                puff_water_avg_node = combine_trial_nodes(puff_water_trials)
                puff_water_avg_node.info["trial_type"] = "CuePuffWater"
                puff_water_avg_node.info["day_index"] = training_day_index
                
                puff_nowater_trials = day_subtree.subtree(cellday_node).select(
                    _element_trial_level, 
                    _self=lambda x: x.info.get("trial_type") == "CuePuffNoWater",
                )
                puff_nowater_avg_node = combine_trial_nodes(puff_nowater_trials)
                puff_nowater_avg_node.info["trial_type"] = "CuePuffNoWater"
                puff_nowater_avg_node.info["day_index"] = training_day_index

                blank_water_trials = day_subtree.subtree(cellday_node).select(
                    _element_trial_level, 
                    _self=lambda x: x.info.get("trial_type") == "CueBlankWater",
                )
                blank_water_avg_node = combine_trial_nodes(blank_water_trials)
                blank_water_avg_node.info["trial_type"] = "CueBlankWater"
                blank_water_avg_node.info["day_index"] = training_day_index
                
                blank_nowater_trials = day_subtree.subtree(cellday_node).select(
                    _element_trial_level, 
                    _self=lambda x: x.info.get("trial_type") == "CueBlankNoWater",
                )
                blank_nowater_avg_node = combine_trial_nodes(blank_nowater_trials)
                blank_nowater_avg_node.info["trial_type"] = "CueBlankNoWater"
                blank_nowater_avg_node.info["day_index"] = training_day_index
                
                delta_water_avg_node = combine_trial_node_delta(puff_water_trials, blank_water_trials)
                delta_water_avg_node.info["trial_type"] = "(Puff - Blank) Water"
                delta_water_avg_node.info["day_index"] = training_day_index

                delta_nowater_avg_node = combine_trial_node_delta(puff_nowater_trials, blank_nowater_trials)
                delta_nowater_avg_node.info["trial_type"] = "(Puff - Blank) NoWater"
                delta_nowater_avg_node.info["day_index"] = training_day_index

                delta_puff_avg_node = combine_trial_node_delta(puff_water_trials, puff_nowater_trials)
                delta_puff_avg_node.info["trial_type"] = "Puff (Water - NoWater)"
                delta_puff_avg_node.info["day_index"] = training_day_index
                
                delta_blank_avg_node = combine_trial_node_delta(blank_water_trials, blank_nowater_trials)
                delta_blank_avg_node.info["trial_type"] = "Blank (Water - NoWater)"
                delta_blank_avg_node.info["day_index"] = training_day_index
                
                plotting_nodes.extend([puff_water_avg_node, puff_nowater_avg_node, blank_water_avg_node, blank_nowater_avg_node, 
                                      delta_water_avg_node, delta_nowater_avg_node, delta_puff_avg_node, delta_blank_avg_node])
                
        for cell_nodes in mice_subtree.select("cell"):
            puff_water_trials = mice_subtree.subtree(cell_nodes).select(
                _element_trial_level, 
                _self=lambda x: x.info.get("trial_type") == "CuePuffWater",
            )
            puff_water_avg_node = combine_trial_nodes(puff_water_trials)
            puff_water_avg_node.info["trial_type"] = "CuePuffWater"
            
            puff_nowater_trials = mice_subtree.subtree(cell_nodes).select(
                _element_trial_level, 
                _self=lambda x: x.info.get("trial_type") == "CuePuffNoWater",
            )
            puff_nowater_avg_node = combine_trial_nodes(puff_nowater_trials)
            puff_nowater_avg_node.info["trial_type"] = "CuePuffNoWater"
            
            blank_water_trials = mice_subtree.subtree(cell_nodes).select(
                _element_trial_level, 
                _self=lambda x: x.info.get("trial_type") == "CueBlankWater",
            )
            blank_water_avg_node = combine_trial_nodes(blank_water_trials)
            blank_water_avg_node.info["trial_type"] = "CueBlankWater"
            
            blank_nowater_trials = mice_subtree.subtree(cell_nodes).select(
                _element_trial_level, 
                _self=lambda x: x.info.get("trial_type") == "CueBlankNoWater",
            )
            blank_nowater_avg_node = combine_trial_nodes(blank_nowater_trials)
            blank_nowater_avg_node.info["trial_type"] = "CueBlankNoWater"
            
            delta_water_avg_node = combine_trial_node_delta(puff_water_trials, blank_water_trials)
            delta_water_avg_node.info["trial_type"] = "(Puff - Blank) Water"
            
            delta_nowater_avg_node = combine_trial_node_delta(puff_nowater_trials, blank_nowater_trials)
            delta_nowater_avg_node.info["trial_type"] = "(Puff - Blank) NoWater"
            
            delta_puff_avg_node = combine_trial_node_delta(puff_water_trials, puff_nowater_trials)
            delta_puff_avg_node.info["trial_type"] = "Puff (Water - NoWater)"
            
            delta_blank_avg_node = combine_trial_node_delta(blank_water_trials, blank_nowater_trials)
            delta_blank_avg_node.info["trial_type"] = "Blank (Water - NoWater)"

            allday_plotting_nodes.extend([puff_water_avg_node, puff_nowater_avg_node, blank_water_avg_node, blank_nowater_avg_node, 
                                      delta_water_avg_node, delta_nowater_avg_node, delta_puff_avg_node, delta_blank_avg_node])
        
    plotting_dataset = DataSet(name="plotting_dataset", nodes=plotting_nodes)
    allday_plotting_dataset = DataSet(name="allday_plotting_dataset", nodes=allday_plotting_nodes)

    def plotting1():
        selected_trial_types = ["CuePuffWater", "CuePuffNoWater", "CueBlankWater", "CueBlankNoWater"]
        n_col = len(selected_trial_types)

        for sorting_day_index in (0, 4):
            fig, axs = plt.subplots(2 * (len(dataset.select("mice")) + 1), 6*n_col, 
                                    figsize=(18, 8), sharex=False, sharey=False, constrained_layout=True,
                                    height_ratios=[1.5, 1,] * len(dataset.select("mice")) + [2, 1])
            for ax in axs.flatten():
                ax.tick_params(axis='y', labelleft=True)
                ax.set_yticks([])
                ax.set_xlim(-5, 5)  
                ax.set_xticks([])
            for ax in axs[1::2, :].flatten():
                ax.set_xticks([-4, 0, 4])

            coroutines = {} 
            row_datasets = {mice_node.mice_id: (
                plotting_dataset.select("trial", mice_id=mice_node.mice_id),
                allday_plotting_dataset.select("trial", mice_id=mice_node.mice_id),
                ) for mice_node in dataset.select("mice")}
            row_datasets["All mice"] = (plotting_dataset, allday_plotting_dataset)
            for row_idx, (row_name, (plotting_dataset_subtree, allday_plotting_dataset_subtree)) in enumerate(row_datasets.items()):
                axs[row_idx*2, 0].set_ylabel(row_name, fontsize=10)

                sorting_day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: (x.info.get("day_index") == sorting_day_index and 
                                                                                    x.info.get("trial_type") == "CuePuffWater"))
                sync_sorting_day_subtree = sync_nodes(sorting_day_subtree, alignment_events, plot_manual=plot_manual_fluo)
                fluo_adv_ts = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                    for single_trial in sync_sorting_day_subtree], 
                                                baseline_subtraction=None)
                sorting_order = get_amplitude_sorted_idxs(fluo_adv_ts, amplitude_range=(0, 2))


                for day_index in range(5):
                    day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: x.info.get("day_index") == day_index)
                    
                    type2dataset = split_dataset_by_trial_type(day_subtree, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                    for type_id, type_name in enumerate(selected_trial_types):
                        if type_name not in type2dataset:
                            for ax in axs[2*row_idx:2*row_idx+2, day_index*n_col+type_id].flatten():
                                ax.remove()
                            continue
                        raw_type_dataset = type2dataset[type_name]
                        type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                    
                        assert [node.object_uid for node in type_dataset] == [node.object_uid for node in sorting_day_subtree], \
                            f"Object uid mismatch between sorting day and other days, got \n" \
                            f"{[node.object_uid.cell_id for node in type_dataset]} \n{[node.object_uid.cell_id for node in sorting_day_subtree]}"
                        

                        specific_plotter = heatmap_view(
                            ax=axs[2*row_idx, day_index*n_col+type_id], datasets=type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=sorting_order
                        )
                        coroutines[specific_plotter] = f"{row_name}_day{day_index}_{type_name}_heatmap_view"
                        
                        specific_plotter = stack_view(
                            ax=axs[2*row_idx+1, day_index*n_col+type_id], datasets=raw_type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo
                        )

                        coroutines[specific_plotter] = f"{row_name}_day{day_index}_{type_name}_stack_view"

                        display_name = type_name.replace("Water", "W").replace("NoW", "NW").replace("Cue", "")
                        axs[0, day_index*n_col+type_id].set_title(f"Day {day_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")
                        
                        axs[-2, day_index*n_col+type_id].set_title(f"Day {day_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")
                
                type2dataset = split_dataset_by_trial_type(allday_plotting_dataset_subtree, 
                                                            plot_manual=plot_manual_fluo,
                                                            _element_trial_level =_element_trial_level,)
                for type_id, type_name in enumerate(selected_trial_types):
                    if type_name not in type2dataset:
                        for ax in axs[2*row_idx:2*row_idx+2, -n_col+type_id].flatten():
                            ax.remove()
                        continue
                    raw_type_dataset = type2dataset[type_name]
                    type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                    specific_plotter = heatmap_view(
                        ax=axs[2*row_idx, -n_col+type_id], datasets=type_dataset, sync_events=alignment_events, 
                        plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=sorting_order
                    )
                    coroutines[specific_plotter] = f"{row_name}_allday_{type_name}_heatmap_view"
                    
                    specific_plotter = stack_view(
                        ax=axs[2*row_idx+1, -n_col+type_id], datasets=raw_type_dataset, sync_events=alignment_events, 
                        plot_manual=plot_manual_fluo
                    )

                    coroutines[specific_plotter] = f"{row_name}_allday_{type_name}_stack_view" 
                    
                    display_name = type_name.replace("Water", "W").replace("NoW", "NW").replace("Cue", "")
                    axs[0, -n_col+type_id].set_title(f"All days\n{display_name}" if type_id==1 else f"\n{display_name}")

                    axs[-2, -n_col+type_id].set_title(f"All days\n{display_name}" if type_id==1 else f"\n{display_name}")
                
            coroutine_cycle(coroutines)
            for i in range(1, n_col):
                for ax in axs[:, i::n_col].flatten():
                    ax.set_yticks([])
            for ax in axs[1::2, :].flatten():
                ax.set_ylim(0, 1.6)
        
            save_path = routing.default_fig_path(dataset, "HeatmapOverview_TrainingDays_PSEExpanded1" + f"_{{}}_{_aligment_style}_sortbyday{sorting_day_index + 1}.png", fov_skip=True)
            default_exit_save(fig, save_path)
        

    def plotting2():
        selected_trial_types = ["(Puff - Blank) Water", "(Puff - Blank) NoWater", 
                                "Puff (Water - NoWater)", "Blank (Water - NoWater)"]
        n_col = len(selected_trial_types)

        for sorting_day_index in (0, 4):
            fig, axs = plt.subplots(2 * (len(dataset.select("mice")) + 1), 6*n_col, 
                                    figsize=(18, 8), sharex=False, sharey=False, constrained_layout=True,
                                    height_ratios=[1.5, 1,] * len(dataset.select("mice")) + [2, 1])
            for ax in axs.flatten():
                ax.tick_params(axis='y', labelleft=True)
                ax.set_yticks([])
                ax.set_xlim(-5, 5)  
                ax.set_xticks([])
            for ax in axs[1::2, :].flatten():
                ax.set_xticks([-4, 0, 4])

            coroutines = {} 
            row_datasets = {mice_node.mice_id: (
                plotting_dataset.select("trial", mice_id=mice_node.mice_id),
                allday_plotting_dataset.select("trial", mice_id=mice_node.mice_id),
                ) for mice_node in dataset.select("mice")}
            row_datasets["All mice"] = (plotting_dataset, allday_plotting_dataset)
            for row_idx, (row_name, (plotting_dataset_subtree, allday_plotting_dataset_subtree)) in enumerate(row_datasets.items()):
                axs[row_idx*2, 0].set_ylabel(row_name, fontsize=10)

                sorting_day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: (x.info.get("day_index") == sorting_day_index and 
                                                                                    x.info.get("trial_type") == "CuePuffWater"))
                sync_sorting_day_subtree = sync_nodes(sorting_day_subtree, alignment_events, plot_manual=plot_manual_fluo)
                fluo_adv_ts = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                    for single_trial in sync_sorting_day_subtree], 
                                                baseline_subtraction=None)
                sorting_order = get_amplitude_sorted_idxs(fluo_adv_ts, amplitude_range=(0, 2))


                for day_index in range(5):
                    day_subtree = plotting_dataset_subtree.select("trial", _self=lambda x: x.info.get("day_index") == day_index)
                    
                    type2dataset = split_dataset_by_trial_type(day_subtree, 
                                                                plot_manual=plot_manual_fluo,
                                                                _element_trial_level =_element_trial_level,)
                    for type_id, type_name in enumerate(selected_trial_types):
                        if type_name not in type2dataset:
                            for ax in axs[2*row_idx:2*row_idx+2, day_index*n_col+type_id].flatten():
                                ax.remove()
                            continue
                        raw_type_dataset = type2dataset[type_name]
                        type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                    
                        assert [node.object_uid for node in type_dataset] == [node.object_uid for node in sorting_day_subtree], \
                            f"Object uid mismatch between sorting day and other days, got \n" \
                            f"{[node.object_uid.cell_id for node in type_dataset]} \n{[node.object_uid.cell_id for node in sorting_day_subtree]}"
                        

                        specific_plotter = heatmap_view(
                            ax=axs[2*row_idx, day_index*n_col+type_id], datasets=type_dataset, sync_events=alignment_events, 
                            plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=sorting_order
                        )
                        coroutines[specific_plotter] = f"{row_name}_day{day_index}_{type_name}_heatmap_view"
                        
                        if type_name == "(Puff - Blank) Water":
                            specific_plotter = subtract_view(
                                ax=axs[2*row_idx+1, day_index*n_col+type_id], 
                                datasets=[type2dataset["CuePuffWater"], type2dataset["CueBlankWater"]], sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1="CuePuffWater", name2="CueBlankWater"), plot_manual=plot_manual_fluo,
                            )
                        elif type_name == "(Puff - Blank) NoWater":
                            specific_plotter = subtract_view(
                                ax=axs[2*row_idx+1, day_index*n_col+type_id], 
                                datasets=[type2dataset["CuePuffNoWater"], type2dataset["CueBlankNoWater"]], sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1="CuePuffNoWater", name2="CueBlankNoWater"), plot_manual=plot_manual_fluo,
                            )
                        elif type_name == "Puff (Water - NoWater)":
                            specific_plotter = subtract_view(
                                ax=axs[2*row_idx+1, day_index*n_col+type_id], 
                                datasets=[type2dataset["CuePuffWater"], type2dataset["CuePuffNoWater"]], sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1="CuePuffWater", name2="CuePuffNoWater"), plot_manual=plot_manual_fluo,
                            )
                        elif type_name == "Blank (Water - NoWater)":
                            specific_plotter = subtract_view(
                                ax=axs[2*row_idx+1, day_index*n_col+type_id], 
                                datasets=[type2dataset["CueBlankWater"], type2dataset["CueBlankNoWater"]], sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1="CueBlankWater", name2="CueBlankNoWater"), plot_manual=plot_manual_fluo,
                            )

                        coroutines[specific_plotter] = f"{row_name}_day{day_index}_{type_name}_subtract_view"

                        display_name = type_name.replace("Water", "W").replace("NoW", "NW").replace("Cue", "").replace("Puff", "P").replace("Blank", "B")
                        axs[0, day_index*n_col+type_id].set_title(f"Day {day_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")
                        
                        axs[-2, day_index*n_col+type_id].set_title(f"Day {day_index+1}\n{display_name}" if type_id==1 else f"\n{display_name}")
                
                type2dataset = split_dataset_by_trial_type(allday_plotting_dataset_subtree, 
                                                            plot_manual=plot_manual_fluo,
                                                            _element_trial_level =_element_trial_level,)
                for type_id, type_name in enumerate(selected_trial_types):
                    if type_name not in type2dataset:
                        for ax in axs[2*row_idx:2*row_idx+2, -n_col+type_id].flatten():
                            ax.remove()
                        continue
                    raw_type_dataset = type2dataset[type_name]
                    type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                    specific_plotter = heatmap_view(
                        ax=axs[2*row_idx, -n_col+type_id], datasets=type_dataset, sync_events=alignment_events, 
                        plot_manual=plot_manual_fluo, modality_name="fluorescence", specified_order=sorting_order
                    )
                    coroutines[specific_plotter] = f"{row_name}_allday_{type_name}_heatmap_view"
                    
                    if type_name == "(Puff - Blank) Water":
                        specific_plotter = subtract_view(
                            ax=axs[2*row_idx+1, -n_col+type_id], 
                            datasets=[type2dataset["CuePuffWater"], type2dataset["CueBlankWater"]], sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1="CuePuffWater", name2="CueBlankWater"), plot_manual=plot_manual_fluo,
                        )
                    elif type_name == "(Puff - Blank) NoWater":
                        specific_plotter = subtract_view(
                            ax=axs[2*row_idx+1, -n_col+type_id], 
                            datasets=[type2dataset["CuePuffNoWater"], type2dataset["CueBlankNoWater"]], sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1="CuePuffNoWater", name2="CueBlankNoWater"), plot_manual=plot_manual_fluo,
                        )
                    elif type_name == "Puff (Water - NoWater)":
                        specific_plotter = subtract_view(
                            ax=axs[2*row_idx+1, -n_col+type_id], 
                            datasets=[type2dataset["CuePuffWater"], type2dataset["CuePuffNoWater"]], sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1="CuePuffWater", name2="CuePuffNoWater"), plot_manual=plot_manual_fluo,
                        )
                    elif type_name == "Blank (Water - NoWater)":
                        specific_plotter = subtract_view(
                            ax=axs[2*row_idx+1, -n_col+type_id], 
                            datasets=[type2dataset["CueBlankWater"], type2dataset["CueBlankNoWater"]], sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1="CueBlankWater", name2="CueBlankNoWater"), plot_manual=plot_manual_fluo,
                        )

                    coroutines[specific_plotter] = f"{row_name}_allday_{type_name}_subtract_view"
                    
                    display_name = type_name.replace("Water", "W").replace("NoW", "NW").replace("Cue", "").replace("Puff", "P").replace("Blank", "B")
                    axs[0, -n_col+type_id].set_title(f"All days\n{display_name}" if type_id==1 else f"\n{display_name}")

                    axs[-2, -n_col+type_id].set_title(f"All days\n{display_name}" if type_id==1 else f"\n{display_name}")
                
            coroutine_cycle(coroutines)
            for i in range(1, n_col):
                for ax in axs[:, i::n_col].flatten():
                    ax.set_yticks([])
            for ax in axs[1::2, :].flatten():
                ax.set_ylim(0, 1.6)
        
            save_path = routing.default_fig_path(dataset, "HeatmapOverview_TrainingDays_PSEExpanded2" + f"_{{}}_{_aligment_style}_sortbyday{sorting_day_index + 1}.png", fov_skip=True)
            default_exit_save(fig, save_path)
    
    
    plotting1()
    plotting2()




def visualize_early_celluar_activity_summary(
        dataset: DataSet,

        early_time_range: tuple[float, float] = (0., 0.5),
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def calculate_trial_early_activity_auc(trial_nodes: DataSet):
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        max_fs = max([single_trial.data.fluorescence.df_f0.fs for single_trial in sync_trial_nodes])
        predefined_t = np.linspace(early_time_range[0], early_time_range[1], 
                                   int((early_time_range[1] - early_time_range[0]) * max_fs * 2))
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None,
                                            _predefined_t=predefined_t)
        early_mask = (group_fluo.t >= early_time_range[0]) & (group_fluo.t <= early_time_range[1])
        return np.trapezoid(group_fluo.raw_array[:, early_mask], group_fluo.t[early_mask])
    
    fig, axs = plt.subplots(1, 5, figsize=(8, 2), sharex=True, sharey=True, constrained_layout=True)

    mice_colors = ("Green", "Purple", "Orange")
    all_puff_auc, all_blank_auc = defaultdict(list), defaultdict(list)
    for mice_node, mice_color in zip(dataset.select("mice"), mice_colors):
        mice_subtree = dataset.subtree(mice_node)
        for cell_node in mice_subtree.select("cell"):
            cell_subtree = mice_subtree.subtree(cell_node)
            training_day_range = list(range(7, 12)) if mice_node.mice_id != "SUS6F" else (7, 8, 10, 11, 12)

            for cellday_node in cell_subtree.select("cellday"):
                if int(cellday_node.day_id) not in training_day_range:
                    continue
                cd_subtree = cell_subtree.subtree(cellday_node)
                training_day_index = training_day_range.index(int(cellday_node.day_id))

                puff_trials = cd_subtree.select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffWater", "CuePuffNoWater"},
                )
                blank_trials = cd_subtree.select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CueBlankWater", "CueBlankNoWater"},
                )
                puff_auc = calculate_trial_early_activity_auc(puff_trials)
                blank_auc = calculate_trial_early_activity_auc(blank_trials)

                puff_auc_mean = np.nanmean(puff_auc)
                # puff_auc_sem = np.std(puff_auc) / np.sqrt(len(puff_auc))
                blank_auc_mean = np.nanmean(blank_auc)
                # blank_auc_sem = np.std(blank_auc) / np.sqrt(len(blank_auc))

                for i in range(5):
                    if i == training_day_index:                        
                        # axss[i].errorbar(puff_auc_mean, blank_auc_mean, xerr=puff_auc_sem, yerr=blank_auc_sem, 
                        #                  color=mice_color, lw=0.8,
                        #                  markerfacecolor=mice_color, markeredgecolor="none", marker="o", markersize=3, alpha=0.7)
                        axs[i].scatter(puff_auc_mean, blank_auc_mean, 
                                        facecolor=mice_color, edgecolor="white", marker="o", alpha=0.7, s=9, lw=0.2)
                        
                        all_puff_auc[i].append(puff_auc_mean)
                        all_blank_auc[i].append(blank_auc_mean)
                    else:
                        axs[i].scatter(puff_auc_mean, blank_auc_mean, 
                                        facecolor="gray", edgecolor="none", marker="o", alpha=0.3, s=5, zorder=-10)
    
    current_xlim, current_ylim = axs[0].get_xlim(), axs[0].get_ylim()
    # coordinate_range = (min(current_xlim[0], current_ylim[0]), max(current_xlim[1], current_ylim[1]))
    coordinate_range = (-0.15, 0.2)

    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    total_selectivity = np.concatenate([np.array(all_puff_auc[i]) - np.array(all_blank_auc[i]) for i in range(5)])
    selectivity_range = np.percentile(np.abs(total_selectivity), 95)
    selectivity_bins = np.linspace(-selectivity_range, selectivity_range, 11)
    for i in range(5):
        add_textonly_legend(axs[i], {mice_node.mice_id: {"color": mice_color} 
                                      for mice_node, mice_color in zip(dataset.select("mice"), mice_colors)})
        

        all_puff_auc[i], all_blank_auc[i] = np.array(all_puff_auc[i]), np.array(all_blank_auc[i])
        result = basic_ttest.stats_total_least_square_regress(all_puff_auc[i], all_blank_auc[i])

        axs[i].axline((0, result["intercept"]), slope=result["slope"], color="k", linestyle="--", lw=0.5, zorder=-8)
        axs[i].axline((0, 0), slope=1, color="gray", linestyle="--", lw=0.5, zorder=-9)
        axs[i].text(0.95, 0.05, f"slope={result['slope']:.2f}", transform=axs[i].transAxes, 
                    ha="right", va="bottom", color="gray", fontsize=5, alpha=0.5)
        
        local_selectivity = all_puff_auc[i] - all_blank_auc[i]
        axins = inset_axes(axs[i], width="20%", height="20%", loc="center right")
        axins.hist(local_selectivity, bins=selectivity_bins, color="k", alpha=0.8)
        axins.hist(total_selectivity, bins=selectivity_bins, color="gray", alpha=0.5, zorder=-10)
        axins.set_xlim(-selectivity_range, selectivity_range)
        axins.set_xticks([-selectivity_range, 0, selectivity_range], ["B",  "", "P"], fontsize=5, ha="center", va="center")
        axins.plot(1, 0, ">k", transform=axins.get_yaxis_transform(), clip_on=False, lw=0.5, markersize=1)
        axins.plot(0, 0, "<k", transform=axins.get_yaxis_transform(), clip_on=False, lw=0.5, markersize=1)
        axins.spines[['right', 'top', 'left', ]].set_visible(False)
        axins.spines['bottom'].set_linewidth(0.5)
        axins.patch.set_facecolor('none')
        axins.axvline(0, color="k", linestyle="--", lw=0.3,)
        axins.tick_params(axis='both', length=1, width=0.5)
        axins.set_yticks([])        
        
        range_str = f"[{early_time_range[0]:.1f}~{early_time_range[1]:.1f}s]"
        axs[i].set_xlabel(f"Puff AUC {range_str}")
        axs[0].set_ylabel(f"Blank AUC {range_str}")
        axs[i].set_title(f"Day {i+1}")
        axs[i].set_aspect("equal")
        axs[i].set_xlim(coordinate_range)
        axs[i].set_ylim(coordinate_range)
        
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=3))

    save_path = routing.default_fig_path(dataset, "EarlyActivitySummary_TrainingDays_" + f"_{{}}_{_aligment_style}_{early_time_range[0]:.1f}_{early_time_range[1]:.1f}.png", fov_skip=True)
    default_exit_save(fig, save_path)


def visualize_later_celluar_activity_summary(
        dataset: DataSet,

        later_time_range: tuple[float, float] = (0., 0.5),
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def calculate_trial_later_activity_auc(trial_nodes: DataSet):
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        max_fs = max([single_trial.data.fluorescence.df_f0.fs for single_trial in sync_trial_nodes])
        predefined_t = np.linspace(later_time_range[0], later_time_range[1], 
                                   int((later_time_range[1] - later_time_range[0]) * max_fs * 2))
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None,
                                            _predefined_t=predefined_t)
        later_mask = (group_fluo.t >= later_time_range[0]) & (group_fluo.t <= later_time_range[1])
        return np.trapezoid(group_fluo.raw_array[:, later_mask], group_fluo.t[later_mask])
    
    fig, axs = plt.subplots(1, 5, figsize=(10, 2), sharex=False, sharey=False, constrained_layout=True)

    mice_colors = ("Green", "Purple", "Orange")
    all_water_auc, all_nowater_auc = defaultdict(list), defaultdict(list)
    for mice_node, mice_color in zip(dataset.select("mice"), mice_colors):
        mice_subtree = dataset.subtree(mice_node)
        for cell_node in mice_subtree.select("cell"):
            cell_subtree = mice_subtree.subtree(cell_node)
            training_day_range = list(range(7, 12)) if mice_node.mice_id != "SUS6F" else (7, 8, 10, 11, 12)

            for cellday_node in cell_subtree.select("cellday"):
                if int(cellday_node.day_id) not in training_day_range:
                    continue
                cd_subtree = cell_subtree.subtree(cellday_node)
                training_day_index = training_day_range.index(int(cellday_node.day_id))

                water_trials = cd_subtree.select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffWater", "CueBlankWater"},
                )   
                nowater_trials = cd_subtree.select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffNoWater", "CueBlankNoWater"},
                )
                water_auc = calculate_trial_later_activity_auc(water_trials)
                nowater_auc = calculate_trial_later_activity_auc(nowater_trials)
                water_auc_mean = np.nanmean(water_auc)
                nowater_auc_mean = np.nanmean(nowater_auc)

                for i in range(5):
                    if i == training_day_index:                        
                        axs[i].scatter(water_auc_mean, nowater_auc_mean, 
                                        facecolor=mice_color, edgecolor="white", marker="o", alpha=0.7, s=9, lw=0.2)
                        
                        all_water_auc[i].append(water_auc_mean)
                        all_nowater_auc[i].append(nowater_auc_mean)
                    else:
                        axs[i].scatter(water_auc_mean, nowater_auc_mean, 
                                        facecolor="gray", edgecolor="none", marker="o", alpha=0.3, s=5, zorder=-10)
    
    current_xlim, current_ylim = axs[0].get_xlim(), axs[0].get_ylim()
    coordinate_range = (min(current_xlim[0], current_ylim[0]), max(current_xlim[1], current_ylim[1]))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    total_selectivity = np.concatenate([np.array(all_water_auc[i]) - np.array(all_nowater_auc[i]) for i in range(5)])
    selectivity_range = np.percentile(np.abs(total_selectivity), 95)
    selectivity_bins = np.linspace(-selectivity_range, selectivity_range, 11)
    for i in range(5):
        add_textonly_legend(axs[i], {mice_node.mice_id: {"color": mice_color} 
                                      for mice_node, mice_color in zip(dataset.select("mice"), mice_colors)})
        
        all_water_auc[i], all_nowater_auc[i] = np.array(all_water_auc[i]), np.array(all_nowater_auc[i])
        result = basic_ttest.stats_total_least_square_regress(all_water_auc[i], all_nowater_auc[i])
    
        axs[i].axline((0, 0), slope=1, color="gray", linestyle="--", lw=0.5, zorder=-9)
        axs[i].axline((0, result["intercept"]), slope=result["slope"], color="k", linestyle="--", lw=0.5, zorder=-8)
        axs[i].text(0.95, 0.05, f"slope={result['slope']:.2f}", transform=axs[i].transAxes, 
                    ha="right", va="bottom", color="k", fontsize=5, alpha=0.5)
        
        local_selectivity = all_water_auc[i] - all_nowater_auc[i]
        axins = inset_axes(axs[i], width="20%", height="20%", loc="center right")
        axins.hist(local_selectivity, bins=selectivity_bins, color="k", alpha=0.8)
        axins.hist(total_selectivity, bins=selectivity_bins, color="gray", alpha=0.5, zorder=-10)
        axins.set_xlim(-selectivity_range, selectivity_range)
        axins.set_xticks([-selectivity_range, 0, selectivity_range], ["NoW", "", "W"], fontsize=5, ha="center", va="center")
        axins.plot(1, 0, ">k", transform=axins.get_yaxis_transform(), clip_on=False, lw=0.5, markersize=1)
        axins.plot(0, 0, "<k", transform=axins.get_yaxis_transform(), clip_on=False, lw=0.5, markersize=1)
        axins.spines[['right', 'top', 'left', ]].set_visible(False)
        axins.spines['bottom'].set_linewidth(0.5)
        axins.patch.set_facecolor('none')
        axins.axvline(0, color="k", linestyle="--", lw=0.3,)
        axins.tick_params(axis='both', length=1, width=0.5)
        axins.set_yticks([])

        range_str = f"[{later_time_range[0]:.1f}~{later_time_range[1]:.1f}s]"
        axs[i].set_xlabel(f"Water AUC {range_str}")
        axs[i].set_ylabel(f"NoWater AUC {range_str}")
        axs[i].set_title(f"Day {i+1}")
        axs[i].set_aspect("equal")
        axs[i].set_xlim(coordinate_range)
        axs[i].set_ylim(coordinate_range)


        from matplotlib.ticker import MaxNLocator
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=3))

    save_path = routing.default_fig_path(dataset, "LaterActivitySummary_TrainingDays_" + f"_{{}}_{_aligment_style}_{later_time_range[0]:.1f}_{later_time_range[1]:.1f}.png", fov_skip=True)
    default_exit_save(fig, save_path)
        




    
def visualize_cellular_evoked_overall_summary(
        dataset: DataSet,

        auc_range: tuple[float, float],
        color: str,
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9


    

    def calculate_trial_activity_auc(trial_nodes: DataSet, selectivity_time_range: tuple[float, float]):
        if len(trial_nodes) == 0:
            return np.nan
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        max_fs = max([single_trial.data.fluorescence.df_f0.fs for single_trial in sync_trial_nodes])
        predefined_t = np.linspace(selectivity_time_range[0], selectivity_time_range[1], 
                                    int((selectivity_time_range[1] - selectivity_time_range[0]) * max_fs * 2))
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None,
                                            _predefined_t=predefined_t)
        inner_mask = (group_fluo.t >= selectivity_time_range[0]) & (group_fluo.t <= selectivity_time_range[1])
        return np.trapezoid(group_fluo.raw_array[:, inner_mask], group_fluo.t[inner_mask])

    
   
    
    # mice_colors = ("Green", "Purple", "Orange")
    # mice_id2color = {mice_node.mice_id: mice_color for mice_node, mice_color in zip(dataset.select("mice"), mice_colors)}


    # puff_auc_mean, blank_auc_mean = defaultdict(dict), defaultdict(dict)
    # for mice_node in dataset.select("mice"):
    #     mice_subtree = dataset.subtree(mice_node)
    #     for cell_node in mice_subtree.select("cell"):
    #         cell_subtree = mice_subtree.subtree(cell_node)
    #         training_day_range = list(range(1, 12)) if mice_node.mice_id != "SUS6F" else (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12)

    #         for cellday_node in cell_subtree.select("cellday"):
    #             if int(cellday_node.day_id) not in training_day_range:
    #                 continue
    #             cd_subtree = cell_subtree.subtree(cellday_node)
    #             training_day_index = training_day_range.index(int(cellday_node.day_id))

    #             # water_trials = cd_subtree.select(
    #             #     _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffWater", "CueBlankWater"},
    #             # )   
    #             # nowater_trials = cd_subtree.select(
    #             #     _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffNoWater", "CueBlankNoWater"},
    #             # )
    #             puff_trials = cd_subtree.select(
    #                 _element_trial_level, _self=lambda x: x.info.get("trial_type") in ("PuffOnly", ) #{"CuePuffWater", "CuePuffNoWater"},
    #             )
    #             blank_trials = cd_subtree.select(
    #                 _element_trial_level, _self=lambda x: x.info.get("trial_type") in ("BlankOnly", ) #{"CueBlankWater", "CueBlankNoWater"},
    #             )
                
    #             puff_auc = calculate_trial_activity_auc(puff_trials, early_selectivity_range)
    #             blank_auc = calculate_trial_activity_auc(blank_trials, early_selectivity_range)
    #             puff_auc_mean[cell_node][training_day_index] = np.nanmean(puff_auc)
    #             blank_auc_mean[cell_node][training_day_index] = np.nanmean(blank_auc)
    
    fig, axs = plt.subplots(2, 1, figsize=(4, 3), constrained_layout=True)
    markers = ["x", "s", "D", "^"]
    all_puff_auc, all_blank_auc = defaultdict(list), defaultdict(list)
    for mice_index, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        
        x_values = []
        puff_aucs, blank_aucs = [], []

        training_day_range = list(range(1, 12)) if mice_node.mice_id != "SUS6F" else (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12)
        n_day = len(training_day_range)

        for session_idx, session_node in enumerate(mice_subtree.select("session")):
            if int(session_node.day_id) not in training_day_range: #or "P1" in session_node.session_id or "P2" in session_node.session_id:
                continue
            session_subtree = mice_subtree.subtree(session_node)
            training_day_index = training_day_range.index(int(session_node.day_id))
            session_index = get_session_index(session_node)

            session_x_value = training_day_index + session_index * 0.1

            puff_trials = session_subtree.select(
                _element_trial_level, _self=lambda x: x.info.get("trial_type") in ("PuffOnly", "CuePuffWater", "CuePuffNoWater"),
            )

            blank_trials = session_subtree.select(
                _element_trial_level, _self=lambda x: x.info.get("trial_type") in ("BlankOnly", "CueBlankWater", "CueBlankNoWater"),
            )

            puff_auc_mean = np.nanmean(calculate_trial_activity_auc(puff_trials, auc_range))
            blank_auc_mean = np.nanmean(calculate_trial_activity_auc(blank_trials, auc_range))
            if np.isnan(puff_auc_mean) or np.isnan(blank_auc_mean):
                continue
            x_values.append(session_x_value)
            puff_aucs.append(puff_auc_mean)
            blank_aucs.append(blank_auc_mean)

        for x_v, puff_v, blank_v in zip(x_values, puff_aucs, blank_aucs):
            all_puff_auc[x_v].append(puff_v)
            all_blank_auc[x_v].append(blank_v)
        x_values += (0.9 + np.arange(n_day - 1)).tolist()
        puff_aucs += [np.nan] * (n_day - 1)
        blank_aucs += [np.nan] * (n_day - 1)
        
        sorted_indices = np.argsort(x_values)
        x_values = np.array(x_values)[sorted_indices]
        puff_aucs = np.array(puff_aucs)[sorted_indices]
        blank_aucs = np.array(blank_aucs)[sorted_indices]
        axs[0].plot(x_values, puff_aucs, color=color, markersize=1, alpha=0.3, lw=0.5, marker=markers[mice_index],)
        axs[1].plot(x_values, blank_aucs, color=color, markersize=1, alpha=0.3, lw=0.5, marker=markers[mice_index], )
    
    all_x_values = sorted(set(all_puff_auc.keys()) | set(all_blank_auc.keys()))
    all_puff_aucs = [np.nanmean(all_puff_auc[x_v]) for x_v in all_x_values]
    all_blank_aucs = [np.nanmean(all_blank_auc[x_v]) for x_v in all_x_values]
    all_x_values += (0.9 + np.arange(10)).tolist()
    all_puff_aucs += [np.nan] * 10
    all_blank_aucs += [np.nan] * 10
    sorted_indices = np.argsort(all_x_values)
    all_x_values = np.array(all_x_values)[sorted_indices]
    all_puff_aucs = np.array(all_puff_aucs)[sorted_indices]
    all_blank_aucs = np.array(all_blank_aucs)[sorted_indices]
    axs[0].plot(all_x_values, all_puff_aucs, color=color, marker="o", markersize=2, markerfacecolor=color, markeredgecolor="k", lw=1.5, alpha=0.8)
    axs[1].plot(all_x_values, all_blank_aucs, color=color, marker="o", markersize=2, markerfacecolor=color, markeredgecolor="k", lw=1.5, alpha=0.8)
    for ax in axs:
        ax.set_xticks(0.3 + np.arange(11), [f"Day{i+1}" for i in range(11)], fontsize=5)
        # ax.set_xlim(-0.5, 4.5)
        ax.axhline(0, color="gray", linestyle="--", lw=0.5, zorder=-10)
        # ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel(DF_F0_SIGN + " AUC")
    axs[0].set_title("Puff trials")
    axs[1].set_title("Blank trials")
    
    axs[0].set_ylim(-0.02, 0.05)
    axs[1].set_ylim(-0.02, 0.05)

    save_path = routing.default_fig_path(dataset, "EvokedSummary_ALLSessions003_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig, save_path, _transparent=True)


    # fig, axs = plt.subplots(1, 2, figsize=(4, 1.5), sharey=True, constrained_layout=True)

    # for cell_node in dataset.select("cell"):
    #     axs[0].plot(np.arange(5), [puff_auc_mean[cell_node].get(i, np.nan) for i in range(5)],
    #                 color="gray", alpha=0.3, marker='o', markersize=1., lw=0.3)
    #     axs[1].plot(np.arange(5), [blank_auc_mean[cell_node].get(i, np.nan) for i in range(5)],
    #                 color="gray", alpha=0.3, marker='o', markersize=1., lw=0.3)
    # puff_bar_heights = [np.nanmean([puff_auc_mean[cell_node].get(i, np.nan) for cell_node in dataset.select("cell")]) for i in range(5)]
    # blank_bar_heights = [np.nanmean([blank_auc_mean[cell_node].get(i, np.nan) for cell_node in dataset.select("cell")]) for i in range(5)]
    # axs[0].plot(np.arange(5), puff_bar_heights, color=color, marker='o', markersize=3, markerfacecolor="k", lw=1.5)
    # axs[1].plot(np.arange(5), blank_bar_heights, color=color, marker='o', markersize=3, markerfacecolor="k", lw=1.5)
    # for ax in axs:
    #     ax.set_xticks(np.arange(5), [f"Day{i+1}" for i in range(5)], fontsize=5)
    #     ax.set_yticks([0, 0.5, 1.0])
    #     ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    #     # add_textonly_legend(ax, {mice_node.mice_id: {"color": mice_color} for mice_node, mice_color in zip(dataset.select("mice"), mice_colors)})
    
    # axs[0].set_ylim(-0.5, 1.5)
    # axs[1].set_ylim(-0.5, 1.5)
    # axs[0].set_ylabel(f"evoked AUC ({DF_F0_SIGN})")
    # axs[0].set_title("Puff trials")
    # axs[1].set_title("Blank trials")
    # # axs[1].set_ylabel(f"evoked AUC ({DF_F0_SIGN})")
    # save_path = routing.default_fig_path(dataset, "EvokedSummary_TrainingDays_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    # default_exit_save(fig, save_path)
    
    # # mice_colors = ("Green", "Purple", "Orange")
    # # mice_id2color = {mice_node.mice_id: mice_color for mice_node, mice_color in zip(dataset.select("mice"), mice_colors)}
    # fig, ax = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
    # all_puff_delta_auc, all_blank_delta_auc = [], []
    # for cell_node in dataset.select("cell"):
    #     puff_delta_auc = puff_auc_mean[cell_node].get(4, np.nan) - puff_auc_mean[cell_node].get(0, np.nan) 
    #     blank_delta_auc = blank_auc_mean[cell_node].get(4, np.nan) - blank_auc_mean[cell_node].get(0, np.nan)
    #     # mice_color = mice_id2color[cell_node.mice_id]
    #     ax.scatter(puff_delta_auc, blank_delta_auc, facecolor=color, edgecolor="white", marker="o", alpha=0.7, s=16, lw=0.2)
    #     all_puff_delta_auc.append(puff_delta_auc)
    #     all_blank_delta_auc.append(blank_delta_auc)

    # all_puff_delta_auc, all_blank_delta_auc = np.array(all_puff_delta_auc), np.array(all_blank_delta_auc)
    # result = basic_ttest.stats_total_least_square_regress(all_puff_delta_auc, all_blank_delta_auc)
    
    # ax.axline((0, result["intercept"]), slope=result["slope"], color=color, linestyle="--", lw=0.5, zorder=-8)
    
    # ax.set_xlabel("Puff AUC [1~2s] Day5 - Day1")
    # ax.set_ylabel("Blank AUC [1~2s] Day5 - Day1")
    # ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    # ax.axvline(0, color="gray", linestyle="--", lw=0.5)
    # # ax.set_aspect("equal")
    # # add_textonly_legend(ax, {mice_node.mice_id: {"color": mice_color} for mice_node, mice_color in zip(dataset.select("mice"), mice_colors)})

    # ax.set_xlim(-1.3, 1.3)
    # ax.set_ylim(-0.8, 0.8)
    # ax.set_xticks([-1, 0, 1])
    # ax.set_yticks([-0.5, 0, 0.5])
    # save_path = routing.default_fig_path(dataset, "EvokedDeltaSummary_TrainingDays_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    # default_exit_save(fig, save_path, _transparent=True)


    
def visualize_daywise_correlation_summary(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):
    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    
    def get_day_index(cs_node: Node) -> Optional[int]:
        day_id_int = int(cs_node.day_id)
        if cs_node.mice_id == "SUS6F":
            if day_id_int == 9:
                return None
            elif day_id_int > 9:    
                return day_id_int - 1
        return day_id_int


    early_selectivity_range = (0.0, 2.0)

    def calculate_trial_activity_auc(trial_nodes: DataSet, selectivity_time_range: tuple[float, float]):
        if len(trial_nodes) == 0:
            return np.nan
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        max_fs = max([single_trial.data.fluorescence.df_f0.fs for single_trial in sync_trial_nodes])
        predefined_t = np.linspace(selectivity_time_range[0], selectivity_time_range[1], 
                                    int((selectivity_time_range[1] - selectivity_time_range[0]) * max_fs * 2))
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None,
                                            _predefined_t=predefined_t)
        inner_mask = (group_fluo.t >= selectivity_time_range[0]) & (group_fluo.t <= selectivity_time_range[1])
        return np.trapezoid(group_fluo.raw_array[:, inner_mask], group_fluo.t[inner_mask])


    plotting_nodes = []
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 2.5), sharex=False, sharey=False, constrained_layout=True)
    fig_corr, axs_corr = plt.subplots(1, 4, figsize=(12, 3), sharex=False, sharey=False, constrained_layout=True)

    import numpy.ma as ma
    
    n_day = 11
    x_tick_grid = np.arange(-0.5, 7*n_day + 0.5, 7)
    spllit_lines = list(sorted(-0. + x_tick_grid) + list(-1. + x_tick_grid) + list(1 + x_tick_grid))

    all_value_matrix = []
    for mice_idx, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        
        n_cell = len(mice_subtree.select("cell"))
        cell_names = [cell_node.cell_id for cell_node in mice_subtree.select("cell")]
        n_session = len(mice_subtree.select("session"))
        print(f"Processing mice {mice_node.mice_id} with {n_cell} cells, {n_day} days (7x{n_day}={7*n_day}), {n_session} sessions.")

        value_matrix = np.full((n_cell, 7*n_day), np.nan)

        for day_node in mice_subtree.select("day"):
            print(f"Processing {mice_node.mice_id} Day {day_node.day_id}")
            day_subtree = mice_subtree.subtree(day_node)

            day_index = get_day_index(day_node)
            if day_index is None or day_index > n_day:
                # print(f"Skipping {day_node} due to missing day index.")
                continue
            
            for cell_session_node in day_subtree.select("cellsession"):
                cell_index = cell_names.index(cell_session_node.cell_id)
                session_index = get_session_index(cell_session_node)
                col_index = session_index + (day_index - 1) * 7

                
                value_matrix[cell_index, col_index] = np.nanmean(calculate_trial_activity_auc(day_subtree.subtree(cell_session_node).select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in ("PuffOnly", "CuePuffWater", "CuePuffNoWater"),
                ), selectivity_time_range=early_selectivity_range))
        tmp_ax = axs[mice_idx]   
        sorting_order = np.argsort(np.nanmean(value_matrix[:, 6*7:7*7], axis=1))
        tmp_ax.imshow(value_matrix[sorting_order], vmin=-1, vmax=1, cmap="RdYlBu_r", aspect="auto")
        tmp_ax.set_ylabel(mice_node.mice_id)
        tmp_ax.set_yticks([0, n_cell-1], ["1", str(n_cell)], )
        tmp_ax.set_xticks(spllit_lines, minor=True)
        tmp_ax.grid(which='minor', axis='x', color='gray', linestyle='--', linewidth=0.5)
        tmp_ax.tick_params(which='minor', bottom=False, left=False)
        tmp_ax.set_xticks(np.arange(3, 7*n_day, 7), [f"day {i+1}" for i in range(n_day)])

        tmp_ax_corr = axs_corr[mice_idx]
        corr_matrix = ma.corrcoef(ma.masked_invalid(value_matrix), rowvar=False)
        tmp_ax_corr.imshow(corr_matrix, vmin=0, vmax=1, cmap="plasma")
        tmp_ax_corr.set_title(f"{mice_node.mice_id}", fontsize=6)
        tmp_ax_corr.set_xticks(spllit_lines, minor=True)        
        tmp_ax_corr.set_yticks(spllit_lines, minor=True)
        
        tmp_ax_corr.grid(which='minor', axis='both', color='gray', linestyle='--', linewidth=0.5)
        tmp_ax_corr.tick_params(which='minor', bottom=False, left=False)
        tmp_ax_corr.set_xticks(np.arange(3, 7*n_day, 7), [f"day{i+1}" for i in range(n_day)])
        tmp_ax_corr.set_yticks(np.arange(3, 7*n_day, 7), [f"day{i+1}" for i in range(n_day)])
        all_value_matrix.append(value_matrix)
    
    all_value_matrix = np.concatenate(all_value_matrix, axis=0)
    all_corrs = ma.corrcoef(ma.masked_invalid(all_value_matrix), rowvar=False)
    print(type(all_corrs), all_corrs.shape)
    axs_corr[-1].imshow(all_corrs, vmin=0, vmax=1, cmap="plasma")
    axs_corr[-1].set_title(f"All mice", fontsize=6)
    axs_corr[-1].set_xticks(spllit_lines, minor=True)      
    axs_corr[-1].set_yticks(spllit_lines, minor=True)
    axs_corr[-1].grid(which='minor', axis='both', color='gray', linestyle='--', linewidth=0.5)
    axs_corr[-1].set_xticks(np.arange(3, 7*n_day, 7), [f"day{i+1}" for i in range(n_day)])
    axs_corr[-1].set_yticks(np.arange(3, 7*n_day, 7), [f"day{i+1}" for i in range(n_day)])
        
        

    save_path = routing.default_fig_path(dataset, "Correlation_Matrix_TrainingDays_PuffevokedRaw_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig, save_path)


    save_path = routing.default_fig_path(dataset, "Correlation_Matrix_TrainingDays_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig_corr, save_path)
        

def visualize_daywise_performance(
        dataset: DataSet,

        color: str,
        _element_trial_level: str = "fovtrial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9



    def calculate_session_lickrate(trial_nodes: DataSet):
        if len(trial_nodes) == 0:
            return np.nan
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        lick_timepoints = [single_trial.data.lick.t for single_trial in sync_trial_nodes]
        threshold = (-1, 0) if trial_nodes.nodes[0].info.get("trial_type") in {"CueWater", "CueNoWater"} else (1, 2)
        return np.mean([np.sum((lick_timepoint >= threshold[0]) & (lick_timepoint < threshold[1])) for lick_timepoint in lick_timepoints])
    
    
    fig, axs = plt.subplots(2, 1, figsize=(3, 3), sharey=True, constrained_layout=True)

    markers = ["o", "x", "s", "D", "^"]
    for mice_index, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        
        x_values = []
        puff_blank_performance = []
        water_nowater_performance = []

        training_day_range = list(range(1, 12)) if mice_node.mice_id != "SUS6F" else (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12)
        n_day = len(training_day_range)
        day_session_count = {day_id: -1 for day_id in training_day_range}

        for session_idx, session_node in enumerate(mice_subtree.select("session")):
            if int(session_node.day_id) not in training_day_range or "P1" in session_node.session_id or "P2" in session_node.session_id:
                continue
            session_subtree = mice_subtree.subtree(session_node)
            training_day_index = training_day_range.index(int(session_node.day_id))
            day_session_count[int(session_node.day_id)] += 1

            session_x_value = training_day_index + day_session_count[int(session_node.day_id)] * 0.15
            x_values.append(session_x_value)

            if training_day_index <= 5:
                puff_blank_performance.append(np.nan)
            else:
                puff_lr = calculate_session_lickrate(session_subtree.select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffWater", "CuePuffNoWater"},
                ))
                blank_lr = calculate_session_lickrate(session_subtree.select(
                    _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CueBlankWater", "CueBlankNoWater"},
                ))
                puff_blank_performance.append(puff_lr - blank_lr)

            print([x.info.get("trial_type") for x in session_subtree.select(_element_trial_level)])
            water_lr = calculate_session_lickrate(session_subtree.select(
                _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffWater", "CueBlankWater", "CueWater"},
            ))
            nowater_lr = calculate_session_lickrate(session_subtree.select(
                _element_trial_level, _self=lambda x: x.info.get("trial_type") in {"CuePuffNoWater", "CueBlankNoWater", "CueNoWater"},
            ))
            water_nowater_performance.append(water_lr - nowater_lr)

        x_values += (0.9 + np.arange(n_day - 1)).tolist()
        water_nowater_performance += [np.nan] * (n_day - 1)
        puff_blank_performance += [np.nan] * (n_day - 1)
        sorted_indices = np.argsort(x_values)
        x_values = np.array(x_values)[sorted_indices]
        water_nowater_performance = np.array(water_nowater_performance)[sorted_indices]
        puff_blank_performance = np.array(puff_blank_performance)[sorted_indices]
        axs[0].plot(x_values, water_nowater_performance, color=color, markersize=1, alpha=0.7, lw=0.5, marker=markers[mice_index],)
        axs[1].plot(x_values, puff_blank_performance, color=color, markersize=1, alpha=0.7, lw=0.5, marker=markers[mice_index], )
    for ax in axs:
        ax.set_xticks(0.3 + np.arange(11), [f"Day{i+1}" for i in range(11)], fontsize=5)
        # ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-3, 5)
        ax.axhline(0, color="gray", linestyle="--", lw=0.5)
        # ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel("Lick rate difference (Hz)")
    axs[0].set_title("Water - NoWater")
    axs[1].set_title("Puff - Blank")

    save_path = routing.default_fig_path(dataset, "Behavior_Performance_TrainingDays_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig, save_path, _transparent=True)


def visualize_sessionwise_weight_distribution(
        dataset: DataSet,

        colormap: str,
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):
    
    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    def get_session_index(cs_node: Node) -> int:
        session_index_str = cs_node.session_id.split("_")[3]
        if session_index_str == "P1":
            return 0
        elif session_index_str == "P2":
            return 6
        else:
            assert session_index_str.startswith("S"), f"Unexpected session id format: {cs_node.session_id}"
            return int(session_index_str[1:])
    
    def get_day_index(cs_node: Node) -> Optional[int]:
        day_id_int = int(cs_node.day_id)
        if cs_node.mice_id == "SUS6F":
            if day_id_int == 9:
                return None
            elif day_id_int > 9:    
                return day_id_int - 1
        return day_id_int

    def calculate_corr_within_range(cs_node: Node, time_range: tuple[float, float]):
        task_start, _ = cs_node.data.timeline.task_time()
        locomotion_trace = cell_session_node.data.locomotion.segment(task_start + time_range[0], task_start + time_range[1])
        # timeline_trace = cell_session_node.data.timeline
        fluorescence_trace = cell_session_node.data.fluorescence.segment(task_start + time_range[0], task_start + time_range[1])
        pad_zero = TimeSeries(v=np.zeros_like(fluorescence_trace.raw_f.t), t=fluorescence_trace.raw_f.t)
        loco_fluo_grouped = grouping_timeseries([locomotion_trace.rate(LOCOMOTION_BIN_SIZE) + pad_zero, 
                                                 fluorescence_trace.detrend_z_score.squeeze(0)]).raw_array
        return np.corrcoef(loco_fluo_grouped)[0, 1]


    plotting_nodes = []
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 2.5), sharex=False, sharey=False, constrained_layout=True)
    fig_pre, axs_pre = plt.subplots(3, 1, figsize=(5, 2.5), sharex=False, sharey=False, constrained_layout=True)
    fig_post, axs_post = plt.subplots(3, 1, figsize=(5, 2.5), sharex=False, sharey=False, constrained_layout=True)

    for mice_idx, mice_node in enumerate(dataset.select("mice")):
        mice_subtree = dataset.subtree(mice_node)
        
        n_cell = len(mice_subtree.select("cell"))
        cell_names = [cell_node.cell_id for cell_node in mice_subtree.select("cell")]
        n_day = len(mice_subtree.select("day"))
        n_day = n_day - 1 if mice_node.mice_id == "SUS6F" else n_day
        n_session = len(mice_subtree.select("session"))
        print(f"Processing mice {mice_node.mice_id} with {n_cell} cells, {n_day} days (7x{n_day}={7*n_day}), {n_session} sessions.")

        value_matrix = np.full((n_cell, 7*n_day), np.nan)
        value_matrix_pre = np.full((n_cell, 7*n_day), np.nan)
        value_matrix_post = np.full((n_cell, 7*n_day), np.nan)

        for day_node in mice_subtree.select("day"):
            print(f"Processing {mice_node.mice_id} Day {day_node.day_id}")
            day_subtree = mice_subtree.subtree(day_node)

            day_index = get_day_index(day_node)
            if day_index is None:
                print(f"Skipping {day_node} due to missing day index.")
                continue

            for cell_session_node in day_subtree.select("cellsession"):
                cell_index = cell_names.index(cell_session_node.cell_id)
                session_index = get_session_index(cell_session_node)
                col_index = session_index + (day_index - 1) * 7

                
                value_matrix[cell_index, col_index] = calculate_corr_within_range(cell_session_node, (0, 300))
                value_matrix_pre[cell_index, col_index] = calculate_corr_within_range(cell_session_node, (0, 50))
                value_matrix_post[cell_index, col_index] = calculate_corr_within_range(cell_session_node, (50, 300))
        for tmp_value_matrix, tmp_ax in zip((value_matrix, value_matrix_pre, value_matrix_post), (axs[mice_idx], axs_pre[mice_idx], axs_post[mice_idx])):
            sorting_order = np.argsort(np.nanmean(tmp_value_matrix[:, 6*7:7*7], axis=1))
            tmp_ax.imshow(tmp_value_matrix[sorting_order], vmin=-1, vmax=1, cmap=colormap, aspect="auto")
            tmp_ax.set_ylabel(mice_node.mice_id)
            tmp_ax.set_yticks([0, n_cell-1], ["1", str(n_cell)], )
            x_tick_grid = np.arange(-0.5, 7*n_day + 0.5, 7)
            tmp_ax.set_xticks(list(sorted(-0.5 + x_tick_grid) + list(-1.5 + x_tick_grid) + list(0.5 + x_tick_grid)), minor=True)
            tmp_ax.grid(which='minor', axis='x', color='gray', linestyle='--', linewidth=0.5)
            tmp_ax.tick_params(which='minor', bottom=False, left=False)
            tmp_ax.set_xticks(np.arange(3, 7*n_day, 7), [f"day {i+1}" for i in range(n_day)])
        
        

    save_path = routing.default_fig_path(dataset, "WeightCorrelationWithLOCOMOTION_sessions_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig, save_path)
    save_path = routing.default_fig_path(dataset, "WeightCorrelationWithLOCOMOTION_sessions_PRE_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig_pre, save_path)
    save_path = routing.default_fig_path(dataset, "WeightCorrelationWithLOCOMOTION_sessions_POST_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig_post, save_path)