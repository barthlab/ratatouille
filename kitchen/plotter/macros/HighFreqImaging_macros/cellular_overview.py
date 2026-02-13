
import numpy as np
import matplotlib.pyplot as plt

from kitchen.calculator.sorting_data import get_amplitude_sorted_idxs
from kitchen.configs import routing
from kitchen.operator.grouping import grouping_events_rate, grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.ax_plotter.basic_plot import heatmap_view, stack_view
from kitchen.plotter.decorators.default_decorators import coroutine_cycle, default_exit_save
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import LOCOMOTION_BIN_SIZE
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import select_truthy_items


def visualize_celluar_activity_with_deconv(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)
    plot_manual_deconv = PlotManual(deconv_fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9
    for mice_node in dataset.select("mice"):
        for cell_session_node in dataset.subtree(mice_node).select("cellsession"):
            cs_subtree = dataset.subtree(cell_session_node)
            type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                       plot_manual=plot_manual_fluo,
                                                       _element_trial_level =_element_trial_level,)
            if len(type2dataset) == 0:
                continue
            
            fig, axs = plt.subplots(4, 4, figsize=(6, 6), sharex=False, sharey=False, 
                                    constrained_layout=True, height_ratios=[1, 0.6, 1, 0.6], width_ratios=[1, 1, 0.6, 0.6])            
            fig.suptitle(f"Cell {cell_session_node.fluorescence.cell_order[0]} in {cell_session_node.session_id}")
            
            coroutines = {}
            axs[0, 0].set_ylabel("Fluorescence", fontsize=9)
            axs[2, 0].set_ylabel("Deconvolved", fontsize=9)
            for idx, (trial_type, type_dataset) in enumerate(type2dataset.items()):
                
                axs[0, idx].set_title(f"{trial_type}")
                axs[0, idx+2].set_title(f"{trial_type}\nzoom-in")
                
                for ax in axs[:, idx]:                    
                    ax.tick_params(axis='y', labelleft=True)
                    ax.set_yticks([])
                    ax.set_xlim(-5, 5)
                for ax in axs[:, idx+2]:
                    ax.tick_params(axis='y', labelleft=True)
                    ax.set_yticks([])
                    ax.set_xlim(-0.5, 1.)
                heatmap_plotter = heatmap_view(ax=axs[0, idx], datasets=type_dataset, sync_events=alignment_events, 
                                              plot_manual=plot_manual_fluo, modality_name="fluorescence")
                coroutines[heatmap_plotter] = f"{trial_type}_heatmap_view"
                heatmap_plotter = heatmap_view(ax=axs[0, idx+2], datasets=type_dataset, sync_events=alignment_events, 
                                              plot_manual=plot_manual_fluo, modality_name="fluorescence")
                coroutines[heatmap_plotter] = f"{trial_type}_heatmap_view_zoomin"

                trace_plotter = stack_view(ax=axs[1, idx], datasets=type_dataset, sync_events=alignment_events,
                                           plot_manual=plot_manual_fluo)
                coroutines[trace_plotter] = f"{trial_type}_stack_view"
                trace_plotter = stack_view(ax=axs[1, idx+2], datasets=type_dataset, sync_events=alignment_events,
                                           plot_manual=plot_manual_fluo)
                coroutines[trace_plotter] = f"{trial_type}_stack_view_zoomin"

                heatmap_plotter = heatmap_view(ax=axs[2, idx], datasets=type_dataset, sync_events=alignment_events, 
                                              plot_manual=plot_manual_fluo, modality_name="deconv_fluorescence")
                coroutines[heatmap_plotter] = f"{trial_type}_deconv_heatmap_view"
                heatmap_plotter = heatmap_view(ax=axs[2, idx+2], datasets=type_dataset, sync_events=alignment_events, 
                                              plot_manual=plot_manual_fluo, modality_name="deconv_fluorescence")
                coroutines[heatmap_plotter] = f"{trial_type}_deconv_heatmap_view_zoomin"

                trace_plotter = stack_view(ax=axs[3, idx], datasets=type_dataset, sync_events=alignment_events,
                                           plot_manual=plot_manual_deconv)
                coroutines[trace_plotter] = f"{trial_type}_deconv_stack_view"
                trace_plotter = stack_view(ax=axs[3, idx+2], datasets=type_dataset, sync_events=alignment_events,
                                           plot_manual=plot_manual_deconv)
                coroutines[trace_plotter] = f"{trial_type}_deconv_stack_view_zoomin"
            coroutine_cycle(coroutines)
            for ax in axs[0, :]:
                ax.set_xticks([])
            for ax in axs[1, :]:
                ax.set_ylim(0, 3.5)   
            for ax in axs[2, :]:
                ax.set_xticks([])         
            for ax in axs[3, :]:
                ax.set_ylim(0, 3.5)

            for ax in axs[1, 1:]:
                ax.tick_params(axis='y', labelleft=False)
            for ax in axs[3, 1:]:
                ax.tick_params(axis='y', labelleft=False)   

            save_path = routing.default_fig_path(cs_subtree, "CelluarOverview_with_deconv" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
            default_exit_save(fig, save_path)


def visualize_celluar_activity_with_behavior(
        dataset: DataSet,

        sort_range: tuple[float, float] = (0, 1),
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    plot_manual_whisk = PlotManual(whisker=True, baseline_subtraction=None)
    plot_manual_loco = PlotManual(locomotion=True, baseline_subtraction=None)
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9
    for mice_node in dataset.select("mice"):
        for cell_session_node in dataset.subtree(mice_node).select("cellsession"):
            cs_subtree = dataset.subtree(cell_session_node)
            type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                       plot_manual=plot_manual_fluo,
                                                       _element_trial_level =_element_trial_level,)
            if len(type2dataset) == 0:
                continue
            
            fig, axs = plt.subplots(5, 7, figsize=(12.5, 7.5), sharex=False, sharey=False, 
                                    constrained_layout=True, 
                                    width_ratios=[1, 1, 1, 0.2, 1, 1, 1],
                                    height_ratios=[1, 1, 1, 1, 0.6])
            fig.suptitle(f"Cell {cell_session_node.fluorescence.cell_order[0]} in {cell_session_node.session_id}")
            
            coroutines = {}

            axs[0, 0].set_ylabel("Trial Order", fontsize=9)
            axs[1, 0].set_ylabel("Sort by Fluorescence", fontsize=9)
            axs[2, 0].set_ylabel("Sort by Whisking", fontsize=9)
            axs[3, 0].set_ylabel("Sort by Locomotion", fontsize=9)
            
            for ax in axs.flatten():
                ax.tick_params(axis='y', labelleft=True)
                ax.set_yticks([])
                ax.set_xlim(-5, 5)

            for row_id in range(5):
                axs[row_id, 3].remove()
                axs[row_id, 0].set_ylim(0, 3.5)
                axs[row_id, 4].set_ylim(0, 3.5)
                axs[row_id, 1].set_ylim(0, 2)
                axs[row_id, 5].set_ylim(0, 2)
                axs[row_id, 2].set_ylim(0, 2)
                axs[row_id, 6].set_ylim(0, 2)

            for type_idx, (trial_type, raw_type_dataset) in enumerate(type2dataset.items()):
                type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                for i, modality_name in enumerate(["fluorescence", "whisker", "locomotion"]):
                    axs[0, type_idx*4+i].set_title(f"{trial_type}\n{modality_name}")

                # sort by fluorescence amplitude
                group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) 
                                                          for single_fluorescence in select_truthy_items(
                                                              [node.data.fluorescence for node in type_dataset])], 
                                             baseline_subtraction=None)
                fluorescence_order = get_amplitude_sorted_idxs(group_fluorescence, amplitude_range=sort_range)

                # sort by whisking amplitude
                group_whisker = grouping_timeseries(select_truthy_items([node.data.whisker for node in type_dataset]), 
                                                    baseline_subtraction=None)
                whisker_order = get_amplitude_sorted_idxs(group_whisker, amplitude_range=sort_range)

                # sort by locomotion amplitude
                group_locomotion = grouping_events_rate(select_truthy_items([node.data.locomotion for node in type_dataset]), 
                                                        bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=None)
                locomotion_order = get_amplitude_sorted_idxs(group_locomotion, amplitude_range=sort_range)
                
                for content_idx, (modality_name, specific_plot_manual) in enumerate(zip(
                    ["fluorescence", "whisker", "locomotion",], 
                    [plot_manual_fluo, plot_manual_whisk, plot_manual_loco])):                    

                    specific_plotter = heatmap_view(ax=axs[0, type_idx*4+content_idx], datasets=type_dataset, sync_events=alignment_events, 
                                                plot_manual=specific_plot_manual, modality_name=modality_name)
                    coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}"

                    specific_plotter = heatmap_view(ax=axs[1, type_idx*4+content_idx], datasets=type_dataset, sync_events=alignment_events, 
                                                plot_manual=specific_plot_manual, modality_name=modality_name, specified_order=fluorescence_order)
                    coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}_sortby_fluorescence"

                    specific_plotter = heatmap_view(ax=axs[2, type_idx*4+content_idx], datasets=type_dataset, sync_events=alignment_events, 
                                                plot_manual=specific_plot_manual, modality_name=modality_name, specified_order=whisker_order)
                    coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}_sortby_whisker"

                    specific_plotter = heatmap_view(ax=axs[3, type_idx*4+content_idx], datasets=type_dataset, sync_events=alignment_events, 
                                                plot_manual=specific_plot_manual, modality_name=modality_name, specified_order=locomotion_order)
                    coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}_sortby_locomotion"

                    trace_plotter = stack_view(ax=axs[4, type_idx*4+content_idx], datasets=type_dataset, sync_events=alignment_events,
                                               plot_manual=specific_plot_manual)
                    coroutines[trace_plotter] = f"{trial_type}_stack_view_{modality_name}"
            coroutine_cycle(coroutines)

            save_path = routing.default_fig_path(cs_subtree, "CelluarOverview_with_behavior" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
            default_exit_save(fig, save_path)







