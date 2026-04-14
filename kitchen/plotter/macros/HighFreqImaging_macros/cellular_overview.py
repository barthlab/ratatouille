
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from kitchen.calculator.basic_metric import PEARSON_CORRELATION
from kitchen.calculator.sorting_data import get_amplitude_diff_sorted_idxs, get_amplitude_sorted_idxs
from kitchen.configs import routing
from kitchen.operator.grouping import grouping_events_rate, grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import style_dicts
from kitchen.plotter.ax_plotter.basic_plot import heatmap_view, stack_view
from kitchen.plotter.decorators.default_decorators import coroutine_cycle, default_exit_save
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import LOCOMOTION_BIN_SIZE
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_timeline
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.structure.neural_data_structure import TimeSeries, TimeSeries_concat
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
            save_path = save_path.replace(f"deconv_{cell_session_node.cell_id}_", f"deconv_{cell_session_node.fluorescence.cell_order[0]:02d}_")
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
            save_path = save_path.replace(f"behavior_{cell_session_node.cell_id}_", f"behavior_{cell_session_node.fluorescence.cell_order[0]:02d}_")
            default_exit_save(fig, save_path)






def visualize_behavior_correlation_distribution(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",
):
    
    from distinctipy import distinctipy

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    all_corrs = {}
    for cell_session_node in dataset.select("cellsession"):
        if cell_session_node.fluorescence is None:
            continue
        print(f"Processing {cell_session_node.session_id}")
        fluo = cell_session_node.fluorescence.z_score
        whisker = cell_session_node.whisker
        locomotion = cell_session_node.locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE)
        trial_onset_t = cell_session_node.data.timeline.filter("TrialOn").t
        concat_fluo = TimeSeries_concat(fluo.batch_segment(ts=trial_onset_t, segment_range=(-5, 5), _auto_align=False))
        concat_whisker = TimeSeries_concat(whisker.batch_segment(ts=trial_onset_t, segment_range=(-5, 5), _auto_align=False))
        concat_locomotion = TimeSeries_concat(locomotion.batch_segment(ts=trial_onset_t, segment_range=(-5, 5), _auto_align=False))
        all_corrs[cell_session_node] = {
            "whisker": PEARSON_CORRELATION(concat_fluo, concat_whisker),
            "locomotion": PEARSON_CORRELATION(concat_fluo, concat_locomotion),
        }
    whisker_corrs = [corrs["whisker"] for cs_node, corrs in all_corrs.items()]
    locomotion_corrs = [corrs["locomotion"] for cs_node, corrs in all_corrs.items()]

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True, constrained_layout=True)

    fovs = list(set([cs_node.fov_id for cs_node in all_corrs.keys()]))
    mice = list(set([cs_node.mice_id for cs_node in all_corrs.keys()]))
    color_by_fov = distinctipy.get_colors(len(fovs), pastel_factor=0.5)
    color_by_mice = distinctipy.get_colors(len(mice), pastel_factor=0.5)

    for ax, color_template in zip(
        axs, 
        ["color by fov", "color by mice"]):
        ax.set_title(color_template)
        ax.set_xlabel("Correlation with whisking")
        ax.set_ylabel("Correlation with locomotion")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        ax.axvline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        ax.spines[['top', 'right']].set_visible(False)

   
    axs[0].scatter(whisker_corrs, locomotion_corrs, 
                   facecolors=[color_by_fov[fovs.index(cs_node.fov_id)] 
                      for cs_node in all_corrs.keys()], s=6, alpha=0.8, edgecolors='none',)
    axs[1].scatter(whisker_corrs, locomotion_corrs, 
                   facecolors=[color_by_mice[mice.index(cs_node.mice_id)] 
                      for cs_node in all_corrs.keys()], s=6, alpha=0.8, edgecolors='none',)
    
    add_textonly_legend(axs[0], {f"{fov}": {"color": color_by_fov[fovs.index(fov)]} for fov in fovs}, ncol=2, fontsize=6)
    add_textonly_legend(axs[1], {f"{mouse}": {"color": color_by_mice[mice.index(mouse)]} for mouse in mice})
    
    save_path = routing.default_fig_path(dataset, "BehaviorCorrelationDistribution.png")
    default_exit_save(fig, save_path)
    
    behavior_corr_index = {cs_node: corr_dict["whisker"] + corr_dict["locomotion"]
                           for cs_node, corr_dict in all_corrs.items()}
    corr_index = sorted(behavior_corr_index.items(), key=lambda x: x[1], reverse=True)

    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)
    plot_manual_whisk = PlotManual(whisker=True, baseline_subtraction=None)
    plot_manual_loco = PlotManual(locomotion=True, baseline_subtraction=None)
    coroutines = {}
    grand_fluo_order = {}
    fig, axs = plt.subplots(7, len(corr_index), figsize=(20, 5), height_ratios=[1, 1, 1, 0.4, 1, 1, 1])
    for col_index, (cs_node, corr_index1) in enumerate(corr_index):
        axs[3, col_index].remove()
        cs_subtree = dataset.subtree(cs_node)
        type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                    plot_manual=plot_manual_fluo,
                                                    _element_trial_level =_element_trial_level,)
        if len(type2dataset) == 0:
            continue
        
        # get amp diff
        group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) 
                                                        for single_fluorescence in select_truthy_items(
                                                            [node.data.fluorescence 
                                                             for node in sync_nodes(cs_subtree.select("trial"), alignment_events, plot_manual=plot_manual_fluo)])], 
                                            baseline_subtraction=None)
        amplitude_range_in_frame1 = np.searchsorted(group_fluorescence.t, (0, 5))
        amplitude_range_in_frame2 = np.searchsorted(group_fluorescence.t, (-5, 0))
        amplitudes1 = np.nanmean(group_fluorescence.raw_array[:, amplitude_range_in_frame1[0]:amplitude_range_in_frame1[1]], axis=1)
        amplitudes2 = np.nanmean(group_fluorescence.raw_array[:, amplitude_range_in_frame2[0]:amplitude_range_in_frame2[1]], axis=1)
        amplitude_diff = amplitudes1 - amplitudes2
        grand_fluo_order[cs_node] = np.sum(amplitude_diff)



        for type_idx, (trial_type, raw_type_dataset) in enumerate(type2dataset.items()):
            type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

            # sort by fluorescence amplitude
            group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) 
                                                        for single_fluorescence in select_truthy_items(
                                                            [node.data.fluorescence for node in type_dataset])], 
                                            baseline_subtraction=None)
            fluorescence_order = get_amplitude_diff_sorted_idxs(group_fluorescence, amplitude_range1=(0, 5), amplitude_range2=(-5, 0))

            # # sort by whisking amplitude
            # group_whisker = grouping_timeseries(select_truthy_items([node.data.whisker for node in type_dataset]), 
            #                                     baseline_subtraction=None)
            # whisker_order = get_amplitude_sorted_idxs(group_whisker, amplitude_range=sort_range)

            # # sort by locomotion amplitude
            # group_locomotion = grouping_events_rate(select_truthy_items([node.data.locomotion for node in type_dataset]), 
            #                                         bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=None)
            # locomotion_order = get_amplitude_sorted_idxs(group_locomotion, amplitude_range=sort_range)
            
            for content_idx, (modality_name, specific_plot_manual) in enumerate(zip(
                ["fluorescence", "whisker", "locomotion",], 
                [plot_manual_fluo, plot_manual_whisk, plot_manual_loco])):                    

                specific_plotter = heatmap_view(ax=axs[type_idx*4+content_idx, col_index], datasets=type_dataset, sync_events=alignment_events, 
                                            plot_manual=specific_plot_manual, modality_name=modality_name, specified_order=fluorescence_order)
                coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}_sorted"
        # coroutine_cycle(coroutines)
    for ax in axs.flatten():
        ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='x', labeltop=False)
        ax.set_yticks([])
        ax.set_xlim(-5, 5)
        ax.set_xticks([])
        # ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    fig.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, wspace=0, hspace=0)
        
    # save_path = routing.default_fig_path(dataset, "BehaviorCorrelationSortedOverview.png")
    # default_exit_save(fig, save_path)


    sorted_grand_fluo_order = sorted(grand_fluo_order.items(), key=lambda x: x[1], reverse=True)
    # fig, axs = plt.subplots(7, len(corr_index), figsize=(20, 5), height_ratios=[1, 1, 1, 0.4, 1, 1, 1])
    # for col_index, (cs_node, corr_index1) in enumerate(sorted_grand_fluo_order):
    #     axs[3, col_index].remove()
    #     cs_subtree = dataset.subtree(cs_node)
    #     type2dataset = split_dataset_by_trial_type(cs_subtree, 
    #                                                 plot_manual=plot_manual_fluo,
    #                                                 _element_trial_level =_element_trial_level,)
    #     if len(type2dataset) == 0:
    #         continue
        
    #     for type_idx, (trial_type, raw_type_dataset) in enumerate(type2dataset.items()):
    #         type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

    #         # sort by fluorescence amplitude
    #         group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) 
    #                                                     for single_fluorescence in select_truthy_items(
    #                                                         [node.data.fluorescence for node in type_dataset])], 
    #                                         baseline_subtraction=None)
    #         fluorescence_order = get_amplitude_diff_sorted_idxs(group_fluorescence, amplitude_range1=(0, 5), amplitude_range2=(-5, 0))

    #         # # sort by whisking amplitude
    #         # group_whisker = grouping_timeseries(select_truthy_items([node.data.whisker for node in type_dataset]), 
    #         #                                     baseline_subtraction=None)
    #         # whisker_order = get_amplitude_sorted_idxs(group_whisker, amplitude_range=sort_range)

    #         # # sort by locomotion amplitude
    #         # group_locomotion = grouping_events_rate(select_truthy_items([node.data.locomotion for node in type_dataset]), 
    #         #                                         bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=None)
    #         # locomotion_order = get_amplitude_sorted_idxs(group_locomotion, amplitude_range=sort_range)
            
    #         for content_idx, (modality_name, specific_plot_manual) in enumerate(zip(
    #             ["fluorescence", "whisker", "locomotion",], 
    #             [plot_manual_fluo, plot_manual_whisk, plot_manual_loco])):                    

    #             specific_plotter = heatmap_view(ax=axs[type_idx*4+content_idx, col_index], datasets=type_dataset, sync_events=alignment_events, 
    #                                         plot_manual=specific_plot_manual, modality_name=modality_name, specified_order=fluorescence_order)
    #             coroutines[specific_plotter] = f"{trial_type}_heatmap_view_{modality_name}_sorted"
    #     # coroutine_cycle(coroutines)
    # for ax in axs.flatten():
    #     ax.tick_params(axis='y', labelleft=False)
    #     ax.tick_params(axis='x', labeltop=False)
    #     ax.set_yticks([])
    #     ax.set_xlim(-5, 5)
    #     ax.set_xticks([])
    #     # ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    # fig.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, wspace=0, hspace=0)
        
    # # save_path = routing.default_fig_path(dataset, "FluorescenceAmplitudeSortedOverview.png")
    # # default_exit_save(fig, save_path)


    avg_PSTH = defaultdict(list)
    type_specific_timeline = defaultdict(list)
    for cs_node, corr_index1 in sorted_grand_fluo_order:
        cs_subtree = dataset.subtree(cs_node)
        type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                    plot_manual=plot_manual_fluo,
                                                    _element_trial_level =_element_trial_level,)
        if len(type2dataset) == 0:
            continue
        
        for trial_type, raw_type_dataset in type2dataset.items():
            type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
            group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) 
                                                        for single_fluorescence in select_truthy_items(
                                                            [node.data.fluorescence for node in type_dataset])], 
                                            baseline_subtraction=None)
            avg_PSTH[trial_type].append(group_fluorescence.mean_ts)
            type_specific_timeline[trial_type].append(type_dataset.nodes[0].data.timeline)
    
    avg_PSTH = {k: grouping_timeseries(v) for k, v in avg_PSTH.items()} 
    fig, ax = plt.subplots(1, len(avg_PSTH), figsize=(6, 3), constrained_layout=True)
    for idx, (trial_type, group_fluorescence) in enumerate(avg_PSTH.items()):
        heatmap_extent = (group_fluorescence.t[0], group_fluorescence.t[-1], 10, 1)
        ax[idx].imshow(group_fluorescence.raw_array, extent=heatmap_extent, cmap="RdYlBu_r", vmin=-1.5, vmax=1.5,
                       **style_dicts.HEATMAP_STYLE)
        ax[idx].set_title(trial_type)
        default_ax_realign(ax[idx])
        label_heatmap_y_ticklabels(ax[idx], group_fluorescence.raw_array.shape[0], (1, 10))
        unit_plot_timeline(timeline=type_specific_timeline[trial_type], ax=ax[idx], y_offset=0, ratio=1.0)
    save_path = routing.default_fig_path(dataset, "AveragePSTH.png")
    default_exit_save(fig, save_path)