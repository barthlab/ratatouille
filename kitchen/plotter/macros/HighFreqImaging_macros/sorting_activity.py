from collections import defaultdict
import logging

from matplotlib.colors import Normalize
from pyparsing import Optional

from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import AdvancedTimeSeries, calculate_group_tuple, grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.macros.HighFreqImaging_macros.posthoc_staining_analysis import register_staining_info
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Node
from kitchen.configs.routing import default_data_path, search_pattern_file
from kitchen.structure.neural_data_structure import TimeSeries, TimeSeries_concat
from kitchen.utils.sequence_kit import find_only_one, select_truthy_items
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.utils.ordering_kit import linkage_order


import os.path as path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distinctipy


logger = logging.getLogger(__name__)


plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 6



plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    
def get_saving_path2():
    return routing.robust_path_join(
        routing.FIGURE_PATH,
        "PassivePuff_HighFreqImaging", 
        "HighFreqImaging_Combine_ROIs",
    )

def get_celltype_deconv_fluo(
    BINSIZE = 20/1000,  # s
    FEATURE_RANGE = (-2, 2.5),  # (min, max) s
    _element_trial_level: str = "trial",
    _alignment_style: str = "Aligned2Trial",
):
    alignment_events = ALL_ALIGNMENT_STYLE[_alignment_style]
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_HighFreqImaging",
        "HighFreqImaging_Combine_ROIs",
        f"ARCHIVE_celltype_deconv_fluo.npy"
    )
    if not os.path.exists(pkl_save_path):  
        bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        dataset = load_dataset(template_id="PassivePuff_HighFreqImaging", cohort_id="HighFreqImaging_Combine_ROIs", 
                            recipe="default_two_photon_mes_parser", name="HFI_SST")

        register_staining_info(dataset, create_small_pic=False, create_big_pic=False)
        all_pkl_data = defaultdict(list)

        for cs_node, cs_subtree in dataset.select_subtree("cellsession"):                
            type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                        plot_manual=plot_manual_fluo,
                                                        _element_trial_level =_element_trial_level,)
            if len(type2dataset) == 0:
                continue
            puff_group_fluorescence = grouping_timeseries(
                [node.data.fluorescence.df_f0.squeeze(0) 
                 for node in sync_nodes(cs_subtree.select("trial", _self=lambda x: x.info.get("trial_type") == "PuffOnly"),
                                        alignment_events, plot_manual=plot_manual_fluo)], 
                baseline_subtraction=None, _predefined_t=bin_centers)
            all_pkl_data["puff_fluo"].append(puff_group_fluorescence.mean)
            blank_group_fluorescence = grouping_timeseries(
                [node.data.fluorescence.df_f0.squeeze(0)
                    for node in sync_nodes(cs_subtree.select("trial", _self=lambda x: x.info.get("trial_type") == "BlankOnly"),
                                        alignment_events, plot_manual=plot_manual_fluo)],
                baseline_subtraction=None, _predefined_t=bin_centers)
            all_pkl_data["blank_fluo"].append(blank_group_fluorescence.mean)

            puff_group_deconv_fluorescence = grouping_timeseries(
                [node.data.fluorescence.delta_deconv_f.squeeze(0)
                 for node in sync_nodes(cs_subtree.select("trial", _self=lambda x: x.info.get("trial_type") == "PuffOnly"),
                                        alignment_events, plot_manual=plot_manual_fluo)], 
                baseline_subtraction=None, _predefined_t=bin_centers)
            all_pkl_data["puff_deconv"].append(puff_group_deconv_fluorescence.mean)
            blank_group_deconv_fluorescence = grouping_timeseries(
                [node.data.fluorescence.delta_deconv_f.squeeze(0)
                 for node in sync_nodes(cs_subtree.select("trial", _self=lambda x: x.info.get("trial_type") == "BlankOnly"),
                                        alignment_events, plot_manual=plot_manual_fluo)],
                baseline_subtraction=None, _predefined_t=bin_centers)
            all_pkl_data["blank_deconv"].append(blank_group_deconv_fluorescence.mean)

            all_pkl_data["red_cohen_d"].append(cs_node.info.get("red_cohen_d", np.nan))
            all_pkl_data["node_name"].append(str(cs_node.coordinate))
            all_pkl_data["day_id"].append(cs_node.info.get("day_id", np.nan))

        np.save(pkl_save_path, {
            "bin_centers": bin_centers,
            "puff_fluo": np.array(all_pkl_data["puff_fluo"]),
            "blank_fluo": np.array(all_pkl_data["blank_fluo"]),
            "puff_deconv": np.array(all_pkl_data["puff_deconv"]),
            "blank_deconv": np.array(all_pkl_data["blank_deconv"]),
            "red_cohen_d": np.array(all_pkl_data["red_cohen_d"]),
            "node_name": np.array(all_pkl_data["node_name"]),
            "day_id": np.array(all_pkl_data["day_id"]),
        })  # type: ignore
        logger.info(f"Saved celltype deconv fluo data to {pkl_save_path}")

    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("Deconv/Fluo loaded from " + pkl_save_path)
    return tmp_save["bin_centers"], tmp_save["puff_fluo"], tmp_save["blank_fluo"], \
        tmp_save["puff_deconv"], tmp_save["blank_deconv"], tmp_save["red_cohen_d"], \
            tmp_save["node_name"], tmp_save["day_id"]
    
    

def sort_all_cell_activity(
        _element_trial_level: str = "trial",
        _alignment_style: str = "Aligned2Trial",
        fluo_range: tuple = (0., 1.),
        deconv_range: tuple = (0.3, 0.5),
):

    alignment_events = ALL_ALIGNMENT_STYLE[_alignment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 6

    fluo_order = {}
    deconv_order = {}

    bin_centers, puff_fluo, blank_fluo, puff_deconv, blank_deconv, red_cohen_d, node_name, day_id = get_celltype_deconv_fluo()

    for single_puff_fluo, single_puff_deconv, single_node_name in zip(puff_fluo, puff_deconv, node_name):
        
        fluo_range_in_frame = np.searchsorted(bin_centers, fluo_range)
        fluo_order[single_node_name] = np.nanmean(single_puff_fluo[fluo_range_in_frame[0]:fluo_range_in_frame[1]])

        deconv_range_in_frame = np.searchsorted(bin_centers, deconv_range)
        deconv_order[single_node_name] = np.nanmean(single_puff_deconv[deconv_range_in_frame[0]:deconv_range_in_frame[1]])


    sorted_fluo_order = sorted(fluo_order.items(), key=lambda x: x[1], reverse=True)
    sorted_deconv_order = sorted(deconv_order.items(), key=lambda x: x[1], reverse=True)
    
    fluo_range_str = f"{fluo_range[0]}-{fluo_range[1]}s"
    deconv_range_str = f"{deconv_range[0]}-{deconv_range[1]}s"

    for order_name, final_order in [
        ("sortby_fluo", sorted_fluo_order), 
        ("sortby_deconv", sorted_deconv_order), 
                                    ]:
        range_str = fluo_range_str if "fluo" in order_name else deconv_range_str if "deconv" in order_name else "NA"
        
        final_order_index = [list(node_name).index(name) for name, _ in final_order]
        reordered_puff_fluo = puff_fluo[final_order_index, :]
        reordered_blank_fluo = blank_fluo[final_order_index, :]
        reordered_puff_deconv = puff_deconv[final_order_index, :]
        reordered_blank_deconv = blank_deconv[final_order_index, :]
        reordered_red_cohen_d = red_cohen_d[final_order_index]
        reordered_day_id = day_id[final_order_index]

        red_cohen_d_list_masked = np.ma.masked_invalid(np.array(reordered_red_cohen_d).reshape(-1, 1))
        day_id_list_masked = np.ma.masked_invalid(np.array(reordered_day_id).reshape(-1, 1))

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "black_magenta",
            ["#000000", "gray", "#ff00ff"]
        )
        cmap.set_bad("gray", alpha=0.01)
        norm = Normalize(vmin=0, vmax=2)
        # cmap = plt.cm.coolwarm.copy()
        # cmap.set_bad("gray", alpha=0.01)
        # norm = Normalize(vmin=0, vmax=2)
        cell_colors = cmap(norm(red_cohen_d_list_masked))


        def unit_plot(puff_data, blank_data, plotting_red_cohen_list, plotting_cell_colors, 
                      v_range, xlim, ylim, save_path, theme_color="black", overlap=False,
                      _annotate=False, _blank_plotting_red_cohen=None):
            fig, ax = plt.subplots(2, 4, figsize=(4., 2.5), 
                                width_ratios=[0.4, 1.2, 1.2, 1.2],
                                height_ratios=[1.2, 0.7],
                                constrained_layout=True)
            
            fig_compare, ax_compare = plt.subplots(1, 1, figsize=(0.9, 1.2), constrained_layout=True)
            ax_compare.tick_params(length=1, pad=1)

            fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(1., 1.2), constrained_layout=True)
            ax_scatter.tick_params(length=1, pad=1)


            for a in ax.flatten():
                a.tick_params(length=1, pad=1)
            if not isinstance(theme_color, list):
                theme_color = [theme_color] * 2
            ax[0, 0].scatter(plotting_red_cohen_list, np.arange(len(plotting_red_cohen_list)) + 0.5, 
                             color=plotting_cell_colors, s=4, alpha=0.8)
            ax[0, 0].set_xticks([0, 2])
            ax[0, 0].set_xlim(-2, 4)
            ax[0, 0].axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax[0, 0].set_yticks([0.5, len(plotting_red_cohen_list) - 0.5], ["1", f"{len(plotting_red_cohen_list)}"])
            ax[0, 0].invert_yaxis()
            ax[0, 0].set_title("Red Cohen's d")
            ax[0, 0].spines[['right', 'top', 'left', ]].set_visible(False)

            annotate_data = {}
            annotate_data_scatter = ([], [])
            for idx, (trial_type, raw_array, cohen_d_data) in enumerate(zip(
                ["Puff", "Blank"], 
                [puff_data, blank_data], 
                [plotting_red_cohen_list, _blank_plotting_red_cohen])):
                group_fluorescence = calculate_group_tuple(raw_array, t=bin_centers)
                sns.heatmap(group_fluorescence.raw_array,  
                            ax=ax[0, idx + 1], cbar=False, 
                            cmap='RdYlBu_r', vmin=-v_range, vmax=v_range)
                # ax[0, idx + 1].set_title(trial_type)
                ax[0, idx + 1].spines[['right', 'top', 'left', 'bottom']].set_visible(True)

                desired_ticks = [0, 0.5]
                tick_positions = [np.argmin(np.abs(group_fluorescence.t - tick)) for tick in desired_ticks]
                ax[0, idx + 1].set_xticks(tick_positions, desired_ticks, rotation=0)
                xlim_ts = np.searchsorted(group_fluorescence.t, xlim)
                ax[0, idx + 1].set_xlim(*xlim_ts)

                compare_t_positions = np.searchsorted(group_fluorescence.t, (0.3, 0.8))
                annotate_data[idx] = np.mean(group_fluorescence.raw_array[:, compare_t_positions[0]:compare_t_positions[1]], axis=1)
                jitter = np.random.uniform(-0.12, 0.12, size=annotate_data[idx].shape)

                if _blank_plotting_red_cohen is not None:
                    ax_scatter.scatter(cohen_d_data, annotate_data[idx], color=[cmap(norm(single_cd)) for single_cd in cohen_d_data], 
                                    lw=0.5, s=4, alpha=0.8)
                    annotate_data_scatter[0].append(np.array(cohen_d_data).flatten())
                    annotate_data_scatter[1].append(annotate_data[idx].flatten())
                ax_compare.scatter(idx + jitter, annotate_data[idx], edgecolor=theme_color[idx], facecolor='white',
                                   lw=0.5, s=8, alpha=1, zorder=2)
                ax_compare.errorbar(idx + 0.3, np.mean(annotate_data[idx]), yerr=np.std(annotate_data[idx]), 
                                    fmt="o",
                                    markeredgecolor="black",
                                    markerfacecolor=theme_color[idx],
                                    markeredgewidth=0.5,
                                    markersize=5,
                                    ecolor='black',
                                    elinewidth=1,
                                    capsize=2,
                                    capthick=1,
                                    alpha=1,
                                    )


                if idx == 0:
                    ax[0, idx + 1].set_yticks([0.5, len(plotting_red_cohen_list) - 0.5], 
                                            ["1", f"{len(plotting_red_cohen_list)}"], rotation=0)
                else:
                    ax[0, idx + 1].set_yticks([0.5, len(raw_array) - 0.5], ["1", f"{len(raw_array)}"], rotation=0)

                if overlap:
                    oreo_plot(ax[1, -1], 
                            group_fluorescence, y_offset=0, ratio=1.0,
                                trace_style = {"color": theme_color[idx], "linewidth": 1, "alpha": 0.7, },
                                fill_between_style=style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.2, })
                    if idx == 0:                    
                        ax[1, -1].axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
                        ax[1, -1].axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
                        ax[1, -1].spines[['right', 'top', ]].set_visible(False)

                        ax[1, -1].set_xlim(*xlim)
                        ax[1, -1].set_ylim(*ylim)
                        ax[1, -1].set_xticks([ 0, 0.5], ["0", "0.5"], rotation=0)
                    ax[1, idx+1].remove()
                else:
                    oreo_plot(ax[1, idx + 1], 
                            group_fluorescence, y_offset=0, ratio=1.0,
                                trace_style = {"color": theme_color[idx], "linewidth": 1, "alpha": 0.7, },
                                fill_between_style=style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.2, })
                    ax[1, idx + 1].axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
                    ax[1, idx + 1].axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
                    ax[1, idx + 1].spines[['right', 'top', ]].set_visible(False)

                    ax[1, idx + 1].set_xlim(*xlim)
                    ax[1, idx + 1].set_ylim(*ylim)
                    ax[1, idx + 1].set_xticks([ 0, 0.5], ["0", "0.5"], rotation=0)
            ax[0, 0].set_ylim(ax[0, 1].get_ylim())
            default_exit_save(fig, save_path, _transparent=True)

            if _annotate:
                basic_ttest.stats_pairwise_unpaired_ttest(annotate_data, ax_compare)
            ax_compare.set_xticks([0.15, 1.15], ["", ""],)
            ax_compare.set_xlim(-0.4, 1.7)
            ax_compare.spines[['right', 'top']].set_visible(False)
            default_exit_save(fig_compare, save_path.replace(".png", f"_compare{'_annotated' if _annotate else ''}.png"), _transparent=True)
            if _blank_plotting_red_cohen is not None:
                ax_scatter.set_xlim(-1.5, 3.5)
                ax_scatter.spines[['right', 'top']].set_visible(False)
                if _annotate:
                    basic_ttest.add_linear_regression(ax_scatter, np.concatenate(annotate_data_scatter[0]), np.concatenate(annotate_data_scatter[1]),
                                                      theme_color='gray')
                default_exit_save(fig_scatter, save_path.replace(".png", f"_scatter{'_annotated' if _annotate else ''}.png"), _transparent=True)


        PSTH_v_range = 1.0
        fluo_v_range = 1.0
        # save_path = path.join(get_saving_path2(), f"AveragePSTH_{order_name}_{range_str}.png")
        # unit_plot(reordered_puff_deconv, reordered_blank_deconv, red_cohen_d_list_masked, cell_colors,
        #           v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.2, 0.8), save_path=save_path, overlap=True)

        # # save_path = path.join(get_saving_path2(), f"AveragePSTH_{order_name}_{range_str}_long.png")
        # # unit_plot(reordered_puff_deconv, reordered_blank_deconv, red_cohen_d_list_masked, cell_colors, 
        # #           v_range=PSTH_v_range, xlim=(-2, 2.5), ylim=(-0.2, 1), save_path=save_path)

        
        # save_path = path.join(get_saving_path2(), f"AverageFluorescence_{order_name}_{range_str}.png")
        # unit_plot(reordered_puff_fluo, reordered_blank_fluo, red_cohen_d_list_masked, cell_colors, 
        #           v_range=fluo_v_range, xlim=(-.5, 1.5), ylim=(-0.1, 0.5), save_path=save_path, overlap=True)

        # day3 = np.where(day_id_list_masked == 3)[0]
        # day3CR = np.where(np.isfinite(np.array(reordered_red_cohen_d).reshape(-1, 1)) & (day_id_list_masked == 3))[0]
        # day4 = np.where(day_id_list_masked == 4)[0]
        # day4CR = np.where(np.isfinite(np.array(reordered_red_cohen_d).reshape(-1, 1)) & (day_id_list_masked == 4))[0]

        # # day4 = np.where(day_id_list_masked == 4)[0]
        # for select_day_name, select_day_type in zip(
        #     ["day3", "CRALL_day3", "day4", "CRALL_day4"],
        #     [day3, day3CR, day4, day4CR]
        #     ):
        #     save_path = path.join(get_saving_path2(), f"AveragePSTH_{order_name}_{range_str}_{select_day_name}.png")
        #     unit_plot(reordered_puff_deconv[select_day_type], reordered_blank_deconv[select_day_type],
        #               red_cohen_d_list_masked[select_day_type], cell_colors[select_day_type], 
        #               v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.2, 0.8), save_path=save_path, overlap=True)

        #     save_path = path.join(get_saving_path2(), f"AverageFluorescence_{order_name}_{range_str}_{select_day_name}.png")
        #     unit_plot(reordered_puff_fluo[select_day_type], reordered_blank_fluo[select_day_type],
        #               red_cohen_d_list_masked[select_day_type], cell_colors[select_day_type],
        #               v_range=fluo_v_range, xlim=(-.5, 1.5), ylim=(-0.1, 0.5), save_path=save_path, overlap=True)

        CR_pos_day3 = np.where((red_cohen_d_list_masked > 1) & (day_id_list_masked == 3))[0]
        CR_neg_day3 = np.where((red_cohen_d_list_masked < 1) & (day_id_list_masked == 3))[0]
        # CR_pos_day4 = np.where((red_cohen_d_list_masked > 1) & (day_id_list_masked == 4))[0]
        # CR_neg_day4 = np.where((red_cohen_d_list_masked < 1) & (day_id_list_masked == 4))[0]

        save_path = path.join(get_saving_path2(), f"AveragePSTH_{order_name}_{range_str}_CRposvsCRneg.png")
        unit_plot(reordered_puff_deconv[CR_pos_day3], reordered_puff_deconv[CR_neg_day3],
                    red_cohen_d_list_masked[CR_pos_day3], cell_colors[CR_pos_day3], theme_color=[f"#ff00ff", f"#000000"],
                    v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.2, 1), save_path=save_path, overlap=True,
                    _blank_plotting_red_cohen=red_cohen_d_list_masked[CR_neg_day3])
        
        save_path = path.join(get_saving_path2(), f"AveragePSTH_{order_name}_{range_str}_CRposvsCRneg.png")
        unit_plot(reordered_puff_deconv[CR_pos_day3], reordered_puff_deconv[CR_neg_day3],
                    red_cohen_d_list_masked[CR_pos_day3], cell_colors[CR_pos_day3], theme_color=[f"#ff00ff", f"#000000"],
                    v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.2, 1), save_path=save_path, overlap=True, _annotate=True,
                    _blank_plotting_red_cohen=red_cohen_d_list_masked[CR_neg_day3])

        save_path = path.join(get_saving_path2(), f"AverageFluorescence_{order_name}_{range_str}_CRposvsCRneg.png")
        unit_plot(reordered_puff_fluo[CR_pos_day3], reordered_puff_fluo[CR_neg_day3],
                    red_cohen_d_list_masked[CR_pos_day3], cell_colors[CR_pos_day3], theme_color=[f"#ff00ff", f"#000000"],
                    v_range=fluo_v_range, xlim=(-0.5, 1.5), ylim=(-0.1, 0.5), save_path=save_path, overlap=True,
                    _blank_plotting_red_cohen=red_cohen_d_list_masked[CR_neg_day3])

        save_path = path.join(get_saving_path2(), f"AverageFluorescence_{order_name}_{range_str}_CRposvsCRneg.png")
        unit_plot(reordered_puff_fluo[CR_pos_day3], reordered_puff_fluo[CR_neg_day3],
                    red_cohen_d_list_masked[CR_pos_day3], cell_colors[CR_pos_day3], theme_color=[f"#ff00ff", f"#000000"],
                    v_range=fluo_v_range, xlim=(-0.5, 1.5), ylim=(-0.1, 0.5), save_path=save_path, overlap=True, _annotate=True,
                    _blank_plotting_red_cohen=red_cohen_d_list_masked[CR_neg_day3])
       
       