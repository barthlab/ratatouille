from collections import defaultdict

from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import scipy

from kitchen.calculator.basic_metric import AVERAGE_VALUE, PEARSON_CORRELATION
from kitchen.configs import routing
from kitchen.operator.grouping import AdvancedTimeSeries, grouping_events_histogram, grouping_events_rate, grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import COHORT_BRIGHTER_COLORS, COHORT_COLORS
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import LOCOMOTION_BIN_SIZE
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.stats_tests.basic_ttest import add_loglog_regression
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Node
from kitchen.configs.routing import default_data_path, search_pattern_file
from kitchen.structure.neural_data_structure import TimeSeries_concat
from kitchen.utils import numpy_kit
from kitchen.utils.sequence_kit import find_only_one, select_truthy_items
from kitchen.plotter.utils.tick_labels import add_textonly_legend

import os.path as path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distinctipy
from scipy.stats import linregress, mannwhitneyu, ttest_ind


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6
from matplotlib.colors import LinearSegmentedColormap
CR_POS_COLOR = "#ff00ff"
CR_NEG_COLOR = "#000000"
cmap = LinearSegmentedColormap.from_list(
"black_magenta",
[CR_NEG_COLOR, "gray", CR_POS_COLOR]
)
cmap.set_bad("gray", alpha=0.01)
norm = Normalize(vmin=0, vmax=2)


def visualize_single_cell_behavior_summary(cs_node: Node, all_puff_trials,
                                           all_fluos, all_locos, all_whiskers, save_path):



    mean_range, var_range = (-1, 10), (-1, 10)
    
    # mean variance position
    fluo_avg = np.nanmean(all_fluos)
    fluo_var = np.nanstd(all_fluos)

    fig1, ax1 = plt.subplots(1, 1, figsize=(1, 1), constrained_layout=True)
    ax1.tick_params(length=1, pad=1)
    ax1.spines[["top", "right"]].set_visible(False)

    ax1.scatter(fluo_avg, fluo_var, c=cmap(norm(cs_node.info.get("red_cohen_d", np.nan))), s=8, edgecolors='k', lw=0.5,)
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_yscale("symlog", linthresh=0.01)
    ax1.set_xlim(*mean_range)
    ax1.set_ylim(*var_range)

    xs = np.linspace(*mean_range, 100)
    ax1.plot(xs, xs, c="gray", lw=0.5, ls="--", zorder=-1)
    ax1.plot(xs, 2*xs, c="gray", lw=0.5, ls="--", zorder=-1)
    ax1.plot(xs, 0.5*xs, c="gray", lw=0.5, ls="--", zorder=-1)
    
    ax1.set_xlabel("Mean", labelpad=1)
    ax1.set_ylabel("SD", labelpad=1)

    
    default_exit_save(fig1, save_path + "_mean_variance.png", _transparent=True)


    fig2, ax2 = plt.subplots(1, 1, figsize=(1.6, 0.6), constrained_layout=True)
    ax2.tick_params(length=1, pad=1)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    fluo = cs_node.data.fluorescence.detrend_z_score.squeeze(0)
    ax2.plot(fluo.t, fluo.v, c="black", lw=0.3, alpha=0.9, zorder=3)
    # locomotion_ts = cs_node.data.locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE, min_t=fluo.t.min(), max_t=fluo.t.max()).z_score()
    # ax2.plot(locomotion_ts.t, locomotion_ts.v + 10, c="blue", lw=0.3, alpha=0.9, zorder=2)
    # whisker_ts = cs_node.data.whisker.z_score()
    # ax2.plot(whisker_ts.t, whisker_ts.v + 20, c="green", lw=0.3, alpha=0.9, zorder=1)

    ax2.set_xlim(fluo.t.min(), fluo.t.max())
    ax2.set_xticks(np.linspace(fluo.t.min(), fluo.t.max(), 5), ["0", "150", "300", "450", "600"])
    ax2.set_xlabel("Time (s)", labelpad=1)
    ax2.set_yticks([0, 1])


    default_exit_save(fig2, save_path + "_traces.png", _transparent=True)

    fig2, ax2 = plt.subplots(1, 1, figsize=(1.6 * 12, 0.6), constrained_layout=True)
    ax2.tick_params(length=1, pad=1)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    fluo = cs_node.data.fluorescence.detrend_z_score.squeeze(0)
    ax2.plot(fluo.t, fluo.v, c="black", lw=0.5, alpha=0.9, zorder=3)
    # locomotion_ts = cs_node.data.locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE, min_t=fluo.t.min(), max_t=fluo.t.max()).z_score()
    # ax2.plot(locomotion_ts.t, locomotion_ts.v + 10, c="blue", lw=0.5, alpha=0.9, zorder=2)
    # whisker_ts = cs_node.data.whisker.z_score()
    # ax2.plot(whisker_ts.t, whisker_ts.v + 20, c="green", lw=0.5, alpha=0.9, zorder=1)
    for puff_t in cs_node.data.timeline.filter("VerticalPuffOn").t:
        ax2.axvline(puff_t, color='gray', lw=0.3, alpha=0.9, zorder=0, ls='--')

    ax2.set_xlim(fluo.t.min(), fluo.t.max())
    ax2.set_xticks(np.linspace(fluo.t.min(), fluo.t.max(), 5), ["0", "150", "300", "450", "600"])
    ax2.set_xlabel("Time (s)", labelpad=1)
    ax2.set_yticks([])
    default_exit_save(fig2, save_path + "_traces_long.png", _transparent=True)
    
    fig3, ax3 = plt.subplots(1, 3, figsize=(2.2, 0.9), constrained_layout=True)

    fluo_array = grouping_timeseries([node.data.fluorescence.df_f0.squeeze(0) for node in all_puff_trials],)
    loco_array = grouping_events_rate([node.data.locomotion for node in all_puff_trials], bin_size=LOCOMOTION_BIN_SIZE)
    whisker_array = grouping_timeseries([node.data.whisker for node in all_puff_trials])

    def plot_heatmap(ax, adv_ts: AdvancedTimeSeries, xtick, xlim, **kwargs):
        ax.tick_params(length=1, pad=1)
        
        sns.heatmap(adv_ts.raw_array, ax=ax, cbar=False, **kwargs)

        tick_positions = [np.argmin(np.abs(adv_ts.t - tick)) for tick in xtick]
        ax.set_xticks(tick_positions, xtick, rotation=0)
        xlim_ts = np.searchsorted(adv_ts.t, xlim)
        ax.set_xlim(*xlim_ts)
        ax.set_yticks([0.5, len(adv_ts) - 0.5], ["1", f"{len(adv_ts)}"], rotation=0)
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
    
    plot_heatmap(ax3[0], fluo_array, xtick=[0, 0.5], xlim=(-0.5, 1.5), cmap='RdYlBu_r', vmin=-1, vmax=1)    
    plot_heatmap(ax3[1], loco_array, xtick=[-3, 0, 3], xlim=(-5, 5), cmap='Blues', vmin=0, vmax=1)    
    plot_heatmap(ax3[2], whisker_array, xtick=[-3, 0, 3], xlim=(-5, 5), cmap='Greens', vmin=0, vmax=1)    


    default_exit_save(fig3, save_path + "_heatmaps.png", _transparent=True)

    # from scipy import stats
    # fig4, ax4 = plt.subplots(1, 1, figsize=(1.5, 1), constrained_layout=True)
    # behav_range_rolling = np.array([(range_start, range_start+2) for range_start in np.linspace(-4, 2, 100)]  )
    # behav_range_center = (behav_range_rolling[:, 0] + behav_range_rolling[:, 1]) / 2
    # behav_range_center_offset = behav_range_center - 0.5

    # ax4.spines[["top", "right"]].set_visible(False)
    # ax4.tick_params(length=1, pad=1)

    # loco_corr_rolling = [stats.spearmanr(single_loco, np.array(all_fluos)).statistic for single_loco in np.array(all_locos).T]
    # ax4.plot(behav_range_center_offset, loco_corr_rolling, c="blue", lw=1, alpha=0.8)
    # # ax4.axvline(x=behav_range_center_offset[np.argmax(loco_corr_rolling)], c="blue", lw=0.5, ls="--")

    # whisker_corr_rolling = [stats.spearmanr(single_whisker, np.array(all_fluos)).statistic for single_whisker in np.array(all_whiskers).T]
    # ax4.plot(behav_range_center_offset, whisker_corr_rolling, c="green", lw=1, alpha=0.8)
    # # ax4.axvline(x=behav_range_center_offset[np.argmax(whisker_corr_rolling)], c="green", lw=0.5, ls="--")
    
    # ax4.axvline(x=0, c='gray', lw=0.5, ls="--")
    # ax4.set_xlabel("Time offset (s)", labelpad=1)
    # ax4.set_ylim(-0.8, 0.8)

    # default_exit_save(fig4, save_path + "_rolling_corr.png", _transparent=True)

    # fig5, ax5 = plt.subplots(1, 3, figsize=(3, 1), constrained_layout=True, width_ratios=[1, 1, 0.5])
    # for ax in ax5:
    #     ax.tick_params(length=1, pad=1)
    # sns.heatmap(np.array(all_locos), ax=ax5[0], cmap='Blues')
    # sns.heatmap(np.array(all_whiskers), ax=ax5[1], cmap='Greens')
    # sns.heatmap(np.array(all_fluos).reshape(-1, 1), ax=ax5[2], cmap='RdYlBu_r')

    # default_exit_save(fig5, save_path + "_raw_matrix_for_corr.png", _transparent=True)


def visualize_all_cell_behavior_summary(dataset: DataSet, save_path):

    mean_range, var_range = (0.01, 100), (0.01, 100)
    

    fig1, ax1 = plt.subplots(1, 1, figsize=(1.4, 1.4), constrained_layout=True)
    ax1.tick_params(length=1, pad=1, which='both')
    ax1.spines[["bottom", "left"]].set_visible(False)

    ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, which='both')
    ax1.tick_params(right=True, labelright=True, left=False, labelleft=False, which='both')

    ax1.xaxis.set_label_position("top")
    ax1.yaxis.set_label_position("right")

    CR_pos_cv, CR_neg_cv = [], []
    for cs_node in dataset.select("cellsession"):
        
        fluo_avg = np.nanmean(cs_node.info["trial_level_corr"]["fluo"])
        fluo_var = np.nanstd(cs_node.info["trial_level_corr"]["fluo"])

        ax1.scatter(fluo_avg, fluo_var, facecolors=COHORT_BRIGHTER_COLORS["SST_JUX"], s=5, edgecolors='w', lw=0.3, marker='s',)

        if cs_node.info.get("red_cohen_d", np.nan) > 1:
            CR_pos_cv.append(fluo_var / fluo_avg)
        elif cs_node.info.get("red_cohen_d", np.nan) <= 1:
            CR_neg_cv.append(fluo_var / fluo_avg)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(*mean_range)
    ax1.set_ylim(*var_range)
    ax1.set_xticks([0.01, 1, 100],)
    ax1.set_yticks([0.01, 1, 100],)

    ax1.set_aspect('equal', adjustable='box') 

    # xs = np.linspace(*mean_range, 100)
    # ax1.plot(xs, xs, c="gray", lw=0.5, ls="--", zorder=-1)
    # ax1.plot(xs, 4*xs, c="gray", lw=0.5, ls="--", zorder=-1)
    # ax1.plot(xs, 0.25*xs, c="gray", lw=0.5, ls="--", zorder=-1)

    all_cell_data_tuple = np.array([(np.nanmean(cs_node.info["trial_level_corr"]["fluo"]), np.nanstd(cs_node.info["trial_level_corr"]["fluo"])) 
                                    for cs_node in dataset.select("cellsession") 
                                    if np.nanmean(cs_node.info["trial_level_corr"]["fluo"]) > 0])
    add_loglog_regression(ax1, all_cell_data_tuple[:, 0], all_cell_data_tuple[:, 1], theme_color=COHORT_COLORS["SST_JUX"])
    
    ax1.set_xlabel(f"{DF_F0_SIGN} mean", labelpad=1)
    ax1.set_ylabel(f"{DF_F0_SIGN} SD", labelpad=1)

    # for spine in ax1.spines.values():
    #     spine.set_edgecolor('gray')
    # ax1.tick_params(which='both', axis='both', color='gray', labelcolor='gray',)
    # ax1.xaxis.label.set_color('gray')
    # ax1.yaxis.label.set_color('gray')

    
    default_exit_save(fig1, save_path + "_mean_variance.png", _transparent=True)

    fig2, ax2 = plt.subplots(1, 1, figsize=(0.7, 0.4), constrained_layout=True)
    ax2.tick_params(length=1, pad=1, which='both')
    ax2.spines[["top", "right", "left"]].set_visible(False)

    CR_pos_cv = np.array(CR_pos_cv)
    CR_neg_cv = np.array(CR_neg_cv)
    all_CV = np.concatenate([CR_neg_cv, CR_pos_cv])
    
    sns.kdeplot(CR_neg_cv, ax=ax2, fill=True, color=CR_NEG_COLOR, alpha=0.6, lw=0, log_scale=True)
    sns.kdeplot(CR_pos_cv, ax=ax2, fill=True, color=CR_POS_COLOR, alpha=0.6, lw=0, log_scale=True)
    sns.kdeplot(all_CV, ax=ax2, fill=False, color="black", lw=0.5, ls="--", log_scale=True)

    ax2.invert_xaxis()
    ax2.set_xlim(10, 0.1)
    ax2.set_xticks([0.2, 1, 5], ["1/5", "1", "5"])
    # ax2.set_xticks([], minor=True)
    
    # ax2.set_xlabel("CV", labelpad=1)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    default_exit_save(fig2, save_path + "_cv_distribution.png", _transparent=True)

    np.save(save_path + "_calcium_cv.npy", all_CV)
    ks_test_result = scipy.stats.ks_2samp(CR_neg_cv, CR_pos_cv)
    print(f"KS test result: {ks_test_result}")


    fig3, ax3 = plt.subplots(1, 1, figsize=(3 if "day" in save_path else 4, 1.6), constrained_layout=True)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(length=1, pad=1, which='both')


    all_cs_node = dataset.select("cellsession").nodes
    all_cs_node_red_cohen_d = np.array([cs_node.info.get("red_cohen_d", np.nan) for cs_node in all_cs_node])
    CR_neg_cells = np.argwhere(all_cs_node_red_cohen_d <= 1)
    CR_pos_cells = np.argwhere(all_cs_node_red_cohen_d > 1)
    UnRegistered_cells = np.argwhere(np.isnan(all_cs_node_red_cohen_d))

    n_neg, n_pos, n_unregistered = len(CR_neg_cells), len(CR_pos_cells), len(UnRegistered_cells)

    for x_offset, cell_indices, group_name, group_color in zip(
        [0, n_pos, n_neg + n_pos],
        [CR_pos_cells, CR_neg_cells, UnRegistered_cells],
        ["CR+", "CR-", "UnRegistered"],
        [CR_POS_COLOR, CR_NEG_COLOR, "darkgray"]
    ):
        cell_median_fluo = [np.median(all_cs_node[int(cell_idx)].info["trial_level_corr"]["fluo"]) for cell_idx in cell_indices]
        sorted_order = np.argsort(cell_median_fluo)[::-1]
        cell_indices = cell_indices[sorted_order].flatten()

        for cell_rel_idx, cell_idx in enumerate(cell_indices):
            cell_node = all_cs_node[int(cell_idx)]
            cell_all_fluo = cell_node.info["trial_level_corr"]["fluo"]
            # single_cell_color = cmap(norm(cell_node.info.get("red_cohen_d", np.nan)))
            sns.stripplot(x=x_offset + cell_rel_idx, y=cell_all_fluo, edgecolor='none', 
                          size=1, jitter=0.3,
                          color="gray", ax=ax3)
            sns.boxplot(x=x_offset + cell_rel_idx, y=cell_all_fluo, linewidth=0.8,
                        boxprops={"facecolor": "none", "edgecolor": group_color},
                        whiskerprops={"color": group_color},
                        capprops={"color": group_color},
                        medianprops={"color": group_color},
                        showfliers=False,
                        width=0.6, 
                        ax=ax3)
    ax3.axhline(0, color='gray', lw=0.5, zorder=-1)
    ax3.set_xticks([n_pos/2, n_pos + n_neg/2, n_neg + n_pos + n_unregistered/2],
                    ["CR+", "CR-", "Unregistered"])
    for label, color in zip(ax3.get_xticklabels(), [CR_POS_COLOR, CR_NEG_COLOR, "darkgray"]):
        label.set_color(color)
    ax3.axvline(n_pos-0.5, color='gray', lw=0.5, ls='--', zorder=-1)
    ax3.axvline(n_neg + n_pos-0.5, color='gray', lw=0.5, ls='--', zorder=-1)
    ax3.set_yticks([0, 1, 2, 3])
    # ax3.set_yscale("log")


    default_exit_save(fig3, save_path + "_fluo_distribution.png", _transparent=True)




    def plot_metric_correlational_matrix(current_metrics, save_path):
        
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), constrained_layout=True)
        ax.tick_params(length=1, pad=1)
        shared_indices = None
        for key_name in current_metrics.keys():
            valid_indices_per_metric = np.isfinite(current_metrics[key_name])
            if shared_indices is None:
                shared_indices = valid_indices_per_metric
            else:
                shared_indices = shared_indices & valid_indices_per_metric
        correlation_matrix = np.corrcoef([np.array(current_metrics[f"{p_name}"])[shared_indices] 
                                        for p_name in current_metrics.keys()])
        sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm", 
                    cbar=False, vmin=-1, vmax=1, ax=ax, square=True, annot_kws={"size": 5})
        ax.set_xticks(np.arange(len(current_metrics.keys())) + 0.5)
        ax.set_xticklabels(current_metrics.keys(), rotation=45, ha='right')
        ax.set_yticks(np.arange(len(current_metrics.keys())) + 0.5)
        ax.set_yticklabels(current_metrics.keys(), rotation=0)
        default_exit_save(fig, save_path, _transparent=True)





    all_metrics = {
        "Mean": numpy_kit._mean,
        "Var": numpy_kit._variance,

        "Mean-CD": lambda x: numpy_kit._mean(np.abs(np.diff(x))),
        "Var-CD": lambda x: numpy_kit._variance(np.diff(x)),
        
        "CV": numpy_kit._coefficient_of_variation,
        "FF": numpy_kit._fano_factor,
        "PR": numpy_kit._normalized_participation_ratio,

        "CV-CD": lambda x: numpy_kit._coefficient_of_variation(np.diff(x)),
        "FF-CD": lambda x: numpy_kit._fano_factor(np.diff(x)),
        "PR-CD": lambda x: numpy_kit._normalized_participation_ratio(np.diff(x)),

        "RCD": numpy_kit._relative_consecutive_difference,
        "lag1 AC": numpy_kit._lag1_autocorrelation
    }
    display_metrics = [
        "CV", "FF", "CV-CD", "PR-CD", "lag1 AC"
    ]

    fig4, ax4 = plt.subplots(1, len(display_metrics), figsize=(2.2, 1), constrained_layout=True)
    for ax in ax4:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=1, pad=1, which='both')
        ax.set_xlim(-0.3, 0.3)

    global_metrics = {key: [] for key in all_metrics}
    annotation_dict = {key: defaultdict(list) for key in all_metrics}
    for x_offset, cell_indices, group_name, group_color in zip(
        [-0.15, 0.15],
        [CR_pos_cells, CR_neg_cells],
        ["CR+", "CR-"],
        [CR_POS_COLOR, CR_NEG_COLOR]
        # [-0.3, 0, 0.3],
        # [CR_pos_cells, CR_neg_cells, UnRegistered_cells],
        # ["CR+", "CR-", "UnRegistered"],
        # [CR_POS_COLOR, CR_NEG_COLOR, "darkgray"]
    ):
        group_cell_data_list = [all_cs_node[int(cell_idx)].info["trial_level_corr"]["fluo"] for cell_idx in cell_indices]

        temp_metrics = {}
        accummulated_metric_id = 0
        for metric_id, (metric_name, metric_func) in enumerate(all_metrics.items()):

            metric_values = [metric_func(x) for x in group_cell_data_list]
            temp_metrics[metric_name] = metric_values
            global_metrics[metric_name].extend(metric_values)

            if metric_name not in display_metrics:
                continue
            annotation_dict[metric_name][x_offset] = metric_values
            sns.stripplot(x=x_offset, y=metric_values, edgecolor='none', 
                            size=1, jitter=0.1,
                            native_scale=True,
                            color="gray", ax=ax4[accummulated_metric_id])
            
            sns.boxplot(x=x_offset, y=metric_values, linewidth=0.8,
                        boxprops={"facecolor": "none", "edgecolor": group_color},
                        whiskerprops={"color": group_color},
                        capprops={"color": group_color},
                        medianprops={"color": group_color},
                        showfliers=False,
                        width=0.1, 
                        native_scale=True,
                        ax=ax4[accummulated_metric_id])
            accummulated_metric_id += 1
        plot_metric_correlational_matrix(current_metrics=temp_metrics, save_path=save_path + f"_{group_name}_correlation_matrix.png")
        
    for metric_id, metric_name in enumerate(display_metrics):
        ax4[metric_id].set_xticks([])
        ax4[metric_id].set_title(metric_name)
        ax4[metric_id].yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    ax4[0].set_yticks([0, 1, 2, ])
    ax4[0].set_ylim(0, 3)
    ax4[1].set_yticks([0, 0.3, 0.6])
    ax4[2].set_yticks([1, 2, ])
    ax4[2].set_ylim(1, 2.5)
    ax4[3].set_yticks([0, 0.2, 0.4])
    ax4[3].set_ylim(0, 0.5)
    ax4[4].set_yticks([0, 0.5])
    ax4[4].set_ylim(-0.5, 1)

    
    plot_metric_correlational_matrix(current_metrics=global_metrics, save_path=save_path + "_ALL_correlation_matrix.png")
    default_exit_save(fig4, save_path + "_variability_metrics.png", _transparent=True, _no_close=True)
    for metric_id, metric_name in enumerate(display_metrics):        
        basic_ttest.stats_anova(annotation_dict[metric_name], ax4[metric_id])
    default_exit_save(fig4, save_path + "_variability_metrics_annotated.png", _transparent=True)

    
    # n_bars = 2
    # fig5, ax5 = plt.subplots(2, n_bars, figsize=(2.2, 1), constrained_layout=True)
    # for ax in ax5.flatten():
    #     ax.spines[["top", "right"]].set_visible(False)
    #     ax.tick_params(length=1, pad=1, which='both')


    # def top_n_nearest_neighbor(x, t=None):
    #     x = np.asarray(x)
    #     t = np.arange(len(x)) if t is None else np.asarray(t)
    #     max_n = len(x)
    #     if len(x) != len(t):
    #         raise ValueError("x and t must have the same length")

    #     # Rank observations by response magnitude (highest first)
    #     ranked = np.argsort(x)[::-1]

    #     results = []
    #     for n in range(2, max_n + 1):
    #         selected_t = np.sort(t[ranked[:n]])

    #         # Distance to nearest selected neighbor
    #         gaps = np.diff(selected_t)
    #         nearest = np.minimum(
    #             np.r_[np.inf, gaps],
    #             np.r_[gaps, np.inf]
    #         )

    #         results.append(np.mean(nearest))

    #     return np.arange(2, max_n + 1), np.array(results)

    # for ax, cell_indices, group_name, group_color in zip(
    #     [ax5[:, 0], ax5[:, 1]],
    #     [CR_pos_cells, CR_neg_cells],
    #     ["CR+", "CR-"],
    #     [CR_POS_COLOR, CR_NEG_COLOR]
    # ):
        
    #     group_cell_data_list = [all_cs_node[int(cell_idx)].info["trial_level_corr"]["fluo"] for cell_idx in cell_indices]
    #     for cell_id, single_cell_data in enumerate(group_cell_data_list):
    #         top_ks, calculated_nearest_gaps = top_n_nearest_neighbor(single_cell_data)
    #         # ax[cell_id].plot(top_ks, calculated_nearest_gaps, color=f"C{cell_id%10}", lw=1)
    #         shuffle_nearest_gaps = np.mean([top_n_nearest_neighbor(np.random.permutation(single_cell_data))[1] 
    #                                         for _ in range(100)], axis=0)
    #         # ax[cell_id].plot(top_ks, shuffle_nearest_gaps, color='gray', lw=1, ls='--')
    #         ax[0].plot(top_ks/len(single_cell_data), calculated_nearest_gaps/shuffle_nearest_gaps, 
    #                             color=f"C{cell_id%10}", lw=1)
    #     ax[0].axhline(1, color='gray', lw=1, ls='--')
    # plt.show()
    # default_exit_save(fig5, save_path + "_autocorrelation.png", _transparent=True)
        

def visualize_single_cell_state_dependent(dataset: DataSet,

    _element_trial_level: str = "trial",
    _alignment_style: str = "Aligned2Trial",
    plot_small_pic: bool = False,):
    
    alignment_events = ALL_ALIGNMENT_STYLE[_alignment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    behav_range_rolling = [(range_start, range_start+2) for range_start in np.linspace(-4, 2, 100)]
    for cs_node, cs_subtree in dataset.select_subtree("cellsession"):
        if cs_node.fluorescence is None:
            continue

        print(f"Processing {cs_node.session_id}")

        all_puff_trials = sync_nodes(cs_subtree.select("trial", _self=lambda x: x.info.get("trial_type") == "PuffOnly"),
                                    alignment_events, plot_manual=plot_manual_fluo)
        all_trial_fluo = [AVERAGE_VALUE(node.data.fluorescence.df_f0.squeeze(0), (0, 1)) for node in all_puff_trials]
        all_trial_loco = [np.array([AVERAGE_VALUE(node.data.locomotion, behav_range) 
                                    for behav_range in behav_range_rolling]).reshape(-1) for node in all_puff_trials]
        all_trial_whisker = [np.array([AVERAGE_VALUE(node.data.whisker, behav_range) 
                                       for behav_range in behav_range_rolling]).reshape(-1) for node in all_puff_trials]


        if plot_small_pic:
            save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
                                f"state_dependent_correlation_{cs_node.session_id}_cell_{cs_node.cell_id}")
            visualize_single_cell_behavior_summary(
                cs_node=cs_node, all_puff_trials=all_puff_trials,
                all_fluos=all_trial_fluo, all_locos=all_trial_loco, all_whiskers=all_trial_whisker, 
                save_path=save_path)

        cs_node.info["trial_level_corr"] = {
            "fluo": np.array(all_trial_fluo),
            "loco": np.array(all_trial_loco),
            "whisker": np.array(all_trial_whisker),
        }

    # save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
    #                     f"state_dependent_correlation_summary_all")
    # visualize_all_cell_behavior_summary(dataset, save_path)
    save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
                        f"state_dependent_correlation_summary_day3")
    visualize_all_cell_behavior_summary(dataset.subset(_self=lambda x: x.info.get("day_id") == 3), save_path)
    # save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
    #                     f"state_dependent_correlation_summary_day4")
    # visualize_all_cell_behavior_summary(dataset.subset(_self=lambda x: x.info.get("day_id") == 4), save_path)


def quick_short_compare():
    cv_ephys = np.load(r"C:\Users\maxyc\PycharmProjects\Ratatouille\cuisine\PassivePuff_JuxtaCellular_FromJS_202509\Variability_of_ephysiology_SST Only\DeltaSpike_COHORTCLUSTER_ephys_cv.npy")
    cv_calcium = np.load(r"C:\Users\maxyc\PycharmProjects\Ratatouille\cuisine\PassivePuff_HighFreqImaging\HighFreqImaging_Combine_ROIs\trial_level_corr\state_dependent_correlation_summary_day3_calcium_cv.npy")

    cv_ephys = cv_ephys[~np.isnan(cv_ephys)]
    cv_calcium = cv_calcium[~np.isnan(cv_calcium)]
    fig1, ax1 = plt.subplots(1, 1, figsize=(1, 0.8), constrained_layout=True)
    ax1.tick_params(length=1, pad=1, which='both')
    ax1.spines[["top", "right", "left"]].set_visible(False)
    
    sns.kdeplot(cv_ephys, ax=ax1, fill=False, color='k', alpha=0.9, lw=1, log_scale=True)
    sns.kdeplot(cv_calcium, ax=ax1, fill=False, color='k', alpha=0.9, lw=1, log_scale=True, ls='--')

    ax1.scatter(cv_ephys, np.zeros_like(cv_ephys) + 0.1, s=7, facecolor='k', edgecolor='none', alpha=0.5)
    ax1.scatter(cv_calcium, np.zeros_like(cv_calcium) + .2, s=5, facecolor='none', edgecolor='k', alpha=0.5, lw=0.5)
    
    ax1.invert_xaxis()
    ax1.set_xlim(20, 0.05)
    ax1.set_xticks([0.2, 1, 5], ["1/5", "1", "5"])
    # ax1.set_xticks([], minor=True)
    
    # ax2.set_xlabel("CV", labelpad=1)
    ax1.set_yticks([])
    ax1.set_ylabel("")
    default_exit_save(fig1, "./cvs_distribution_compare.png", _transparent=True)


    ks_test_result = scipy.stats.ks_2samp(cv_ephys, cv_calcium)
    print(f"KS test result: {ks_test_result}")

if __name__ == "__main__":
    quick_short_compare()
