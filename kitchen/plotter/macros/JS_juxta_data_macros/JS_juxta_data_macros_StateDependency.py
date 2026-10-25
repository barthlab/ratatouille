
import os.path as path
import os
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import logging
from collections import defaultdict


from kitchen.operator.grouping import AdvancedTimeSeries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_Decomposition_Weights
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SPVtypes import get_grand_cluster_result
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS, COHORT_BRIGHTER_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.stats_tests.basic_ttest import add_loglog_regression
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.potential import SPIKE_ANNOTATION_EARLY_WINDOW, GCaMP_kernel
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)

import numpy as np





plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6

def plot_variability(cell_data, cell_group, 
                     full_group_color_dict,  
                     save_path, ax1_mean_range = None, _total_kde=False, log_scale=True, ax3_ylim=None):
    # cell_data: list of list 
    # cell_group: list of group name
    existed_groups = set(cell_group)
    group_color_dict = {group_name: full_group_color_dict[group_name] 
                        for group_name in full_group_color_dict.keys() 
                        if group_name in existed_groups}
    group_zorder = {"SST": 2, "PYR": 1, "PV": 0}

    fig1, ax1 = plt.subplots(1, 1, figsize=(1.4, 1.4), constrained_layout=True)
    ax1.tick_params(length=1, pad=1, which='both')
    ax1.spines[["top", "right"]].set_visible(False)

    group_cvs = defaultdict(list)
    for single_cell_data, single_cell_group in zip(cell_data, cell_group):
        data_avg = np.nanmean(single_cell_data)
        data_var = np.nanstd(single_cell_data)
        ax1.scatter(data_avg, data_var, c=group_color_dict[single_cell_group], s=8, edgecolors='white', lw=0.3,
                    zorder=group_zorder.get(single_cell_group, 0)
                    )

        group_cvs[single_cell_group].append(data_var / data_avg)
    if log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    if ax1_mean_range:
        xs = np.linspace(*ax1_mean_range, 100)
        ax1.plot(xs, xs, c="gray", lw=0.5, ls="--", zorder=-20)
        ax1.plot(xs, 4*xs, c="gray", lw=0.5, ls="--", zorder=-20)
        ax1.plot(xs, 0.25*xs, c="gray", lw=0.5, ls="--", zorder=-20)
        ax1.set_xlim(*ax1_mean_range)
        ax1.set_ylim(*ax1_mean_range)
        ax1.set_yticks([0.01, 1, 100])
        # for tick in ax1.get_yticklabels():
        #     tick.set_rotation(90)
        #     tick.set_horizontalalignment("right")
        #     tick.set_verticalalignment("center")

        all_sst_data_tuple = np.array([(np.nanmean(single_cell_data), np.nanstd(single_cell_data)) 
                              for single_cell_data, single_cell_group in zip(cell_data, cell_group) 
                              if single_cell_group.startswith("SST") and np.nanmean(single_cell_data) > 0])
        add_loglog_regression(ax1, all_sst_data_tuple[:, 0], all_sst_data_tuple[:, 1], theme_color=COHORT_COLORS["SST_JUX"])

        # all_pv_data_tuple = np.array([(np.nanmean(single_cell_data), np.nanstd(single_cell_data)) 
        #                                 for single_cell_data, single_cell_group in zip(cell_data, cell_group) 
        #                                 if single_cell_group.startswith("PV") and np.nanmean(single_cell_data) > 0])
        # add_loglog_regression(ax1, all_pv_data_tuple[:, 0], all_pv_data_tuple[:, 1], theme_color=COHORT_COLORS["PV_JUX"], offsets=0.1)

        # all_pyr_data_tuple = np.array([(np.nanmean(single_cell_data), np.nanstd(single_cell_data)) 
        #                                 for single_cell_data, single_cell_group in zip(cell_data, cell_group) 
        #                                 if single_cell_group.startswith("PYR") and np.nanmean(single_cell_data) > 0])
        # add_loglog_regression(ax1, all_pyr_data_tuple[:, 0], all_pyr_data_tuple[:, 1], theme_color=COHORT_COLORS["PYR_JUX"], offsets=0.2)

    ax1.set_aspect('equal', adjustable='box') 
    ax1.set_xlabel(r"$\Delta$" + " FR mean (Hz)", labelpad=1)
    ax1.set_ylabel(r"$\Delta$" + " FR SD (Hz)", labelpad=1)
    
    default_exit_save(fig1, save_path + "_mean_variance.png", _transparent=True)

    fig2, ax2 = plt.subplots(1, 1, figsize=(0.7, 0.4), constrained_layout=True)
    ax2.tick_params(length=1, pad=1, which='both')
    ax2.spines[["top", "right", "left"]].set_visible(False)

    for group_idx, (group_name, group_color) in enumerate(group_color_dict.items()):
        group_cvs_array = np.array(group_cvs[group_name])
        sns.kdeplot(group_cvs_array, ax=ax2, fill=True, color=group_color, alpha=0.8, lw=0, 
                    log_scale=True, zorder=-group_idx
                    )
    if _total_kde:
        all_cvs = np.concatenate([np.array(group_cvs[group_name]) for group_name in group_color_dict.keys()])
        sns.kdeplot(all_cvs, ax=ax2, fill=False, color="black", lw=0.5, ls="--", log_scale=True)
    ax2.invert_xaxis()
    ax2.set_xlim(10, 0.1)
    ax2.set_xticks([0.2, 1, 5], ["1/5", "1", "5"])
    # ax2.set_xticks([], minor=True)
    
    # ax2.set_xlabel("CV", labelpad=1)
    ax2.set_yticks([])
    ax2.set_ylabel("")
    default_exit_save(fig2, save_path + "_cv_distribution.png", _transparent=True)

    np.save(save_path + "_ephys_cv.npy", all_cvs)
    # ks_test_result = scipy.stats.ks_2samp(CR_neg_cv, CR_pos_cv)
    # print(f"KS test result: {ks_test_result}")


    fig3, ax3s = plt.subplots(2, 1, figsize=(3. if "SST Only" in save_path else 4., 2.5), constrained_layout=True)
    for ax3 in ax3s:
        ax3.spines[["top", "right"]].set_visible(False)
        ax3.tick_params(length=1, pad=1, which='both')

    cell_group_indices = {
        group_name: np.argwhere(np.array(cell_group) == group_name).flatten()
        for group_name in group_color_dict.keys()
    }
    group_offset = np.cumsum([0] + [len(indices) for indices in list(cell_group_indices.values())[:-1]])

    for x_offset, cell_indices, group_name in zip(
        group_offset,
        cell_group_indices.values(),
        list(cell_group_indices.keys()),        
    ):
        group_color = group_color_dict[group_name]

        cell_median_data = [np.median(cell_data[int(cell_idx)]) for cell_idx in cell_indices]
        sorted_order = np.argsort(cell_median_data)[::-1]
        cell_indices = cell_indices[sorted_order].flatten()

        for cell_rel_idx, cell_idx in enumerate(cell_indices):
            cell_single_data = cell_data[int(cell_idx)]
            # single_cell_color = cmap(norm(cell_node.info.get("red_cohen_d", np.nan)))
            sns.stripplot(x=x_offset + cell_rel_idx, y=cell_single_data, edgecolor='none', 
                          size=1, jitter=0.3,
                          color="gray", ax=ax3s[0])
            sns.boxplot(x=x_offset + cell_rel_idx, y=cell_single_data, linewidth=0.8,
                        boxprops={"facecolor": "none", "edgecolor": group_color},
                        whiskerprops={"color": group_color},
                        capprops={"color": group_color},
                        medianprops={"color": group_color},
                        showfliers=False,
                        width=0.6, 
                        ax=ax3s[0])
            sns.stripplot(x=x_offset + cell_rel_idx, y=np.diff(cell_single_data), edgecolor='none', 
                          size=1, jitter=0.3,
                          color="gray", ax=ax3s[1])
            sns.boxplot(x=x_offset + cell_rel_idx, y=np.diff(cell_single_data), linewidth=0.8,
                        boxprops={"facecolor": "none", "edgecolor": group_color},
                        whiskerprops={"color": group_color},
                        capprops={"color": group_color},
                        medianprops={"color": group_color},
                        showfliers=False,
                        width=0.6, 
                        ax=ax3s[1])

    cell_group_center = (
        group_offset + 
        np.cumsum([len(indices) for indices in cell_group_indices.values()])
    ) / 2 - 0.5
    for ax3 in ax3s:
        ax3.set_xticks(cell_group_center, list(group_color_dict.keys()))
        for label, color in zip(ax3.get_xticklabels(), group_color_dict.values()):
            label.set_color(color)
        if ax3_ylim is not None:
            ax3.set_ylim(ax3_ylim)
            ax3.axhline(0, color='gray', lw=0.5, zorder=-1)

        for group_boundary in group_offset[1:]:
            ax3.axvline(group_boundary - 0.5, color='gray', lw=0.5, ls='--', zorder=-1)
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
    n_bars = len(cell_group_indices)
    for x_offset, cell_indices, group_name in zip(
        np.linspace(-0.15, 0.15, n_bars),
        cell_group_indices.values(),
        list(cell_group_indices.keys()),        
    ):
        
        group_color = group_color_dict[group_name]
        group_cell_data_list = [cell_data[int(cell_idx)] for cell_idx in cell_indices]

        
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
                            size=1, jitter=0.1 if n_bars <= 3 else 0.05,
                            native_scale=True,
                            color="gray", ax=ax4[accummulated_metric_id])
            
            sns.boxplot(x=x_offset, y=metric_values, linewidth=0.8 if n_bars <=3 else 0.4,
                        boxprops={"facecolor": "none", "edgecolor": group_color},
                        whiskerprops={"color": group_color},
                        capprops={"color": group_color},
                        medianprops={"color": group_color},
                        showfliers=False,
                        width=0.1 if n_bars <= 3 else 0.05, 
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
    ax4[1].set_yticks([0, 4, 8, 12])
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
    
        
    # n_bars = len(cell_group_indices)
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


 
    # normalized_lags = np.linspace(0, 20, 100)
    # for ax, cell_indices, group_name in zip(
    #     [ax5[:, 0], ax5[:, 1]],
    #     cell_group_indices.values(),
    #     list(cell_group_indices.keys()),        
    # ):
        
    #     group_color = group_color_dict[group_name]
    #     group_cell_data_list = [cell_data[int(cell_idx)] for cell_idx in cell_indices]

    #     for cell_id, single_cell_data in enumerate(group_cell_data_list):
    #         top_ks, calculated_nearest_gaps = top_n_nearest_neighbor(single_cell_data)
    #         # ax[cell_id].plot(top_ks, calculated_nearest_gaps, color=f"C{cell_id%10}", lw=1)
    #         shuffle_nearest_gaps = np.mean([top_n_nearest_neighbor(np.random.permutation(single_cell_data))[1] 
    #                                         for _ in range(100)], axis=0)
    #         # ax[cell_id].plot(top_ks, shuffle_nearest_gaps, color='gray', lw=1, ls='--')
    #         ax[0].plot(top_ks/len(single_cell_data), calculated_nearest_gaps/shuffle_nearest_gaps, 
    #                          color=f"C{cell_id%10}", lw=1)
    #     ax[0].axhline(1, color='gray', lw=1, ls='--')
    #     # for cell_id in range(len(group_cell_data_list), 20):
    #     #     ax[cell_id].remove()
    #     # max_len = max(len(acf) for acf in aligned_acfs)
    #     # pad_autocorr = [np.pad(acf, (0, max_len - len(acf)), 'constant', constant_values=np.nan) 
    #     #                 for acf in aligned_acfs]
    #     # avg_autocorr = np.nanmean(pad_autocorr, axis=0)
    #     # ax[1].plot(normalized_lags, avg_autocorr, color=group_color, lw=1)
    # plt.show()
    # default_exit_save(fig5, save_path + "_autocorrelation.png", _transparent=True)
    
    
def Visualize_Variability_of_ephysiology(list_of_cell: list[dict]):
    node_cluster_name, _ = get_grand_cluster_result()
    print(node_cluster_name)

    for select_group_ids, select_group_title in zip([
        # ("SST",  ),
        # ("SST", "PV", "PYR"),
        # ("SST3", "SST4", "SST5", "PV1", "PV2", "PYR5"),
        ("SST4", "SST5", ),
    ], [
        # "SST Only", 
        # "Classes",
        "Clusters",
        ]):

        LFP_data, early_spike_num, sustained_spike_num, spont_spike_num = [], [], [], []
        total_spike, delta_spike, delta_spike_ratio = [], [], []
        delta_early, delta_sustained = [], []
        cohort_name_list, cluster_name_list = [], []
        cohort_cluster_name_list, cohort_cluster_color_dict = [], {}
        for data_dict in list_of_cell:
            # cohort_cluster_id = data_dict["cohort_name"].split("_")[0]
            cohort_cluster_id = data_dict["cohort_name"].split("_")[0] + str(int(node_cluster_name[data_dict["node_name"]]))
            if cohort_cluster_id not in select_group_ids:
                continue

            LFP_data.append(-np.array(data_dict["LFP peak [mV]"]))

            single_cell_early_spike_ts = data_dict["Early Spike [ms]"]
            early_spike_num.append([len(early_spike_t) for early_spike_t in single_cell_early_spike_ts])

            single_cell_sustained_spike_ts = data_dict["Sustained Spike [ms]"]
            sustained_spike_num.append([len(sustained_spike_t) for sustained_spike_t in single_cell_sustained_spike_ts])

            spont_spike_num.append(data_dict["PreStim Spont. FR [Hz]"])

            total_spike.append([len(early_spike_t) + len(sustained_spike_t)
                                for early_spike_t, sustained_spike_t in zip(single_cell_early_spike_ts, single_cell_sustained_spike_ts)])
            
            delta_spike.append([(len(early_spike_t) + len(sustained_spike_t)) / 0.5 - spont_fr
                                for early_spike_t, sustained_spike_t, spont_fr in 
                                zip(single_cell_early_spike_ts, single_cell_sustained_spike_ts, data_dict["PreStim Spont. FR [Hz]"])])
            
            delta_spike_ratio.append([((len(early_spike_t) + len(sustained_spike_t)) / 0.5 - spont_fr) / (spont_fr + 1)
                                for early_spike_t, sustained_spike_t, spont_fr in 
                                zip(single_cell_early_spike_ts, single_cell_sustained_spike_ts, data_dict["PreStim Spont. FR [Hz]"])])
            
            delta_early.append([(len(early_spike_t) / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0]) - spont_fr)
                                for early_spike_t, spont_fr in 
                                zip(single_cell_early_spike_ts, data_dict["PreStim Spont. FR [Hz]"])])
            
            delta_sustained.append([(len(sustained_spike_t) / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1]) - spont_fr)
                                for sustained_spike_t, spont_fr in 
                                zip(single_cell_sustained_spike_ts, data_dict["PreStim Spont. FR [Hz]"])])


            
            cohort_name_list.append(data_dict["cohort_name"])
            cluster_name_list.append(node_cluster_name[data_dict["node_name"]])
            cohort_cluster_name_list.append(cohort_cluster_id)
            cohort_cluster_color_dict[cohort_cluster_name_list[-1]] = CLUSTER_COLORS[node_cluster_name[data_dict["node_name"]]]
            # cohort_cluster_color_dict[cohort_cluster_name_list[-1]] = COHORT_BRIGHTER_COLORS[data_dict["cohort_name"]]

        cohort_cluster_color_dict = {name: cohort_cluster_color_dict[name] for name in select_group_ids}
        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"LFP")
        # # plot_variability(LFP_data, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # # plot_variability(LFP_data, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        # plot_variability(LFP_data, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",)

        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"EarlySpike")
        # # plot_variability(early_spike_num, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # # plot_variability(early_spike_num, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        # p1=plot_variability(early_spike_num, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
                        # ax1_mean_range=(0.1, 9), _total_kde=True)

        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"SustainedSpike")
        # # plot_variability(sustained_spike_num, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # # plot_variability(sustained_spike_num, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        # p2=plot_variability(sustained_spike_num, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
                        # ax1_mean_range=(0.1, 50), _total_kde=True)

        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"SpontSpike")
        # # plot_variability(spont_spike_num, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # # plot_variability(spont_spike_num, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        # p3=plot_variability(spont_spike_num, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
        #                 ax1_mean_range=(0.1, 50), _total_kde=True)

        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"TotalSpike")
        # # plot_variability(total_spike, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # # plot_variability(total_spike, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        # p4=plot_variability(total_spike, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
                        # ax1_mean_range=(0.1, 60), _total_kde=True)

        save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"DeltaSpike")
        # plot_variability(delta_spike, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # plot_variability(delta_spike, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        p5=plot_variability(delta_spike, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
                        ax1_mean_range=(0.01, 100), _total_kde=True, ax3_ylim=(-15, 89) if select_group_title == "SST Only" else (-55, 100))

        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"DeltaEarlySpike")
        # p6=plot_variability(delta_early, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
        #                 ax1_mean_range=(2, 150), _total_kde=True, ax3_ylim=(-30, 150) if select_group_title == "SST Only" else (-40, 200))

        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"DeltaSustained")
        # p7=plot_variability(delta_sustained, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
        #                 ax1_mean_range=(0.1, 80), _total_kde=True, ax3_ylim=(-15, 89) if select_group_title == "SST Only" else (-55, 100))
        
        # save_path = path.join(get_saving_path(), f"Variability_of_ephysiology_{select_group_title}",  f"DeltaSpikeRatio")
        # # plot_variability(delta_spike_ratio, cohort_name_list, COHORT_COLORS, save_path + "_COHORT",)
        # # plot_variability(delta_spike_ratio, cluster_name_list, CLUSTER_COLORS, save_path + "_CLUSTER",)
        # plot_variability(delta_spike_ratio, cohort_cluster_name_list, cohort_cluster_color_dict, save_path + "_COHORTCLUSTER",
        #                  ax1_mean_range=(-1, 39),)

        exit()
        
    
        
