from collections import defaultdict
import logging
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt     
import seaborn as sns
import matplotlib.patheffects as pe
import numpy as np
import os.path as path
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, SparsePCA, NMF, FastICA

from kitchen.calculator import calculate_physiology
from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import grouping_timeseries
from kitchen.plotter import color_scheme
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_all_physiology_features, get_weight_tuple_PSTH
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SPVtypes import get_grand_cluster_result
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.utils.ordering_kit import linkage_order
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_BRIGHTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_PSTH_mean, get_all_cellsession_Waveform_mean, get_saving_path
from kitchen.settings.potential import NARROW_SPIKE_RANGE_FOR_VISUALIZATION, SPIKE_ANNOTATION_EARLY_WINDOW
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)


def temporal_pattern_SVD(variant,
                       task_name: str="TemporalPattern", **kwargs):
    PSTH_weight_matrix, node_names, cohort_names = get_weight_tuple_PSTH(variant, **kwargs)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    n_cells, n_timepoints = PSTH_weight_matrix.shape
    logger.info(f"Performing stability test for SVD with on "
                f"PSTH data ({variant}) with {n_cells} cells and {n_timepoints} timepoints.")


    bins = np.arange(-1, 1.5 + kwargs["binsize"], kwargs["binsize"])
    bin_centers = (bins[:-1] + bins[1:]) / 2

    solver = TruncatedSVD(n_components=5, random_state=42)
    component_projection = solver.fit_transform(PSTH_weight_matrix)
    reconstruct_matrix = solver.inverse_transform(component_projection)

    node2label, node_order = get_grand_cluster_result()
    label_node_names = list(node2label.keys())
    node_to_row = {name: i for i, name in enumerate(node_names)}
    sorted_order = [node_to_row[label_node_names[i]] for i in node_order ]    


    desired_lim = (-0.3, 0.8)
    lim_tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_lim]

    desired_ticks = [0, 0.1, 0.5]
    # desired_ticks = [0, 0.5]
    tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_ticks]

    sorted_labels = np.array([
        node2label[node_names[i]] for i in sorted_order
    ])
    boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    n_labels = len(np.unique(sorted_labels))

    fig, axs = plt.subplots(1, 2, figsize=(6, 2.1), constrained_layout=True)
    # fig, axs = plt.subplots(1, 2, figsize=(2, 1.8), constrained_layout=True)

    for ax, matrix in zip(axs, [PSTH_weight_matrix, reconstruct_matrix]):
        sns.heatmap(matrix[sorted_order],
                            ax=ax, cbar=False,
                            cmap='YlOrBr', 
                            vmin=0, vmax=100,)
                            # cmap='Spectral', 
                            # vmin=-2, vmax=2,)
                            # norm=SymLogNorm(linthresh=1., vmin=0, vmax=100, linscale=0.1),)
                
        ax.set_xticks(tick_positions, desired_ticks, rotation=0, fontsize=8)  
        
        ax.set_xlim(lim_tick_positions[0], lim_tick_positions[-1])
        ax.set_xlabel("Time (s)", labelpad=1)
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
        ax.tick_params(length=1, pad=1, axis='y')
        # ax.tick_params(pad=1, length=1, axis='x')
        ax.tick_params(pad=1, axis='x')

        for boundary in boundaries:
            ax.axhline(boundary, color="black", linewidth=0.5)

        yticks = []
        for label_id in range(1, n_labels+1):
            label_position = np.argwhere(sorted_labels == label_id)[:, 0]
            yticks.append(label_position.mean() + 0.5)
        ax.set_yticks(yticks, [str(i) for i in range(1, n_labels+1)])
        for tick in ax.get_yticklabels():
            text = tick.get_text()
            tick.set_color(CLUSTER_COLORS[int(text)])
            tick.set_fontsize(8)
            tick.set_weight('bold')
            tick.set_rotation(0)

        ax2 = ax.twiny()
        ax2.set_xticks(tick_positions, [""]*len(tick_positions), rotation=0)  
        ax2.set_xlim(lim_tick_positions[0], lim_tick_positions[-1])


    save_path = path.join(get_saving_path(), task_name, 
                        f"RawVSReconstruct_{variant}_SVD5.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    timebin_correlation = np.corrcoef(numpy_kit.zscore(PSTH_weight_matrix, axis=1), rowvar=False)
    fig, ax = plt.subplots(1, 1, figsize=(4., 4.), constrained_layout=True)


    mask = np.tril(np.ones_like(timebin_correlation, dtype=bool), k=-1)
    sns.heatmap(timebin_correlation, ax=ax, cbar=False, cmap='coolwarm', 
                vmin=-0.5, vmax=0.5, mask=mask, linewidths=0, square=True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axis_off()
    ax.set_xticks([])  
    ax.set_yticks([])  

    for xt in tick_positions:
        ax.axvline(xt, color="black", linewidth=1)  
        ax.axhline(xt, color="black", linewidth=1)  

    ax.set_xlim(lim_tick_positions[0], lim_tick_positions[-1])
    ax.set_ylim(lim_tick_positions[-1], lim_tick_positions[0],)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
    ax.tick_params(length=0, pad=1)

    save_path = path.join(get_saving_path(), task_name, 
                        f"TimebinCorrelation_{variant}_SVD5.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)




def temporal_pattern_Early_spike(task_name: str="TemporalPattern",):

    
    from itertools import groupby
    import matplotlib.pyplot as plt    
    from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_trial_wise_spike_timing_features
    from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_LFP_mean

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    
    global_LFP = []

    def _good_cell(cell_spike_timing):
        num_trial = len(cell_spike_timing)
        total_spike_time = sum([len(trial) for trial in cell_spike_timing])
        return num_trial >= 5 and total_spike_time >= 5

    global_y_offset = 0
    node2label, node_order = get_grand_cluster_result()
    label_node_names = list(node2label.keys())
    node_to_spikes = {}
    annotate_data={}
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), constrained_layout=True)
    fig_dist, ax_dist = plt.subplots(1, 1, figsize=(2.4, 1.3), constrained_layout=True)
    for cohort_name_group in [["SST_JUX", "SST_WC"], ["PV_JUX"], ["PYR_JUX", "PYR_WC"]]:
        for cohort_name in cohort_name_group:
            trial_wise_spike_timing = get_trial_wise_spike_timing_features(cohort_name)
            all_trial_type = trial_wise_spike_timing["all_trial_type"]
            indices_500ms = np.where(all_trial_type == "500msPuff")[0]
            for i in indices_500ms:
                cell_data = trial_wise_spike_timing["all_spikes"][i]
                if _good_cell(cell_data):
                    node_name = trial_wise_spike_timing["all_node_name"][i]
                    node_to_spikes[node_name] = (cell_data, cohort_name)

            if "WC" not in cohort_name:
                LFP_t, all_LFP = get_all_cellsession_LFP_mean(cohort_name)
                global_LFP.append(TimeSeries(t=LFP_t, v=np.mean(all_LFP, axis=0)))

    ordered_node_names = [label_node_names[i] for i in node_order
                          if label_node_names[i] in node_to_spikes]
    for cluster_id, cluster_nodes in groupby(ordered_node_names, key=node2label.__getitem__):
        cluster_y_start = global_y_offset
        median_early_spike_500ms = []

        celltype_specific_data = defaultdict(list)
        for node_name in cluster_nodes:
            cell_data, cohort_name = node_to_spikes[node_name]
            ax.eventplot(np.concatenate(cell_data), lineoffsets=global_y_offset,
                             orientation='horizontal', linelengths=1, linewidths=0.3,
                             color=COHORT_COLORS[cohort_name.replace("_WC", "_JUX")])
            global_y_offset -= 1
            median_early_spike_500ms.append(np.median(np.concatenate([trial[np.where((trial < 40) & (trial > 0))] 
                                                              for trial in cell_data])))
            # celltype_specific_data[cohort_name.replace("_WC", "_JUX")].append(
                # np.median(np.concatenate([trial[np.where((trial < 100) & (trial > 0))] 
                #                                               for trial in cell_data])))
            celltype_specific_data[cohort_name.replace("_WC", "_JUX")].append(median_early_spike_500ms[-1])

        print(f"Cluster {cluster_id} celltype specific data:")
        for celltype, values in celltype_specific_data.items():
            print(f"  {celltype}: {np.mean(values):.3f} +/- {np.std(values):.3f}")
        ax.fill_betweenx(y=[global_y_offset+0.5, cluster_y_start+0.5],
                         x1=0, x2=0.5 * 1000,
                        alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10,)
        global_y_offset -= 2

        median_early_spike_500ms = np.array(median_early_spike_500ms)        
        median_early_spike_500ms = median_early_spike_500ms[median_early_spike_500ms>16]  # filter out artifacts        
        median_early_spike_500ms = median_early_spike_500ms[~np.isnan(median_early_spike_500ms)]        
        if cluster_id == 3:
            continue
        ax_dist.scatter(np.sort(median_early_spike_500ms), np.arange(1, 1 + len(median_early_spike_500ms))/len(median_early_spike_500ms),
                                facecolor=CLUSTER_COLORS[cluster_id], edgecolor='black',
                                s=7, lw=0.5, alpha=0.9, zorder=10)
        ax_dist.plot(np.sort(median_early_spike_500ms), np.arange(1, 1 + len(median_early_spike_500ms))/len(median_early_spike_500ms),
                        color=CLUSTER_COLORS[cluster_id], lw=1, zorder=-1, alpha=0.9)
        print(f"cluster {cluster_id}: Median early spike times for all cells: {np.mean(median_early_spike_500ms):.3f}+/-{np.std(median_early_spike_500ms):.3f}")
        annotate_data[f"Cluster_{cluster_id}"] = median_early_spike_500ms



    global_y_offset -= 5
    all_global_LFP = grouping_timeseries(global_LFP)
    ax.plot(all_global_LFP.t * 1000, all_global_LFP.v * 10 + global_y_offset, color="k", lw=1,)

    ax.tick_params(
        length=1,
        pad=1     
    )
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(-20, 80)
    ax.set_xticks([0, 20, 40, 60,])
    ax.axvline(0, color='gray', lw=0.5, ls="--", zorder=-1)
    ax.axvline(20, color='gray', lw=0.5, ls="--", zorder=-1)
    ax.set_xlabel("Time from puff onset (ms)", labelpad=1)



    save_path = path.join(get_saving_path(), task_name, f"EarlySpikeDistribution_sorted_by_cluster.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    ax_dist.tick_params(
        length=1,
        pad=1     
    )
    ax_dist.spines[['right', 'top']].set_visible(False)
    ax_dist.set_xticks([18, 20, 22, 24, 26])
    ax_dist.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax_dist.set_xlabel("Median early spike time (ms)", labelpad=1)
    ax_dist.set_ylabel("Cumulative distribution", labelpad=1)

    save_path = path.join(get_saving_path(), task_name, f"EarlySpikeDistribution_CDF_by_cluster.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig_dist.savefig(save_path, dpi=500, transparent=True)
    logger.info("Plot saved to " + save_path)

    
    basic_ttest.stats_pairwise_ks(annotate_data, ax=ax_dist)
    for cluster_id, y_values in annotate_data.items():
        ax_dist.text(np.mean(y_values), 0.2, f"{np.mean(y_values):.2f}±{np.std(y_values):.2f}", 
                     ha='center', va='bottom', fontsize=3, rotation=45, zorder=2000)
    save_path = path.join(get_saving_path(), task_name, f"EarlySpikeDistribution_CDF_by_cluster_annotated.png")
    fig_dist.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig_dist)
    logger.info("Plot saved to " + save_path)



def temporal_pattern_early_vs_sustained(task_name: str="TemporalPattern",):
    sst_wc_physiology = get_all_physiology_features("SST_WC")
    sst_jux_physiology = get_all_physiology_features("SST_JUX")
    pv_jux_physiology = get_all_physiology_features("PV_JUX")
    pyr_jux_physiology = get_all_physiology_features("PYR_JUX")
    pyr_wc_physiology = get_all_physiology_features("PYR_WC")

    combined_physiology = pd.concat(
        [sst_wc_physiology, 
         sst_jux_physiology, 
         pv_jux_physiology, 
         pyr_jux_physiology,
         pyr_wc_physiology], ignore_index=True)

    node2label, node_order = get_grand_cluster_result()
    label_node_names = list(node2label.keys())
    physiology_by_node = combined_physiology.set_index("node_name")
    ordered_node_names = [label_node_names[i] for i in node_order
                          if label_node_names[i] in physiology_by_node.index]
    combined_physiology = physiology_by_node.loc[ordered_node_names].copy()
    combined_physiology["cluster"] = [node2label[name] for name in ordered_node_names]
    cluster_order = list(dict.fromkeys(combined_physiology["cluster"]))

    
    import matplotlib.pyplot as plt    

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
       
    annotate_data = {}
    fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.6), constrained_layout=True)
    x_offset = 0.5
    ax.tick_params(
        length=1,
        pad=1     
    )
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_yscale('symlog', linthresh=1, linscale=0.1)
    ax.set_xlim(-0.3, max(0, len(cluster_order) - 1) + 0.3 + x_offset)
    ax.set_xticks(np.arange(len(cluster_order)) + x_offset / 2,
                  [str(cluster_id) for cluster_id in cluster_order])
    for label, cluster_id in zip(ax.get_xticklabels(), cluster_order):
        label.set_color(CLUSTER_COLORS[cluster_id])
        label.set_fontweight("bold")
        label.set_fontsize(8)
    ax.set_xlabel("Cluster", labelpad=1)
    ax.set_ylabel("Spike rate (Hz)", labelpad=1)
    ax.set_yticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])

    for cluster_x, cluster_id in enumerate(cluster_order):
        cluster_data = combined_physiology[combined_physiology["cluster"] == cluster_id]
        early_spike_FR = cluster_data["Early Spike Num\nAvg"].values / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0])
        sustained_spike_FR = cluster_data["Sustained Spike Num\nAvg"].values / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1])
        annotate_data[(cluster_x, cluster_x + x_offset)] = (early_spike_FR, sustained_spike_FR)

        ax.scatter(cluster_x - 0.2, np.nanmean(early_spike_FR),
                    facecolor=CLUSTER_COLORS[cluster_id], edgecolor="black", lw=0.5, s=8, zorder=1, alpha=0.9)
        ax.scatter(cluster_x + x_offset + 0.2, np.nanmean(sustained_spike_FR),
                    facecolor=CLUSTER_COLORS[cluster_id], edgecolor="black", lw=0.5, s=8, zorder=1, alpha=0.9)

        
        for zorder_id, cohort_name in enumerate(cluster_data["cohort_name"].unique()):
            cohort_mask = cluster_data["cohort_name"].values == cohort_name
            cohort_early = early_spike_FR[cohort_mask]
            cohort_sustained = sustained_spike_FR[cohort_mask]
            color = COHORT_BRIGHTER_COLORS[cohort_name.replace("WC", "JUX")]
            ax.scatter([cluster_x] * len(cohort_early), cohort_early,
                       edgecolor=color, facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
            ax.scatter([cluster_x + x_offset] * len(cohort_sustained), cohort_sustained,
                       edgecolor=color, facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)

        for node_early, node_sustained, cohort_name in zip(early_spike_FR, sustained_spike_FR, cluster_data["cohort_name"]):
            ax.plot([cluster_x, cluster_x + x_offset], [node_early, node_sustained],
                    color=COHORT_BRIGHTER_COLORS[cohort_name.replace("WC", "JUX")], lw=0.5, alpha=0.8, zorder=-1)
                
    save_path = path.join(get_saving_path(), task_name, f"EarlyVsSustained_Compare.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    logger.info("Plot saved to " + save_path)

    
    basic_ttest.stats_pairwise_paired_ttest(annotate_data, ax)
                
    save_path = path.join(get_saving_path(), task_name, f"EarlyVsSustained_Compare_annotated.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def temporal_pattern_variability_visualize(variant='raw',
                          task_name: str="TemporalPattern", **kwargs):
    PSTH_weight_matrix, node_names, cohort_names = get_weight_tuple_PSTH(variant, **kwargs)

    bins = np.arange(-1, 1.5 + kwargs["binsize"], kwargs["binsize"])
    bin_centers = (bins[:-1] + bins[1:]) / 2


    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    # n_sep = 5

    n_cells, n_timepoints = PSTH_weight_matrix.shape
    logger.info(f"Performing stability test for SVD with on "
                f"PSTH data ({variant}) with {n_cells} cells and {n_timepoints} timepoints.")
    
    node2label, node_order = get_grand_cluster_result()
    label_node_names = list(node2label.keys())
    available_nodes = set(node_names)
    cluster_order = list(dict.fromkeys(
        node2label[label_node_names[i]] for i in node_order
        if label_node_names[i] in available_nodes
    ))
    node_labels = np.array([node2label.get(name) for name in node_names])

    
    fig, axs = plt.subplots(1, 2, figsize=(3.9, 0.9), constrained_layout=True)
    for ax in axs:
        ax.spines[['right', 'top',]].set_visible(False)
        ax.tick_params(length=1, pad=1, which='both')
    variances = []
    for cluster_id in cluster_order:
        sub_psth = PSTH_weight_matrix[node_labels == cluster_id]
        variance = np.var(sub_psth, axis=0, ddof=1)
        variances.append(variance)
    variances = np.array(variances)
    total_variance = np.sum(variances, axis=0)   
    global_variance = np.std(PSTH_weight_matrix, axis=0, ddof=1)
    percentages = np.divide(
        variances * 100,
        total_variance,
        out=np.zeros_like(variances),
        where=total_variance > 0
    )

    axs[0].plot(bin_centers, global_variance, color='black', lw=0.5, alpha=1)
    axs[0].set_yticks([0, 50, 100], ["0", "50", "100"])
    axs[0].set_ylabel("SD (Hz)", labelpad=1)
    axs[0].set_xticks([0, 0.5], ["0", "0.5"])
    axs[0].set_xlim(-0.5, 1)
    axs[0].set_xlabel("Time from puff onset (s)", labelpad=1)    
    axs[0].axvspan(0, 0.5, 
                alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
    # axs[0].set_yscale('symlog', linthresh=1, linscale=0.1)

    
    axs[1].stackplot(bin_centers, *percentages, 
                     linewidth=0.2, edgecolor=(0, 0, 0, 0.5), linestyle='-', 
                    alpha=1, colors=[CLUSTER_COLORS[cluster_id]
                                        for cluster_id in cluster_order])
    axs[1].axvline(0, color='black', lw=0.5, alpha=0.5, ls='--')
    axs[1].axvline(0.5, color='black', lw=0.5, alpha=0.5, ls='--')
    axs[1].set_xlim(-0.5, 1)
    axs[1].set_xticks([0, 0.5], ["0", "0.5"])
    axs[1].set_xlabel("Time from puff onset (s)", labelpad=1)
    axs[1].set_yticks([0, 50, 100], ["0", "50", "100"], rotation=90, ha='right', va='center')
    axs[1].set_ylim(0, 100)    
    axs[1].set_ylabel("Variance share (%)", labelpad=1)
    axs[1].spines[['right', 'top', 'bottom', 'left']].set_visible(True)

    save_path = path.join(get_saving_path(), task_name, 
                        f"VarianceDist_{variant}_SVD.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)
