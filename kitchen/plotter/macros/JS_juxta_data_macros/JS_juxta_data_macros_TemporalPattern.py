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
    sorted_order = [node_to_row[label_node_names[i]] for i in node_order]    


    desired_lim = (-0.3, 0.8)
    lim_tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_lim]

    desired_ticks = [0, 0.1, 0.5]
    tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_ticks]

    sorted_labels = np.array([
        node2label[node_names[i]] for i in sorted_order
    ])
    boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    n_labels = len(np.unique(sorted_labels))

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3), constrained_layout=True)

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

        for boundary in boundaries:
            ax.axhline(boundary, color="black", linewidth=0.5)

        yticks = []
        for label_id in range(1, n_labels+1):
            label_position = np.argwhere(sorted_labels == label_id)[:, 0]
            yticks.append(label_position.mean())
        ax.set_yticks(yticks, [str(i) for i in range(1, n_labels+1)])
        for tick in ax.get_yticklabels():
            text = tick.get_text()
            tick.set_color(CLUSTER_COLORS[int(text)])
            tick.set_fontsize(8)
            tick.set_weight('bold')
            tick.set_rotation(0)

        ax2 = ax.twiny()
        ax2.set_xticks(tick_positions, [""]*len(tick_positions), rotation=0)  
        # ax2.tick_params(length=1, pad=1, axis='y')
        ax2.set_xlim(lim_tick_positions[0], lim_tick_positions[-1])

        #     avg_psth = matrix[sorted_order][label_position].mean(axis=0)
        #     ax.plot(np.arange(len(bin_centers)), -avg_psth/50 + label_center, lw=1, color="#bc5090", alpha=0.7)

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
    fig, ax = plt.subplots(1, 1, figsize=(2., 2.5), constrained_layout=True)
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
        for node_name in cluster_nodes:
            cell_data, cohort_name = node_to_spikes[node_name]
            ax.eventplot(np.concatenate(cell_data), lineoffsets=global_y_offset,
                             orientation='horizontal', linelengths=1, linewidths=0.3,
                             color=COHORT_COLORS[cohort_name])
            global_y_offset -= 1
        ax.fill_betweenx(y=[global_y_offset+0.5, cluster_y_start+0.5],
                         x1=0, x2=0.5 * 1000,
                        alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10,)
        global_y_offset -= 2


    global_y_offset -= 5
    all_global_LFP = grouping_timeseries(global_LFP)
    ax.plot(all_global_LFP.t * 1000, all_global_LFP.v * 10 + global_y_offset, color="k", lw=1,)

    ax.tick_params(
        length=1,
        pad=1     
    )
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(-20, 60)
    ax.set_xticks([0, 20, 40,])
    ax.axvline(0, color='gray', lw=0.5, ls="--", zorder=-1)
    ax.axvline(20, color='gray', lw=0.5, ls="--", zorder=-1)
    ax.set_xlabel("Time from puff onset (ms)", labelpad=1)



    save_path = path.join(get_saving_path(), task_name, f"EarlySpikeDistribution_sorted_by_cluster.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
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
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), constrained_layout=True)
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
