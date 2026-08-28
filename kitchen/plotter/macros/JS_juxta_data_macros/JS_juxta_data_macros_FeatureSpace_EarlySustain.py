import logging
from matplotlib.gridspec import GridSpec
import numpy as np
import os.path as path
import os
import pandas as pd

from kitchen.calculator import calculate_physiology
from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import grouping_timeseries
from kitchen.plotter import color_scheme
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_ClusteringAnalysis import get_putative_labels
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_all_physiology_features
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_LFP_mean, get_all_cellsession_PSTH_mean, get_all_cellsession_Waveform_mean, get_saving_path
from kitchen.settings.potential import NARROW_SPIKE_RANGE_FOR_VISUALIZATION, SPIKE_ANNOTATION_EARLY_WINDOW
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)



# def get_all_physiology_features_trial_level(
#         dataset_name: str,
# ):
               
#     pkl_save_path = routing.robust_path_join(
#         routing.DATA_PATH,
#         "PassivePuff_JuxtaCellular_FromJS_202509",
#         dataset_name,
#         "ARCHIVE_all_physiology_features.csv"
#     )
#     if not os.path.exists(pkl_save_path):     
#         dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
#                                 recipe="default_ephys", name=dataset_name)
#         all_physiology_features = []
#         for cell_session_node in dataset.select("cellsession"):
#             puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
#             if puff_trials_500ms is None:
#                 continue

#             LFP_peak = [one_trial.potential.aspect(4).segment(*SPIKE_ANNOTATION_EARLY_WINDOW).v.min() for one_trial in puff_trials_500ms]

#             early_spikes = [one_trial.potential.spikes.filter("early_spike") for one_trial in puff_trials_500ms]
#             early_spike_median_time = [np.median(early_spike.t) for early_spike in early_spikes if len(early_spike) > 0]
#             early_spike_first_time = [np.min(early_spike.t) for early_spike in early_spikes if len(early_spike) > 0]
#             early_spike_num = [len(early_spike) for early_spike in early_spikes]

#             sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike") for one_trial in puff_trials_500ms]
#             sustained_spike_median_time = [np.median(sustained_spike.t) for sustained_spike in sustained_spikes if len(sustained_spike) > 0]
#             sustained_spike_num = [len(sustained_spike) for sustained_spike in sustained_spikes]
#             sustained_spike_first_time = [np.min(sustained_spike.t) for sustained_spike in sustained_spikes if len(sustained_spike) > 0]

#             pre_stim_spont_spike_num = [np.sum((-1 < one_trial.potential.spikes.t) & (one_trial.potential.spikes.t < 0)) 
#                                         for one_trial in puff_trials_500ms if len(one_trial.potential.spikes) > 0]
            
#             all_spikes = cell_session_node.potential.spikes.t
#             potential_timeseries = cell_session_node.potential.aspect('raw')
#             spike_timeseries = potential_timeseries.batch_segment(all_spikes, NARROW_SPIKE_RANGE_FOR_VISUALIZATION)
 
#             if len(spike_timeseries) > 0:
                
#                 grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
#                 zscored_waveform = numpy_kit.zscore(grouped_spike_timeseries.raw_array, axis=1)
                
#                 width_ms, asymmetry_au = calculate_physiology.calculate_spike_width_and_asymmetry(zscored_waveform, grouped_spike_timeseries.fs)
#                 cv2_au = calculate_physiology.calculate_cv2(all_spikes)
#                 acg_rise_time_s = calculate_physiology.calculate_autocorrelogram_rise_time(all_spikes)
#             else:
#                 width_ms = np.nan
#                 asymmetry_au = np.nan
#                 cv2_au = np.nan
#                 acg_rise_time_s = np.nan
            
#             all_physiology_features.append({
#                 "LFP peak\nAvg [mV]": collapse_mean(LFP_peak),
#                 "LFP peak\nStd [mV]": collapse_std(LFP_peak),

#                 "Early Spike Median Timing\nAvg [ms]": collapse_mean(early_spike_median_time) * 1000,
#                 "Early Spike Median Timing\nStd [ms]": collapse_std(early_spike_median_time) * 1000,
#                 "Early Spike Num\nAvg": collapse_mean(early_spike_num),
#                 "Early Spike Num\nStd": collapse_std(early_spike_num),
#                 "Early Spike First Timing\nAvg [ms]": collapse_mean(early_spike_first_time) * 1000,
#                 "Early Spike First Timing\nStd [ms]": collapse_std(early_spike_first_time) * 1000,

#                 "Sustained Spike Median Timing\nAvg [ms]": collapse_mean(sustained_spike_median_time) * 1000,
#                 "Sustained Spike Median Timing\nStd [ms]": collapse_std(sustained_spike_median_time) * 1000,
#                 "Sustained Spike Num\nAvg": collapse_mean(sustained_spike_num),
#                 "Sustained Spike Num\nStd": collapse_std(sustained_spike_num),
#                 "Sustained Spike First Timing\nAvg [ms]": collapse_mean(sustained_spike_first_time) * 1000,
#                 "Sustained Spike First Timing\nStd [ms]": collapse_std(sustained_spike_first_time) * 1000,

#                 "Spont. FR [Hz]": collapse_mean(pre_stim_spont_spike_num),
#                 "Spont. FR\nStd [Hz]": collapse_std(pre_stim_spont_spike_num),

#                 "Spike Width [ms]": collapse_mean(width_ms),
#                 "Spike Width\nStd [ms]": collapse_std(width_ms),
#                 "Spike Asymmetry [a.u.]": collapse_mean(asymmetry_au),
#                 "Spike Asymmetry\nStd [a.u.]": collapse_std(asymmetry_au),
#                 "CV2 [a.u.]": cv2_au,
#                 "ACG Rise Time [s]": acg_rise_time_s,

#                 "node_name": str(cell_session_node.coordinate),
#                 "cohort_name": dataset_name,
#             })
            
#         all_physiology_features = pd.DataFrame(all_physiology_features)
#         all_physiology_features.to_csv(pkl_save_path, index=False)  
#         logger.info("Physiology features saved to " + pkl_save_path)

#     all_physiology_features = pd.read_csv(pkl_save_path)
#     logger.info("Physiology features loaded from " + pkl_save_path)
#     return all_physiology_features


def VisualizeExample_Physiology_Fingerprint_Vertical():
    sst_wc_physiology = get_all_physiology_features("SST_WC")
    sst_jux_physiology = get_all_physiology_features("SST_JUX")
    pv_jux_physiology = get_all_physiology_features("PV_JUX")
    pyr_jux_physiology = get_all_physiology_features("PYR_JUX")

    combined_physiology = pd.concat(
        [sst_wc_physiology, 
         sst_jux_physiology, 
         pv_jux_physiology, 
         pyr_jux_physiology], ignore_index=True)

    
    import matplotlib.pyplot as plt    
    import seaborn as sns

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    cohort_names = ["SST_JUX", "PV_JUX", "PYR_JUX"]
    cohort_labels = ["SST", "PV", "PYR"]
    rng = np.random.default_rng(0)
    for feature_name in combined_physiology.columns[:-2]:
        fig, ax = plt.subplots(1, 1, figsize=(0.9, 1.2), constrained_layout=True)
        ax.tick_params(
            length=1,
            pad=1
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlim(-0.4, 2.7)
        ax.set_xticks(np.arange(3) + 0.2, cohort_labels, fontsize=6)

        for label, color in zip(
            ax.get_xticklabels(),
            [COHORT_COLORS["SST_JUX"],
             COHORT_COLORS["PV_JUX"],
             COHORT_COLORS["PYR_JUX"],]
        ):
            label.set_color(color)

        for cohort_id, cohort_name in enumerate(cohort_names):

            cohort_data = combined_physiology[combined_physiology["cohort_name"] == cohort_name]
            values = cohort_data[feature_name].dropna().values
            if len(values) == 0:
                continue

            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                np.full(len(values), cohort_id) + jitter, values,
                edgecolor=COHORT_COLORS[cohort_name],
                facecolor="white", lw=0.5, s=4, alpha=0.9, zorder=2,
            )
            ax.errorbar(cohort_id + 0.4, np.nanmean(values), 
                        yerr=np.nanstd(values) if "ACG" not in feature_name else None,
                        fmt="o",
                        markeredgecolor="black",
                        markerfacecolor=COHORT_COLORS[cohort_name],
                        markeredgewidth=0.5,
                        markersize=3,
                        ecolor='black',
                        elinewidth=0.5,
                        capsize=2,
                        capthick=0.5,
                        alpha=0.9,)
        
        ax.set_title(feature_name)
        ax.set_xlabel("")

        if "ACG" in feature_name:
            ax.set_yscale("log")
        elif "FR" in feature_name and "Std" not in feature_name:
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(-5, 110)
        elif "Asymmetry" in feature_name and "Std" not in feature_name:
            ax.set_ylim(-0.6, 0.6)


        valid_feature_name = feature_name.replace("\n", "_").replace("[", "").replace("]", "").replace(",", "")
        save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"{valid_feature_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)


def VisualizeExample_Physiology_Fingerprint_EarlyVsSustained():
    sst_wc_physiology = get_all_physiology_features("SST_WC")
    sst_jux_physiology = get_all_physiology_features("SST_JUX")
    pv_jux_physiology = get_all_physiology_features("PV_JUX")
    pyr_jux_physiology = get_all_physiology_features("PYR_JUX")

    combined_physiology = pd.concat(
        [sst_wc_physiology, 
         sst_jux_physiology, 
         pv_jux_physiology, 
         pyr_jux_physiology], ignore_index=True)

    
    import matplotlib.pyplot as plt    
    import seaborn as sns

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    
    fig, axs = plt.subplots(1, 2, figsize=(4, 1.5), constrained_layout=True)
    x_offset = 0.7
    for ax in axs:
        ax.tick_params(
            length=1,
            pad=1     
        )
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_yscale('symlog', linthresh=10)
        ax.set_xlim(-0.3, 2.3 + x_offset)
        ax.set_xticks(np.arange(3) + x_offset / 2, ["SST", "PV", "PYR"], fontsize=8)
        for label, color in zip(ax.get_xticklabels(), [COHORT_COLORS["SST_JUX"], COHORT_COLORS["PV_JUX"], COHORT_COLORS["PYR_JUX"]]):
            label.set_color(color)
        ax.set_ylabel("Spike Rate [Hz]", fontsize=7)
    for zorder_id, (cohort_id, cohort_name) in enumerate(zip([0, 1, 2, 0], ["SST_JUX", "PV_JUX",  "PYR_JUX", "SST_WC"])):
        
        cohort_data = combined_physiology[combined_physiology["cohort_name"] == cohort_name]
        baseline_spont_FR = cohort_data["Spont. FR [Hz]"].values
        early_spike_FR = cohort_data["Early Spike Num\nAvg"].values / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0])
        sustained_spike_FR = cohort_data["Sustained Spike Num\nAvg"].values / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1])
        early_spike_FR_normed = early_spike_FR - baseline_spont_FR
        sustained_spike_FR_normed = sustained_spike_FR - baseline_spont_FR

        axs[0].scatter([cohort_id,] * len(early_spike_FR), early_spike_FR, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
        axs[0].scatter([cohort_id + x_offset,] * len(sustained_spike_FR), sustained_spike_FR, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)

        axs[1].scatter([cohort_id,] * len(early_spike_FR_normed), early_spike_FR_normed, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
        axs[1].scatter([cohort_id + x_offset,] * len(sustained_spike_FR_normed), sustained_spike_FR_normed, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)

        # ax.plot([cohort_id, cohort_id + x_offset], [np.mean(early_spike_FR), np.mean(sustained_spike_FR)], color="k", lw=1.5)

        for node_early, node_baseline, node_sustained in zip(early_spike_FR, baseline_spont_FR, sustained_spike_FR):
            axs[0].plot([cohort_id, cohort_id + x_offset], [node_early, node_sustained], 
                        color=COHORT_COLORS[cohort_name], 
                    lw=0.5, alpha=0.8, zorder=-1)
        for node_early, node_sustained in zip(early_spike_FR_normed, sustained_spike_FR_normed):
            axs[1].plot([cohort_id, cohort_id + x_offset], [node_early, node_sustained], 
                        color=COHORT_COLORS[cohort_name], 
                    lw=0.5, alpha=0.8, zorder=-1)

    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlyVsSustained_Scatter.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def VisualizeExample_Physiology_Fingerprint_EarlySpikeDistribution():

    
    import matplotlib.pyplot as plt    
    import seaborn as sns

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    
    fig, axs = plt.subplots(1, 2, figsize=(2, 1.5), constrained_layout=True)
    global_LFP = []
    for cohort_id, cohort_name in enumerate(["SST_JUX", "PV_JUX",  "PYR_JUX"]):
        cell_physiology = get_all_physiology_features(cohort_name)
        bin_centers, all_spikes_histogram, _, node_names = get_all_cellsession_PSTH_mean(cohort_name, 10/1000,  (-0.5, 1.0),)
        mask = (bin_centers > 0) & (bin_centers < 0.5)
        sorted_order = np.argsort(np.mean(all_spikes_histogram[:, mask], axis=1))
        sorted_node_names = [node_names[i] for i in sorted_order]
        y_offsets = np.linspace(0, 1, len(sorted_node_names)) * 1.2 + cohort_id * 1.5
        cell_first_spike_avg_time = [cell_physiology[cell_physiology["node_name"] == node_name]["Early Spike First Timing\nAvg [ms]"].values[0] 
                                    for node_name in sorted_node_names]
        cell_first_spike_std_time = [cell_physiology[cell_physiology["node_name"] == node_name]["Early Spike First Timing\nStd [ms]"].values[0]
                                    for node_name in sorted_node_names]
        cell_median_spike_avg_time = [cell_physiology[cell_physiology["node_name"] == node_name]["Early Spike Median Timing\nAvg [ms]"].values[0]
                                    for node_name in sorted_node_names]
        cell_median_spike_std_time = [cell_physiology[cell_physiology["node_name"] == node_name]["Early Spike Median Timing\nStd [ms]"].values[0]
                                    for node_name in sorted_node_names]

        LFP_t, all_LFP = get_all_cellsession_LFP_mean(cohort_name, (-0.02, 0.06))
        global_LFP.append(TimeSeries(t=LFP_t, v=np.mean(all_LFP, axis=0)))

        axs[0].errorbar(cell_first_spike_avg_time, y_offsets, xerr=cell_first_spike_std_time, 
                        fmt="o",
                        linestyle="none",
                        markeredgecolor=COHORT_COLORS[cohort_name],
                        markerfacecolor="white",
                        markeredgewidth=0.5,
                        markersize=2,
                        ecolor=COHORT_COLORS[cohort_name],
                        elinewidth=0.5,
                        alpha=0.9,)
        axs[1].errorbar(cell_median_spike_avg_time, y_offsets, xerr=cell_median_spike_std_time, 
                        fmt="o",
                        linestyle="none",
                        markeredgecolor=COHORT_COLORS[cohort_name],
                        markerfacecolor="white",
                        markeredgewidth=0.5,
                        markersize=2,
                        ecolor=COHORT_COLORS[cohort_name],
                        elinewidth=0.5,
                        alpha=0.9,)

    all_global_LFP = grouping_timeseries(global_LFP)
    axs[0].plot(all_global_LFP.t * 1000, all_global_LFP.v - 0.1, color="k", lw=1,)
    axs[1].plot(all_global_LFP.t * 1000, all_global_LFP.v - 0.1, color="k", lw=1,)

    for ax in axs:
        ax.tick_params(
            length=1,
            pad=1     
        )
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(-5, 35)
        ax.axvline(0, color='gray', lw=0.5, ls="--", zorder=-1)
        ax.axvspan(0, 0.5 * 1000, alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
    


    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlySpikeDistribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)