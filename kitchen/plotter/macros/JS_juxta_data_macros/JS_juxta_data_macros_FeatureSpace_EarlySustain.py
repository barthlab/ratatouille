from collections import defaultdict
import logging
from pathlib import Path
import time
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import os.path as path
import os
import pandas as pd

from kitchen.calculator import calculate_physiology
from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import grouping_timeseries
from kitchen.plotter import color_scheme
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_all_physiology_features, get_detailed_spike_timing_features, get_trial_wise_spike_timing_features
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_BRIGHTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_LFP_mean, get_all_cellsession_PSTH_mean, get_all_cellsession_Waveform_mean, get_saving_path, get_trialwise_detail_LFP_and_timing
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.settings.potential import NARROW_SPIKE_RANGE_FOR_VISUALIZATION, SPIKE_ANNOTATION_EARLY_WINDOW
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)



def VisualizeExample_Physiology_Fingerprint_Vertical(annotated: bool=True, bon_ferroni_adj: int=1):
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
            which="both",
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

        annotate_data = {}
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
            annotate_data[cohort_id+0.4] = values
        if annotated:
            basic_ttest.stats_pairwise_unpaired_ttest(annotate_data, ax, bon_ferroni_adj=bon_ferroni_adj)
            for x_coor, y_values in annotate_data.items():
                ax.text(x_coor, np.nanmean(y_values), f"{np.nanmean(y_values):.2f}±{np.nanstd(y_values):.2f}", 
                        ha='center', va='bottom', fontsize=3)
        ax.set_title(feature_name)
        ax.set_xlabel("")

        if "ACG" in feature_name:
            ax.set_yscale("log")
        # elif "FR" in feature_name and "Std" not in feature_name:
        #     ax.set_yscale("symlog", linthresh=10)
        #     ax.set_ylim(-5, 110)
        elif "Asymmetry" in feature_name and "Std" not in feature_name:
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([-0.5, 0, 0.5])


        valid_feature_name = feature_name.replace("\n", "_").replace("[", "").replace("]", "").replace(",", "")
        save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"{valid_feature_name}{'_annotated' if annotated else ''}.png")
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
       
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.7), constrained_layout=True)
    for ax in axs:
        ax.tick_params(
            length=1,
            pad=1     
        )
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_yscale('symlog', linthresh=1, linscale=0.1)
        ax.set_xscale('symlog', linthresh=1, linscale=0.1)
        ax.set_xticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
        ax.set_yticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Early spike rate (0-40ms, Hz)", fontsize=6, labelpad=1)
        ax.set_ylabel("Sustained spike rate (40-500ms, Hz)", fontsize=6, labelpad=1)

        ax.plot([0, 100], [0, 100], color="gray", lw=0.5, linestyle="--", zorder=-10)


    grouping_data_early, grouping_data_sustained = defaultdict(list), defaultdict(list) 
    for zorder_id, (cohort_id, cohort_name) in enumerate(zip([0, 2, 1, 0, 1], ["SST_JUX", "PV_JUX",  "PYR_JUX", "SST_WC", "PYR_WC"])):
        
        cohort_data = combined_physiology[combined_physiology["cohort_name"] == cohort_name]
        baseline_spont_FR = cohort_data["Spont. FR [Hz]"].values
        early_spike_FR = cohort_data["Early Spike Num\nAvg"].values / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0])
        sustained_spike_FR = cohort_data["Sustained Spike Num\nAvg"].values / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1])
        early_spike_FR_normed = early_spike_FR - baseline_spont_FR
        sustained_spike_FR_normed = sustained_spike_FR - baseline_spont_FR

        axs[0].scatter(early_spike_FR, sustained_spike_FR,
                        facecolor=COHORT_BRIGHTER_COLORS[cohort_name.replace("WC", "JUX")], 
                        edgecolor="black", lw=0.5, s=15, zorder=-zorder_id, alpha=0.9)

        axs[1].scatter(early_spike_FR_normed, sustained_spike_FR_normed,
                        facecolor=COHORT_BRIGHTER_COLORS[cohort_name.replace("WC", "JUX")], 
                        edgecolor="black", lw=0.5, s=15, zorder=-zorder_id, alpha=0.9)
        grouping_data_early[cohort_name.split("_")[0]].append(early_spike_FR)
        grouping_data_sustained[cohort_name.split("_")[0]].append(sustained_spike_FR)
    for key in grouping_data_early:
        grouping_data_early[key] = np.concatenate(grouping_data_early[key])
        grouping_data_sustained[key] = np.concatenate(grouping_data_sustained[key])
        print(f"{key}: early FR: {np.mean(grouping_data_early[key]):.3f}+/-{np.std(grouping_data_early[key]):.3f},"
                f" sustained FR: {np.mean(grouping_data_sustained[key]):.3f}+/-{np.std(grouping_data_sustained[key]):.3f}")

    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlyVsSustained_Scatter.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    annotate_data = {}
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5), constrained_layout=True)
    x_offset = 0.5
    for ax in axs:
        ax.tick_params(
            length=1,
            pad=1     
        )
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_yscale('symlog', linthresh=1, linscale=0.1)
        ax.set_xlim(-0.3, 2.3 + x_offset)
        ax.set_xticks(np.arange(3) + x_offset / 2, ["SST", "PV", "PYR"])
        for label, color in zip(ax.get_xticklabels(), [COHORT_COLORS["SST_JUX"], COHORT_COLORS["PV_JUX"], COHORT_COLORS["PYR_JUX"]]):
            label.set_color(color)
        ax.set_ylabel("Spike rate (Hz)", labelpad=1)
        ax.set_yticks([0, 1, 10, 100] + [i for i in range(2, 10)] + [i for i in range(20, 100, 10)] + [200,], 
                      labels=["0", "1", "10", "100"] + ["",]*17)
    for zorder_id, (cohort_id, cohort_name) in enumerate(zip([0, 1, 2,], ["SST_JUX", "PV_JUX",  "PYR_JUX", ])):
        
        cohort_data = combined_physiology[combined_physiology["cohort_name"] == cohort_name]
        baseline_spont_FR = cohort_data["Spont. FR [Hz]"].values
        early_spike_FR = cohort_data["Early Spike Num\nAvg"].values / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0])
        sustained_spike_FR = cohort_data["Sustained Spike Num\nAvg"].values / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1])
        early_spike_FR_normed = early_spike_FR - baseline_spont_FR
        sustained_spike_FR_normed = sustained_spike_FR - baseline_spont_FR

        axs[0].scatter([cohort_id,] * len(early_spike_FR), early_spike_FR, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
        axs[0].scatter(cohort_id-0.1, np.nanmean(early_spike_FR),
                       facecolor=COHORT_COLORS[cohort_name], edgecolor="black", lw=0.5, s=10, zorder=zorder_id+1, alpha=0.9)
        
        axs[0].scatter([cohort_id + x_offset,] * len(sustained_spike_FR), sustained_spike_FR, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
        axs[0].scatter(cohort_id + x_offset + 0.1, np.nanmean(sustained_spike_FR),
                       facecolor=COHORT_COLORS[cohort_name], edgecolor="black", lw=0.5, s=10, zorder=zorder_id+1, alpha=0.9)
        annotate_data[(cohort_id, cohort_id+x_offset)] = (early_spike_FR, sustained_spike_FR)


        axs[1].scatter([cohort_id,] * len(early_spike_FR_normed), early_spike_FR_normed, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
        axs[1].scatter(cohort_id-0.1, np.nanmean(early_spike_FR_normed),
                       facecolor=COHORT_COLORS[cohort_name], edgecolor="black", lw=0.5, s=10, zorder=zorder_id+1, alpha=0.9)
        
        axs[1].scatter([cohort_id + x_offset,] * len(sustained_spike_FR_normed), sustained_spike_FR_normed, 
                       edgecolor=COHORT_COLORS[cohort_name], facecolor="white", lw=0.5, s=6, zorder=zorder_id, alpha=0.9)
        axs[1].scatter(cohort_id + x_offset + 0.1, np.nanmean(sustained_spike_FR_normed),
                       facecolor=COHORT_COLORS[cohort_name], edgecolor="black", lw=0.5, s=10, zorder=zorder_id+1, alpha=0.9)
        
        # ax.plot([cohort_id, cohort_id + x_offset], [np.mean(early_spike_FR), np.mean(sustained_spike_FR)], color="k", lw=1.5)

        for node_early, node_baseline, node_sustained in zip(early_spike_FR, baseline_spont_FR, sustained_spike_FR):
            axs[0].plot([cohort_id, cohort_id + x_offset], [node_early, node_sustained], 
                        color=COHORT_COLORS[cohort_name], 
                    lw=0.5, alpha=0.8, zorder=-1)
        for node_early, node_sustained in zip(early_spike_FR_normed, sustained_spike_FR_normed):
            axs[1].plot([cohort_id, cohort_id + x_offset], [node_early, node_sustained], 
                        color=COHORT_COLORS[cohort_name], 
                    lw=0.5, alpha=0.8, zorder=-1)
            
    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlyVsSustained_Compare.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    basic_ttest.stats_pairwise_paired_ttest(annotate_data, axs[0])
    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlyVsSustained_Compare_annotated.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), constrained_layout=True)
    fig_pie, axs_pie = plt.subplots(3, 1, figsize=(0.8, 1.55), constrained_layout=True)
    fig_bar, axs_bar = plt.subplots(1, 1, figsize=(1.15, 1.75), constrained_layout=True)
    ax.tick_params(
        length=1,
        pad=1     
    )
    ax.spines[['right', 'top']].set_visible(False)
    for ax_pie in axs_pie:
        ax_pie.tick_params(
            length=1,
            pad=1     
        )
        ax_pie.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    axs_bar.tick_params(
        length=1,
        pad=1     
    )
    axs_bar.spines[['right', 'top', 'left']].set_visible(False)

    for zorder_id, (cohort_id, cohort_name) in enumerate(zip([0, 1, 2,], ["SST_JUX", "PV_JUX",  "PYR_JUX", ])):
        
        cohort_data = combined_physiology[combined_physiology["cohort_name"] == cohort_name]
        baseline_spont_FR = cohort_data["Spont. FR [Hz]"].values
        early_spike_FR = cohort_data["Early Spike Num\nAvg"].values / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0])
        sustained_spike_FR = cohort_data["Sustained Spike Num\nAvg"].values / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1])

        # delta_FR = sustained_spike_FR - early_spike_FR
        selective_index = (sustained_spike_FR - early_spike_FR) / (early_spike_FR + 1)
        ax.plot(np.sort(selective_index), np.linspace(0, 1, len(selective_index)), lw=1, color=COHORT_BRIGHTER_COLORS[cohort_name])

        ax_pie = axs_pie[zorder_id]
        up, down, same = np.sum(selective_index > 0.5), np.sum(selective_index < -0.5), np.sum(np.abs(selective_index) <= 0.5)
        
        def func(pct):
            absolute = int(round(pct/100.*len(selective_index)))
            if absolute == 0:
                return ""
            return f"{pct:.0f}%"
        wedges, texts, autotexts = ax_pie.pie([up, same, down], 
                    # labels=data.index, 
                    colors=[COHORT_COLORS[cohort_name]]*3, 
                    autopct=func,
                    shadow=False, 
                    startangle=90, 
                    pctdistance=0.6,
                    counterclock=False,
                    wedgeprops={'edgecolor': 'black', 'lw': 0})
        alphas = [0.9, 0.5, 0.1]        
        for wedge, alpha in zip(wedges, alphas):
            wedge.set_alpha(alpha)
            
        for wedge, autotext, prefix, radius in zip(wedges, autotexts, ["up", "same", "down"], [0.7, 0.55, 0.55]):
            if autotext.get_text() != "":
                autotext.set_text(prefix)
                # autotext.set_fontsize(5)

            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            autotext.set_position((x, y))
            autotext.set_ha('center')
            autotext.set_va('center')
        ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

        counts = np.array([down, same, up], dtype=float)
        percentages = 100 * counts / counts.sum() 

        bottom = 0
        for percentage, category, alpha in zip(
            percentages, ["Down", "Same", "Up"], reversed(alphas)
        ):
            axs_bar.bar(
                cohort_id, percentage,
                bottom=bottom,
                width=0.65,
                facecolor=to_rgba(COHORT_COLORS[cohort_name], alpha),
                edgecolor='black',
                linewidth=0.7,
                label=category if cohort_id == 0 else None,
            )
            if percentage > 0:
                axs_bar.text(
                    cohort_id, bottom + percentage / 2,
                    f"{percentage:.0f}%",
                    # f"{category}\n{percentage:.0f}%",
                    ha="center", va="center", fontsize=6,
                )
            bottom += percentage
        
    ax.axvline(-0.5, lw=0.5, color='gray', ls='--')
    ax.axvline(0.5, lw=0.5, color='gray', ls='--')
    ax.text(-0.8, 1.05, "down", ha='center')
    ax.text(1.8, 1.05, "up", ha='center')
    ax.text(0., 1.05, "same", ha='center')
    ax.set_xticks([-1, 0, 1, 2, ])
    ax.set_xlabel("Fold Change", labelpad=1)
    ax.set_ylabel("Cumulative Fraction", labelpad=1)
    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlyVsSustained_foldchange.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlyVsSustained_pie.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig_pie.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig_pie)
    logger.info("Plot saved to " + save_path)

    axs_bar.set_ylim(0, 102)
    axs_bar.set_yticks([])
    axs_bar.set_ylabel("Cells (%)", labelpad=1)
    axs_bar.set_xticks([0, 1, 2], ["SST", "PV", "PYR"])
    

    axs_bar.legend(
        handles=[
            Patch(facecolor=(0, 0, 0, a), edgecolor="black", label=c)
            for c, a in zip(["Up", "Same", "Down"], alphas)
        ],
        frameon=False, fontsize=6, ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        handlelength=1, handletextpad=0.3, columnspacing=0.6,
    )

    for label, cohort_name in zip(
        axs_bar.get_xticklabels(), ["SST_JUX", "PV_JUX", "PYR_JUX"]
    ):
        label.set_color(COHORT_COLORS[cohort_name])


    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", "EarlyVsSustained_bar.png",)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_bar.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig_bar)
    logger.info("Plot saved to " + save_path)



def VisualizeExample_Physiology_Fingerprint_EarlySpikeDistribution():

    
    import matplotlib.pyplot as plt    
    import seaborn as sns

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), constrained_layout=True)
    fig_dist, ax_dist = plt.subplots(1, 1, figsize=(1.5, 1.5), constrained_layout=True)
    global_LFP = []

    def _good_cell(cell_spike_timing):
        num_trial = len(cell_spike_timing)
        total_spike_time = sum([np.sum((-1000<=trial) & (trial<=1500)) for trial in cell_spike_timing])
        return num_trial >= 5 and total_spike_time >= 1

    global_y_offset = 0
    annotate_data = {}
    for cohort_id, cohort_name_group in zip([2, 1, 0], [["SST_JUX", "SST_WC"], ["PV_JUX",],  ["PYR_JUX", "PYR_WC"]]):

        puff_trials_500ms, puff_trials_100ms = [], []
        for cohort_name in cohort_name_group:
            trial_wise_spike_timing = get_trial_wise_spike_timing_features(cohort_name)
            all_trial_type = trial_wise_spike_timing["all_trial_type"]
            indices_500ms = np.where(all_trial_type == "500msPuff")[0]
            indices_100ms = np.where(all_trial_type == "100msPuff")[0]
            if len(indices_500ms) > 0:
                puff_trials_500ms.extend([trial_wise_spike_timing["all_spikes"][i] for i in indices_500ms
                                          if _good_cell(trial_wise_spike_timing["all_spikes"][i])])
            if len(indices_100ms) > 0:
                puff_trials_100ms.extend([trial_wise_spike_timing["all_spikes"][i] for i in indices_100ms
                                          if _good_cell(trial_wise_spike_timing["all_spikes"][i])])

            if "WC" not in cohort_name:
                LFP_t, all_LFP = get_all_cellsession_LFP_mean(cohort_name)
                global_LFP.append(TimeSeries(t=LFP_t, v=np.mean(all_LFP, axis=0)))

        print(f"{cohort_name_group}: Found {len(puff_trials_500ms)} cells for 500ms puff and {len(puff_trials_100ms)} cells for 100ms puff.")

        median_early_spike_500ms = [np.median(np.concatenate([trial[np.where((trial < 40) & (trial > 0))] 
                                                              for trial in cell_data])) 
                                    for cell_data in puff_trials_500ms]
        median_early_spike_100ms = [np.median(np.concatenate([trial[np.where((trial < 40) & (trial > 0))] 
                                                              for trial in cell_data])) 
                                    for cell_data in puff_trials_100ms]

        
        median_all = np.array(median_early_spike_100ms + median_early_spike_500ms)
        median_all = median_all[median_all>16]  # filter out artifacts
        median_all = median_all[~np.isnan(median_all)]
        ax_dist.scatter(np.sort(median_all), np.arange(1, 1 + len(median_all))/len(median_all),
                        facecolor=COHORT_BRIGHTER_COLORS[cohort_name_group[0]], edgecolor='black',
                        s=7, lw=0.5, alpha=0.9, zorder=10)
        ax_dist.plot(np.sort(median_all), np.arange(1, 1 + len(median_all))/len(median_all),
                     color=COHORT_BRIGHTER_COLORS[cohort_name_group[0]], lw=1, zorder=-1, alpha=0.9)
        print(f"{cohort_name_group}: Median early spike times for all cells: {np.mean(median_all):.3f}+/-{np.std(median_all):.3f}")
        annotate_data[cohort_name_group[0]] = median_all


        sort_order_500ms = np.argsort(median_early_spike_500ms)
        sort_order_100ms = np.argsort(median_early_spike_100ms)
        puff_trials_500ms_sorted = [puff_trials_500ms[i] for i in sort_order_500ms]
        puff_trials_100ms_sorted = [puff_trials_100ms[i] for i in sort_order_100ms]

        for cell_data in puff_trials_500ms_sorted:
            ax.eventplot(np.concatenate(cell_data), lineoffsets=global_y_offset,
                             orientation='horizontal', linelengths=1, linewidths=0.3,
                             color=COHORT_COLORS[cohort_name_group[0]])
            global_y_offset -= 1
        ax.fill_betweenx(y=[global_y_offset+0.5, global_y_offset + len(puff_trials_500ms_sorted)+0.5],
                         x1=0, x2=0.5 * 1000,
                        alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10,)
        global_y_offset -= 1

        # if len(puff_trials_100ms_sorted) == 0 or not "SST_JUX" in cohort_name_group:
        #     global_y_offset -= 1
        #     continue
        # for cell_data in puff_trials_100ms_sorted:
        #     ax.eventplot(np.concatenate(cell_data), lineoffsets=global_y_offset,
        #                      orientation='horizontal', linelengths=1, linewidths=0.2,
        #                      color=COHORT_COLORS[cohort_name_group[0]])
        #     global_y_offset -= 1
        # ax.fill_betweenx(y=[global_y_offset+0.5, global_y_offset + len(puff_trials_100ms_sorted)+0.5],
        #                  x1=0, x2=0.1 * 1000,
        #                  alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10,)
        global_y_offset -= 1


    global_y_offset -= 5
    all_global_LFP = grouping_timeseries(global_LFP)
    ax.plot(all_global_LFP.t * 1000, all_global_LFP.v * 10 + global_y_offset, color="k", lw=1,)

    ax.tick_params(
        length=1,
        pad=1     
    )
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlim(-20, 80)
    ax.set_xticks([0, 20, 40, 60, ])
    ax.axvline(0, color='gray', lw=0.5, ls="--", zorder=-1)
    ax.axvline(20, color='gray', lw=0.5, ls="--", zorder=-1)
    ax.set_xlabel("Time from puff onset (ms)", labelpad=1)

    ax_dist.tick_params(
        length=1,
        pad=1     
    )
    ax_dist.spines[['right', 'top']].set_visible(False)
    ax_dist.set_xticks([18, 20, 22, 24, 26])
    ax_dist.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax_dist.set_xlabel("Median early spike time (ms)", labelpad=1)
    ax_dist.set_ylabel("Cumulative distribution", labelpad=1)


    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlySpikeDistribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlySpikeDistribution_CDF.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig_dist.savefig(save_path, dpi=500, transparent=True)
    logger.info("Plot saved to " + save_path)

    
    basic_ttest.stats_pairwise_ks(annotate_data, ax=ax_dist)
    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlySpikeDistribution_CDF_annotated.png")
    fig_dist.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig_dist)
    logger.info("Plot saved to " + save_path)




def VisualizeExample_Physiology_Fingerprint_EarlySpikeIndicatedTrailState(_annotate: bool = True):

    
    import matplotlib.pyplot as plt    
    import seaborn as sns

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    
    fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.2), constrained_layout=True)

    intrude_offset = 0.2
    annotate_data = {}
    for cohort_offset, cohort_id, cohort_name in zip([-0.1, 0, 0.1], [2, 1, 0], ["SST_JUX", "PV_JUX",  "PYR_JUX",]):

        puff_trials_500ms, puff_trials_100ms = [], []

        LFP_t, all_LFP, all_spikes, all_node_name = get_trialwise_detail_LFP_and_timing(cohort_name)
        LFP_indices = (np.searchsorted(LFP_t, 0/1000), np.searchsorted(LFP_t, 40/1000))
        assert len(all_LFP) == len(all_spikes) == len(all_node_name), \
            f"Inconsistent data lengths for cohort {cohort_name}: {len(all_LFP)}, {len(all_spikes)}, {len(all_node_name)}"

        sustained_summary, lfp_summary = defaultdict(list), defaultdict(list) 
        for cell_idx, cell_LFP, cell_spikes, cell_node_name in zip(range(len(all_LFP)), all_LFP, all_spikes, all_node_name):
            assert len(cell_LFP) == len(cell_spikes), f"Inconsistent trial lengths for cell {cell_node_name}: {len(cell_LFP)} LFP trials, {len(cell_spikes)} spike trials"
            assert sum([len(trial) for trial in cell_LFP]) >= 5, f"No enough LFP data for cell {cell_node_name}, total trials: {sum([len(trial) for trial in cell_LFP])}"
            assert sum([len(trial) for trial in cell_spikes]) >= 5, f"No enough spike data for cell {cell_node_name}, total trials: {sum([len(trial) for trial in cell_spikes])}"

            sustained_cell, lfp_cell = defaultdict(list), defaultdict(list) 
            for trial_LFP, trial_spikes in zip(cell_LFP, cell_spikes):
                early_spike_num = np.sum((trial_spikes < 40) & (trial_spikes > 0))
                early_spike_num = min(early_spike_num, 2)
                sustained_spike_num = np.sum((trial_spikes > 40) & (trial_spikes < 500))
                LFP_size = np.min(trial_LFP[LFP_indices[0]:LFP_indices[1]])
                sustained_cell[early_spike_num].append(sustained_spike_num)
                lfp_cell[early_spike_num].append(LFP_size)

            sustained_cell = {key: np.mean(value) for key, value in sorted(sustained_cell.items(), key=lambda item: item[0])}
            lfp_cell = {key: np.mean(value) for key, value in sorted(lfp_cell.items(), key=lambda item: item[0])}
            if tuple(sustained_cell.keys()) != (0, 1, 2):
                continue
            for key, value in sustained_cell.items():
                sustained_summary[key].append(value)
            for key, value in lfp_cell.items():
                lfp_summary[key].append(value)

            # ax.plot(list(sustained_cell.keys()), list(sustained_cell.values()), 
            #            color=COHORT_BRIGHTER_COLORS[cohort_name], lw=0.5, alpha=0.2)
            ax.plot([0+intrude_offset, 1-intrude_offset], [lfp_cell[0], lfp_cell[1]], 
                       color=COHORT_BRIGHTER_COLORS[cohort_name], lw=0.5, alpha=0.7, zorder=cohort_id)
            ax.plot([1+intrude_offset, 2-intrude_offset], [lfp_cell[1], lfp_cell[2]], 
                       color=COHORT_BRIGHTER_COLORS[cohort_name], lw=0.5, alpha=0.7, zorder=cohort_id)
        print(cohort_name, [len(sustained_summary[i]) for i in range(3)], [len(lfp_summary[i]) for i in range(3)])
        avg_sustained = {key: np.mean(value) for key, value in sorted(sustained_summary.items(), key=lambda item: item[0])}
        avg_lfp = {key: np.mean(value) for key, value in sorted(lfp_summary.items(), key=lambda item: item[0])}
        std_sustained = {key: np.std(value) for key, value in sorted(sustained_summary.items(), key=lambda item: item[0])}
        std_lfp = {key: np.std(value) for key, value in sorted(lfp_summary.items(), key=lambda item: item[0])}
        # ax.errorbar(list(avg_sustained.keys()), list(avg_sustained.values()), yerr=list(std_sustained.values()), 
        #               fmt="o",
        #                 markeredgecolor="black",
        #                 markerfacecolor=COHORT_BRIGHTER_COLORS[cohort_name],
        #                 markeredgewidth=0.5,
        #                 markersize=3,
        #                 ecolor='black',
        #                 elinewidth=0.5,
        #                 capsize=2,
        #                 capthick=0.5,
        #                 alpha=0.9,
        #                 )
        annotate_data[(0+cohort_offset, 1+cohort_offset)] = (lfp_summary[0], lfp_summary[1])
        annotate_data[(1+cohort_offset, 2+cohort_offset)] = (lfp_summary[1], lfp_summary[2])
        ax.errorbar(np.array(list(avg_lfp.keys())) + cohort_offset, list(avg_lfp.values()), yerr=list(std_lfp.values()), 
                      fmt="o",
                        markeredgecolor="black",
                        markerfacecolor=COHORT_BRIGHTER_COLORS[cohort_name],
                        markeredgewidth=0.5,
                        markersize=3,
                        ecolor='black',
                        elinewidth=0.5,
                        capsize=1,
                        capthick=0.5,
                        alpha=1,
                        zorder=cohort_id,
                      )
        # ax.plot(np.array(list(avg_lfp.keys())) + cohort_offset, list(avg_lfp.values()), 
        #         color=COHORT_BRIGHTER_COLORS[cohort_name], lw=1, alpha=0.9, zorder=cohort_id)

    ax.tick_params(
        length=1,
        pad=1     
    )
    ax.set_xlabel("Early spike number (#)", labelpad=1)
    ax.set_ylabel("Mean LFP amplitude (mV)", labelpad=1)
    ax.set_xticks([0, 1, 2], ["0", "1", "2+"])
    ax.axhline(0, color='gray', lw=0.5, ls='--', zorder=-1)
    ax.spines[['right', 'top']].set_visible(False)
    if _annotate:
        basic_ttest.stats_pairwise_paired_ttest(annotate_data, ax)
    save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlySpikeIndicatedTrailState{'_annotated' if _annotate else ''}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



    
    

    intrude_offset = 0.2
    for cohort_offset, cohort_id, cohort_name in zip([-0., 0, 0.], [2, 1, 0], ["SST_JUX", "PV_JUX",  "PYR_JUX",]):
        annotate_data = {}
        
        fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.2), constrained_layout=True)

        puff_trials_500ms, puff_trials_100ms = [], []

        LFP_t, all_LFP, all_spikes, all_node_name = get_trialwise_detail_LFP_and_timing(cohort_name)
        LFP_indices = (np.searchsorted(LFP_t, 0/1000), np.searchsorted(LFP_t, 40/1000))
        assert len(all_LFP) == len(all_spikes) == len(all_node_name), \
            f"Inconsistent data lengths for cohort {cohort_name}: {len(all_LFP)}, {len(all_spikes)}, {len(all_node_name)}"

        sustained_summary, lfp_summary = defaultdict(list), defaultdict(list) 
        for cell_idx, cell_LFP, cell_spikes, cell_node_name in zip(range(len(all_LFP)), all_LFP, all_spikes, all_node_name):
            assert len(cell_LFP) == len(cell_spikes), f"Inconsistent trial lengths for cell {cell_node_name}: {len(cell_LFP)} LFP trials, {len(cell_spikes)} spike trials"
            assert sum([len(trial) for trial in cell_LFP]) >= 5, f"No enough LFP data for cell {cell_node_name}, total trials: {sum([len(trial) for trial in cell_LFP])}"
            assert sum([len(trial) for trial in cell_spikes]) >= 5, f"No enough spike data for cell {cell_node_name}, total trials: {sum([len(trial) for trial in cell_spikes])}"

            sustained_cell, lfp_cell = defaultdict(list), defaultdict(list) 
            for trial_LFP, trial_spikes in zip(cell_LFP, cell_spikes):
                early_spike_num = np.sum((trial_spikes < 40) & (trial_spikes > 0))
                early_spike_num = min(early_spike_num, 2)
                sustained_spike_num = np.sum((trial_spikes > 40) & (trial_spikes < 500))
                LFP_size = np.min(trial_LFP[LFP_indices[0]:LFP_indices[1]])
                sustained_cell[early_spike_num].append(sustained_spike_num)
                lfp_cell[early_spike_num].append(LFP_size)

            sustained_cell = {key: np.mean(value) for key, value in sorted(sustained_cell.items(), key=lambda item: item[0])}
            lfp_cell = {key: np.mean(value) for key, value in sorted(lfp_cell.items(), key=lambda item: item[0])}
            if tuple(sustained_cell.keys()) != (0, 1, 2):
                continue
            for key, value in sustained_cell.items():
                sustained_summary[key].append(value)
            for key, value in lfp_cell.items():
                lfp_summary[key].append(value)

            # ax.plot(list(sustained_cell.keys()), list(sustained_cell.values()), 
            #            color=COHORT_BRIGHTER_COLORS[cohort_name], lw=0.5, alpha=0.2)
            ax.plot([0+intrude_offset, 1-intrude_offset], [lfp_cell[0], lfp_cell[1]], 
                       color=COHORT_BRIGHTER_COLORS[cohort_name], lw=0.5, alpha=0.7, zorder=cohort_id)
            ax.plot([1+intrude_offset, 2-intrude_offset], [lfp_cell[1], lfp_cell[2]], 
                       color=COHORT_BRIGHTER_COLORS[cohort_name], lw=0.5, alpha=0.7, zorder=cohort_id)
        print(cohort_name, [len(sustained_summary[i]) for i in range(3)], [len(lfp_summary[i]) for i in range(3)])
        avg_sustained = {key: np.mean(value) for key, value in sorted(sustained_summary.items(), key=lambda item: item[0])}
        avg_lfp = {key: np.mean(value) for key, value in sorted(lfp_summary.items(), key=lambda item: item[0])}
        std_sustained = {key: np.std(value) for key, value in sorted(sustained_summary.items(), key=lambda item: item[0])}
        std_lfp = {key: np.std(value) for key, value in sorted(lfp_summary.items(), key=lambda item: item[0])}
        # ax.errorbar(list(avg_sustained.keys()), list(avg_sustained.values()), yerr=list(std_sustained.values()), 
        #               fmt="o",
        #                 markeredgecolor="black",
        #                 markerfacecolor=COHORT_BRIGHTER_COLORS[cohort_name],
        #                 markeredgewidth=0.5,
        #                 markersize=3,
        #                 ecolor='black',
        #                 elinewidth=0.5,
        #                 capsize=2,
        #                 capthick=0.5,
        #                 alpha=0.9,
        #                 )
        annotate_data[(0+cohort_offset, 1+cohort_offset)] = (lfp_summary[0], lfp_summary[1])
        annotate_data[(1+cohort_offset, 2+cohort_offset)] = (lfp_summary[1], lfp_summary[2])
        ax.errorbar(np.array(list(avg_lfp.keys())) + cohort_offset, list(avg_lfp.values()), yerr=list(std_lfp.values()), 
                      fmt="o",
                        markeredgecolor="black",
                        markerfacecolor=COHORT_BRIGHTER_COLORS[cohort_name],
                        markeredgewidth=0.5,
                        markersize=3,
                        ecolor='black',
                        elinewidth=0.5,
                        capsize=1,
                        capthick=0.5,
                        alpha=1,
                        zorder=cohort_id,
                      )
        # ax.plot(np.array(list(avg_lfp.keys())) + cohort_offset, list(avg_lfp.values()), 
        #         color=COHORT_BRIGHTER_COLORS[cohort_name], lw=1, alpha=0.9, zorder=cohort_id)

        ax.tick_params(
            length=1,
            pad=1     
        )
        ax.set_xlabel("Early spike number (#)", labelpad=1)
        ax.set_ylabel("Mean LFP amplitude (mV)", labelpad=1)
        ax.set_xticks([0, 1, 2], ["0", "1", "2+"])
        ax.axhline(0, color='gray', lw=0.5, ls='--', zorder=-1)
        ax.spines[['right', 'top']].set_visible(False)
        if _annotate:
            basic_ttest.stats_pairwise_paired_ttest(annotate_data, ax)
        save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"EarlySpikeIndicatedTrailState{'_annotated' if _annotate else ''}_{cohort_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)