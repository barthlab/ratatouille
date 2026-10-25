from collections import defaultdict
import os
import os.path as path
from typing import Optional

from scipy import stats

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import AdvancedTimeSeries, grouping_events_histogram, grouping_timeseries
import kitchen.plotter.color_scheme as color_scheme
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.stats_tests import basic_ttest
import kitchen.plotter.style_dicts as style_dicts
from kitchen.configs import routing
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import COHORT_BRIGHTER_COLORS, COHORT_COLORS

import logging
import numpy as np

from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, NARROW_SPIKE_RANGE_FOR_VISUALIZATION
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)


plot_manual_spike4Hz = PlotManual(potential=4,)
plot_manual_spike300Hz = PlotManual(potential=300.)



def get_saving_path(dataset_name: Optional[str] = None):
    return routing.robust_path_join(
        routing.FIGURE_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name if dataset_name is not None else "",
    )


def get_all_cellsession_PSTH_mean(
          dataset_name: str, 
          BINSIZE,  # s
          FEATURE_RANGE = (-1, 1.5),  # (min, max) s
    ):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        f"ARCHIVE_PSTH_{FEATURE_RANGE[0]}_{FEATURE_RANGE[1]}_{BINSIZE}.npy"
    )
    
    if not os.path.exists(pkl_save_path):        
        bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        all_spikes_histogram, all_baseline_fr, all_node_name = [], [], []
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is None:
                continue
            spikes_histogram = grouping_events_histogram([single_trial.potential.spikes.segment(*FEATURE_RANGE) 
                                                                for single_trial in puff_trials_500ms], 
                                                                bins=bins, use_event_value_as_weight=False)
            if spikes_histogram.data_num < 5 or (spikes_histogram.mean.min() == spikes_histogram.mean.max()):
                    continue
            all_spikes_histogram.append(spikes_histogram.mean)
            baseline_fr = np.mean(spikes_histogram.mean[spikes_histogram.t < 0.])
            all_baseline_fr.append(baseline_fr)
            all_node_name.append(str(cell_session_node.coordinate))
        all_baseline_fr = np.array(all_baseline_fr)
        all_node_name = np.array(all_node_name)
        all_spikes_histogram = np.stack(all_spikes_histogram, axis=0)
        np.save(pkl_save_path, {
            "spike_histogram": all_spikes_histogram, 
            "baseline_fr": all_baseline_fr,
            "bin_centers": bin_centers,
            "node_name": all_node_name,
            })  # type: ignore
        logger.info("PSTH saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("PSTH loaded from " + pkl_save_path)
    return tmp_save["bin_centers"], tmp_save["spike_histogram"], tmp_save["baseline_fr"], tmp_save["node_name"]


def SummaryMetric_PSTH(
        BINSIZE = 5/1000,  # s
        FEATURE_RANGE = (-0.5, 1),  # (min, max) s
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import matplotlib.patches as patches
    import matplotlib.patheffects as pe

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    fig, axs = plt.subplots(1, 3, figsize=(6, 0.7), constrained_layout=True)  

    for ax in axs:
        ax.tick_params(
            pad=1,       # smaller gap between ticks and tick labels
            length=1,
        )
    
    for dataset_index, dataset_name_group in enumerate(( ("SST_JUX", "SST_WC"), ("PV_JUX",), ("PYR_JUX", "PYR_WC") ) ):  
        ax = axs[dataset_index]
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name_group[0]]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name_group[0]]
        
        all_spikes_histogram = []
        for single_dataset_name in dataset_name_group:
            bin_centers, single_group_spike_histogram, _, _ = get_all_cellsession_PSTH_mean(single_dataset_name, BINSIZE, FEATURE_RANGE)
            all_spikes_histogram.append(single_group_spike_histogram)
        all_spikes_histogram = np.concatenate(all_spikes_histogram, axis=0)

        oreo_plot(ax, AdvancedTimeSeries(t=bin_centers, 
                                        v=np.mean(all_spikes_histogram, axis=0),
                                        variance=np.std(all_spikes_histogram, axis=0), 
                                        raw_array=all_spikes_histogram), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name_group[0]], "lw": 0.5, "alpha": 0.9, "zorder": -dataset_index,}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, "zorder": -dataset_index})
        ax.axvspan(0, 0.5, 
                   alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
        ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim(-10, 200)
        ax.set_yticks([0, 50])
        ax.set_xlim(-0.5, 1)
        ax.set_xticks([0, 0.5])

    save_path = path.join(get_saving_path(), f"SummaryMetric_PSTH.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    fig, axs = plt.subplots(3, 1, figsize=(1., 2.4), constrained_layout=True)
    for ax in axs:
        ax.tick_params(
            pad=1,       # smaller gap between ticks and tick labels
            length=1,
        )
    
    scale_factor = 200
    for dataset_index, dataset_name_group in enumerate((("SST_JUX", "SST_WC"),  ("PV_JUX",), ("PYR_JUX", "PYR_WC"),) ):  
        ax=axs[dataset_index]
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name_group[0]]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name_group[0]]
        
        all_spikes_histogram = []
        for single_dataset_name in dataset_name_group:
            bin_centers, single_group_spike_histogram, _, _ = get_all_cellsession_PSTH_mean(single_dataset_name, BINSIZE, FEATURE_RANGE)
            all_spikes_histogram.append(single_group_spike_histogram)
        all_spikes_histogram = np.concatenate(all_spikes_histogram, axis=0)

        oreo_plot(ax, AdvancedTimeSeries(t=bin_centers, 
                                        v=np.mean(all_spikes_histogram, axis=0),
                                        variance=np.std(all_spikes_histogram, axis=0), 
                                        raw_array=all_spikes_histogram), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name_group[0]], "lw": 1, "alpha": 0.9, "zorder": -dataset_index,}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, "zorder": -dataset_index})
        
        ax.axhline(0, color=COHORT_COLORS[dataset_name_group[0]], 
                   linestyle='--', lw=1, alpha=0.5, zorder=-10)

        LFP_t, all_LFP = get_all_cellsession_LFP_mean(dataset_name_group[0], (-0.02, 0.06))
        oreo_plot(ax, AdvancedTimeSeries(t=LFP_t, 
                                        v=np.mean(all_LFP, axis=0)* scale_factor - 100, 
                                        variance=np.std(all_LFP, axis=0), 
                                        raw_array=all_LFP), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name_group[0]], "lw": 1., "alpha": 0.9, "zorder": -dataset_index}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, "zorder": -dataset_index})
        ax.axhline(-100, color=COHORT_COLORS[dataset_name_group[0]], 
                    linestyle='--', lw=1, alpha=0.5, zorder=-10)

        avg_lfp = np.mean(all_LFP, axis=0)
        print(f"{dataset_name_group[0]}: lfp minima {np.min(avg_lfp):.5f} at {LFP_t[np.argmin(avg_lfp)]:.5f}")

    for ax in axs:
        # ax.axvspan(0, 0.5, alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
        # ax.axvline(0, color='gray', linestyle='--', lw=0.5, zorder=-10)
        ax.axvline(20/1000, color='gray', linestyle='--', lw=0.5, zorder=-10)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.set_ylim(-300, 400)
        ax.set_yticks([0, 200], ["", ""])
        ax.set_xlim(0/1000, 40/1000)
        ax.set_xticks([0, 20/1000, 40/1000], ["0", "20", "40"])

    save_path = path.join(get_saving_path(), "SummaryMetric_PSTH_Zoomin.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)




def PlainHeatmap_PSTH(
        BINSIZE = 5/1000,  # s
        FEATURE_RANGE = (-0.5, 1.0),  # (min, max) s
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import SymLogNorm

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    fig, axs = plt.subplots(1, 3, figsize=(6, 1.2), constrained_layout=True)
    for ax, dataset_name_group in zip(axs, ( ("SST_JUX", "SST_WC"), ("PV_JUX",), ("PYR_JUX", "PYR_WC") )):
        ax.tick_params(
            pad=1,       # smaller gap between ticks and tick labels
            length=1,
        )
            
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name_group[0]]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name_group[0]]
        all_spikes_histogram = []
        for single_dataset_name in dataset_name_group:
            bin_centers, single_group_spike_histogram, _, _ = get_all_cellsession_PSTH_mean(single_dataset_name, BINSIZE, FEATURE_RANGE)
            all_spikes_histogram.append(single_group_spike_histogram)
        all_spikes_histogram = np.concatenate(all_spikes_histogram, axis=0)
        mask = (bin_centers > 0) & (bin_centers < 0.5)
        sorted_order = np.argsort(np.mean(all_spikes_histogram[:, mask], axis=1))
        sns.heatmap(all_spikes_histogram[sorted_order],  
                    ax=ax, cbar=False,
                    cmap='YlOrBr', 
                    vmin=0, vmax=100,)
                    # norm=SymLogNorm(linthresh=1., vmin=0, vmax=100, linscale=0.1),)
        
        desired_ticks = [0, 0.5]
        tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_ticks]
        ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

        ax.invert_yaxis()
        ax.set_xlabel("Time (s)", labelpad=1)
        ax.set_yticks([0.5, len(all_spikes_histogram)-0.5], 
                      [f"{len(all_spikes_histogram)}", "1"], rotation=0)
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

    save_path = path.join(get_saving_path(), "PlainHeatmap_PSTH.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def SurvivalHeatmap_PSTH(
        BINSIZE = 10/1000,  # s
        FEATURE_RANGE = (-0.5, 1.0),  # (min, max) s,
        threshold = 1 ,
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import SymLogNorm

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    fig, axs = plt.subplots(1, 3, figsize=(4, 1.2), constrained_layout=True)
    
    fig_cd, ax_cd = plt.subplots(1, 1, figsize=(1.2, 1.2), constrained_layout=True)
    annotation_data = {}
    for ax, dataset_name_group in zip(axs, ( ("SST_JUX", "SST_WC"), ("PV_JUX",), ("PYR_JUX", "PYR_WC") )):
        ax.tick_params(
            pad=1,       # smaller gap between ticks and tick labels
            length=1,
        )
            
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name_group[0]]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name_group[0]]
        all_spikes_histogram = []
        for single_dataset_name in dataset_name_group:
            bin_centers, single_group_spike_histogram, _, _ = get_all_cellsession_PSTH_mean(single_dataset_name, BINSIZE, FEATURE_RANGE)
            all_spikes_histogram.append(single_group_spike_histogram)
        all_spikes_histogram = np.concatenate(all_spikes_histogram, axis=0)
        mask = (bin_centers > 0) & (bin_centers < 0.5)
        sorted_order = np.argsort(np.mean(all_spikes_histogram[:, mask], axis=1))
        sns.heatmap(all_spikes_histogram[sorted_order],  
                    ax=ax, cbar=False,
                    cmap='YlOrBr', 
                    vmin=0, vmax=100,)
                    # norm=SymLogNorm(linthresh=1., vmin=0, vmax=100, linscale=0.1),)
        
        desired_ticks = [0, 0.5]
        tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_ticks]
        ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

        # baseline_bins
        baseline_mask = (bin_centers >= FEATURE_RANGE[0]) & (bin_centers < 0)
        baseline_mean = np.nanmean(all_spikes_histogram[:, baseline_mask], axis=1)
        baseline_SD_mask = (bin_centers >= FEATURE_RANGE[0]) & (bin_centers < 40/1000)
        baseline_SD = np.nanstd(all_spikes_histogram[:, baseline_SD_mask], axis=1)
        
        # survival bin: the last bin has larger values than baseline mean + threshold * baseline_SD
        above_threshold = (bin_centers[None, :] >= 0) & (all_spikes_histogram > (baseline_mean + threshold * baseline_SD)[:, None]) 
        survival_bin = np.max(np.where(above_threshold, np.arange(len(bin_centers))[None, :], -1), axis=1,)
        
        survival_time = np.full(len(survival_bin), np.nan)
        valid = survival_bin >= 0
        survival_time[valid] = bin_centers[survival_bin[valid]]

        # Overlay on the heatmap (coordinates are bin/row centers).
        ax.scatter(
            survival_bin[sorted_order] + 0.5,
            np.arange(len(sorted_order)) + 0.5,
            s=8,
            color="black",
            marker='*',
            linewidths=0,
            zorder=3,
        )

        annotation_data[dataset_name_group[0]] = survival_time
        
        ax_cd.scatter(np.sort(survival_time), np.linspace(0, 1, len(survival_time),),
                        facecolor=COHORT_BRIGHTER_COLORS[dataset_name_group[0]], edgecolor='black',
                        s=7, lw=0.5, alpha=0.9, zorder=10)
        ax_cd.plot(np.sort(survival_time), np.linspace(0, 1, len(survival_time),),
                     color=COHORT_BRIGHTER_COLORS[dataset_name_group[0]], lw=1, zorder=-1, alpha=0.9)

        ax.invert_yaxis()
        ax.set_xlabel("Time (s)", labelpad=1)
        ax.set_yticks([0.5, len(all_spikes_histogram)-0.5], 
                      [f"{len(all_spikes_histogram)}", "1"], rotation=0)
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

        ax.axvline(tick_positions[1], color='gray', lw=0.5, linestyle='--', alpha=0.5, zorder=9)
    save_path = path.join(get_saving_path(), "SurvivalHeatmap_PSTH.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    ax_cd.tick_params(
        length=1,
        pad=1     
    )
    ax_cd.axvspan(0, 0.5, alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
    ax_cd.spines[['right', 'top']].set_visible(False)
    ax_cd.set_xticks([0, 0.5], ["0", "0.5"])
    ax_cd.set_xlim(-0.3, 1.1)
    ax_cd.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax_cd.set_xlabel("Survival time (s)", labelpad=1)
    ax_cd.set_ylabel("Cumulative distribution", labelpad=1)

    save_path = path.join(get_saving_path(), "SurvivalHeatmap_CD.png")
    fig_cd.savefig(save_path, dpi=500, transparent=True)
    basic_ttest.stats_pairwise_ks(annotation_data, ax=ax_cd, loc='bottom right')
    for dataset_name, survival_time in annotation_data.items():        
        metric_text = f"{np.nanmedian(survival_time):.3f}±{np.nanstd(survival_time):.3f}"
        ax_cd.text(
            0.05, 0.95 - 0.05 * list(annotation_data.keys()).index(dataset_name),
            f"{dataset_name}: {metric_text}",
            transform=ax_cd.transAxes,
            fontsize=3,
            verticalalignment='top',
            zorder=1000
        )
    save_path = path.join(get_saving_path(), "SurvivalHeatmap_CD_annotated.png")
    fig_cd.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig_cd)
    logger.info("Plot saved to " + save_path)
    
    



def get_all_cellsession_LFP_mean(
          dataset_name: str, 
          FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
    ):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        f"ARCHIVE_LFP_{FEATURE_RANGE[0]}_{FEATURE_RANGE[1]}.npy"
    )
    if not os.path.exists(pkl_save_path):        
        all_LFP = []
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is None:
                continue
            LFP_timeseries = grouping_timeseries([one_trial.potential.aspect(4).segment(*FEATURE_RANGE) 
                                                  for one_trial in puff_trials_500ms])
            if LFP_timeseries.data_num < 5 or (LFP_timeseries.mean.min() == LFP_timeseries.mean.max()):
                    continue
            all_LFP.append(LFP_timeseries.mean_ts)
        all_LFP = grouping_timeseries(all_LFP)
        np.save(pkl_save_path, {
                "LFP": all_LFP.raw_array, 
                "t": all_LFP.t,
            })  # type: ignore
        logger.info("LFP saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("LFP loaded from " + pkl_save_path)
    return tmp_save["t"], tmp_save["LFP"]


def get_trialwise_detail_LFP_and_timing(
          dataset_name: str, 
          FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
    ):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        f"ARCHIVE_trialwise_LFP_and_Timing_{FEATURE_RANGE[0]}_{FEATURE_RANGE[1]}.npy"
    )
    if not os.path.exists(pkl_save_path):        
        all_LFP, all_spikes, all_node_name = [], [], []
        LFP_t = np.linspace(FEATURE_RANGE[0], FEATURE_RANGE[1], 1000) 
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is None:
                continue
            LFP_timeseries = grouping_timeseries([one_trial.potential.aspect(4).segment(*FEATURE_RANGE) 
                                                  for one_trial in puff_trials_500ms], _predefined_t=LFP_t)
            spiking_timing = [one_trial.potential.spikes.t * 1000 for one_trial in puff_trials_500ms]
            if LFP_timeseries.data_num < 5 or (LFP_timeseries.mean.min() == LFP_timeseries.mean.max()) or sum([len(one_trial) for one_trial in spiking_timing]) < 5:
                    continue
            
            all_spikes.append(spiking_timing)
            all_LFP.append(LFP_timeseries.raw_array)
            all_node_name.append(cell_session_node.coordinate)
        all_spikes = np.array(all_spikes, dtype=object)
        all_LFP = np.array(all_LFP, dtype=object)
        all_node_name = np.array(all_node_name, dtype=object)
        np.save(pkl_save_path, {
                "LFP": all_LFP, 
                "spikes": all_spikes, 
                "t": LFP_t,
                "node_name": all_node_name,
            })  # type: ignore
        logger.info("LFP saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("LFP loaded from " + pkl_save_path)
    return tmp_save["t"], tmp_save["LFP"], tmp_save["spikes"], tmp_save["node_name"]


def SummaryMetric_LFP(
        FEATURE_RANGE = (-0.02, 0.06),  # (min, max) s
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import matplotlib.patches as patches

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    y_offset = 1.2
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.tick_params(length=1, pad=1)
    for dataset_index, dataset_name in enumerate(("PYR_JUX", "PV_JUX", "SST_JUX", ) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        LFP_t, all_LFP = get_all_cellsession_LFP_mean(dataset_name, FEATURE_RANGE)
        oreo_plot(ax, AdvancedTimeSeries(t=LFP_t, 
                                        v=np.mean(all_LFP, axis=0) + dataset_index * y_offset, 
                                        variance=np.std(all_LFP, axis=0), 
                                        raw_array=all_LFP), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name], "lw": 1., "alpha": 0.9, "zorder": -dataset_index}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, "zorder": -dataset_index})
        ax.axhline(dataset_index * y_offset, color=COHORT_COLORS[dataset_name], 
                   linestyle='--', lw=1, alpha=0.5, zorder=-10)
    ax.axvspan(0, 0.5, alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
    ax.axvline(0, color='gray', linestyle='--', lw=0.5, zorder=-10)
    ax.spines[['right', 'top']].set_visible(False)
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("LFP [mV]")
            
    ax.set_xlim(-0.02, 0.06)
    ax.set_xticks([0/1000, 20/1000, 40/1000], ["0", "20", "40"])
    fig.set_size_inches(1.4, 1.4)

    save_path = path.join(get_saving_path(), "SummaryMetric_LFP.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


def get_all_cellsession_Waveform_mean(
          dataset_name: str, 
    ):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        "ARCHIVE_WAVEFORM_short.npy"
    )
    if not os.path.exists(pkl_save_path):        
        all_waveforms, all_node_name = [], []
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        for cell_session_node in dataset.select("cellsession"):
            potential_timeseries = cell_session_node.potential.aspect('raw')
            
            spike_timeseries = potential_timeseries.batch_segment(
                cell_session_node.potential.spikes.t, NARROW_SPIKE_RANGE_FOR_VISUALIZATION)
            if len(spike_timeseries) == 0:
                continue
            grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
            zscored_waveform = np.mean(numpy_kit.zscore(grouped_spike_timeseries.raw_array, axis=1), axis=0)
            all_waveforms.append(TimeSeries(v=zscored_waveform, t=grouped_spike_timeseries.t))
            all_node_name.append(str(cell_session_node.coordinate))
        all_waveforms = grouping_timeseries(all_waveforms)
        np.save(pkl_save_path, {
                "waveform": all_waveforms.raw_array, 
                "t": all_waveforms.t,
                "node_name": all_node_name,
            })  # type: ignore
        logger.info("Waveform saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("Waveform loaded from " + pkl_save_path)
    return tmp_save["t"], tmp_save["waveform"], tmp_save["node_name"]


def get_all_cellsession_type_specific_Waveform_mean(
          dataset_name: str, 
          spike_type: str,
    ):
    assert spike_type in ("early_spike", "sustained_spike", "regular_spike")
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        f"ARCHIVE_WAVEFORM_short_{spike_type}.npy"
    )
    if not os.path.exists(pkl_save_path):        
        all_waveforms, all_node_name = [], []
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        for cell_session_node in dataset.select("cellsession"):
            potential_timeseries = cell_session_node.potential.aspect('raw')
            
            spike_timeseries = potential_timeseries.batch_segment(
                cell_session_node.potential.spikes.filter(spike_type).t, NARROW_SPIKE_RANGE_FOR_VISUALIZATION)
            if len(spike_timeseries) == 0:
                continue
            grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
            zscored_waveform = np.mean(numpy_kit.zscore(grouped_spike_timeseries.raw_array, axis=1), axis=0)
            all_waveforms.append(TimeSeries(v=zscored_waveform, t=grouped_spike_timeseries.t))
            all_node_name.append(str(cell_session_node.coordinate))
        all_waveforms = grouping_timeseries(all_waveforms)
        np.save(pkl_save_path, {
                "waveform": all_waveforms.raw_array, 
                "t": all_waveforms.t,
                "node_name": all_node_name,
            })  # type: ignore
        logger.info("Waveform saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("Waveform loaded from " + pkl_save_path)
    return tmp_save["t"], tmp_save["waveform"], tmp_save["node_name"]


def SummaryMetric_Waveform(
):
    # plotting
    import matplotlib.pyplot as plt    

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for dataset_index, dataset_name in enumerate(("SST_JUX", "PV_JUX",  "PYR_JUX", ) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_BRIGHTER_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_BRIGHTER_COLORS[dataset_name]
        waveform_t, all_waveforms, _ = get_all_cellsession_Waveform_mean(dataset_name)
        baseline = np.mean(all_waveforms[:, waveform_t < -1/1000],)
        oreo_plot(ax, AdvancedTimeSeries(t=waveform_t * 1000, 
                                        v=np.mean(all_waveforms, axis=0) - baseline, 
                                        variance=np.std(all_waveforms, axis=0), 
                                        raw_array=all_waveforms), 0, 1, 
                    {"color": COHORT_BRIGHTER_COLORS[dataset_name], "lw": 1., "alpha": 0.9, "zorder": -dataset_index},
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.3, "zorder": -dataset_index})
        # ax.axhline(dataset_index * y_offset, color=COHORT_COLORS[dataset_name], 
        #            linestyle='--', lw=1, alpha=0.5, zorder=-10)
    ax.spines[['right', 'top', ]].set_visible(False)
    # ax.set_xlabel("Time [ms]")
    # ax.set_ylabel("Norm. Vm [a.u.]")
            
    fig.set_size_inches(1.5, 1.5)

    save_path = path.join(get_saving_path(), "SummaryMetric_Waveform.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



    fig, axs = plt.subplots(3, 3, figsize=(4, 4), 
                            sharey='row', sharex='col', constrained_layout=True)
    for dataset_index, dataset_name in enumerate(("SST_JUX", "PV_JUX",  "PYR_JUX", ) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_BRIGHTER_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_BRIGHTER_COLORS[dataset_name]
        for spike_type_id, spike_type in enumerate(("early_spike", "sustained_spike", "regular_spike")):
            waveform_t, all_waveforms, _ = get_all_cellsession_type_specific_Waveform_mean(dataset_name, spike_type)
            baseline = np.mean(all_waveforms[:, waveform_t < -1/1000],)
            oreo_plot(axs[dataset_index, spike_type_id], AdvancedTimeSeries(t=waveform_t * 1000, 
                                            v=np.mean(all_waveforms, axis=0) - baseline, 
                                            variance=np.std(all_waveforms, axis=0), 
                                            raw_array=all_waveforms), 0, 1, 
                        {"color": COHORT_BRIGHTER_COLORS[dataset_name], "lw": 1., "alpha": 0.9, "zorder": -dataset_index},
                        style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.3, "zorder": -dataset_index})
            # ax.axhline(dataset_index * y_offset, color=COHORT_COLORS[dataset_name], 
            #            linestyle='--', lw=1, alpha=0.5, zorder=-10)
            axs[0, spike_type_id].set_title(spike_type.replace("_", " ").title())
    for ax in axs.flatten():
        ax.spines[['right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        # ax.set_yticks([])   
    
    # ax.set_xlabel("Time [ms]")
    # ax.set_ylabel("Norm. Vm [a.u.]")
            
    save_path = path.join(get_saving_path(), "SummaryMetric_Waveform_Typespecific.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)
