from collections import defaultdict
import os
import os.path as path
from typing import Optional

from scipy import stats

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import AdvancedTimeSeries, grouping_events_histogram, grouping_timeseries
import kitchen.plotter.color_scheme as color_scheme
from kitchen.plotter.plotting_manual import PlotManual
import kitchen.plotter.style_dicts as style_dicts
from kitchen.configs import routing
from kitchen.plotter.macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros_Settings import COHORT_COLORS

import logging
import numpy as np

from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT
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
          BINSIZE = 10/1000,  # s
          FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
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
        all_spikes_histogram, all_baseline_fr = [], []
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
        all_baseline_fr = np.array(all_baseline_fr)
        all_spikes_histogram = np.stack(all_spikes_histogram, axis=0)
        np.save(pkl_save_path, {
            "spike_histogram": all_spikes_histogram, 
            "baseline_fr": all_baseline_fr,
            "bin_centers": bin_centers,
            })  # type: ignore
        logger.info("PSTH saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("PSTH loaded from " + pkl_save_path)
    return tmp_save["bin_centers"], tmp_save["spike_histogram"], tmp_save["baseline_fr"]


def SummaryMetric_PSTH(
        BINSIZE = 10/1000,  # s
        FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import matplotlib.patches as patches

    plt.rcParams["font.family"] = "Arial"
    x_offset, y_offset = 0.1, 20
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for dataset_index, dataset_name in enumerate(("PYR_JUX", "SST_JUX", "PV_JUX", ) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        bin_centers, all_spikes_histogram, _ = get_all_cellsession_PSTH_mean(dataset_name, BINSIZE, FEATURE_RANGE)
        oreo_plot(ax, AdvancedTimeSeries(t=bin_centers + dataset_index * x_offset, 
                                        v=np.mean(all_spikes_histogram, axis=0) + dataset_index * y_offset, 
                                        variance=np.std(all_spikes_histogram, axis=0), 
                                        raw_array=all_spikes_histogram), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name], "lw": 1.5, "zorder": -dataset_index}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.7, "zorder": -dataset_index})
        ax.axhline(dataset_index * y_offset, color=COHORT_COLORS[dataset_name], 
                   linestyle='--', lw=1, alpha=0.5, zorder=-10)
    parallelogram = patches.Polygon(xy=list(zip([0, 0.5, 0.7, 0.2], [0, 0, 40, 40])), 
                                    alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)

    ax.add_patch(parallelogram)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Firing Rate [Hz]")
            
    # ax.set_xlim(-0.25, 0.75)
    ax.set_ylim(-5, None)
    fig.set_size_inches(3.5, 2.5)

    save_path = path.join(get_saving_path(), "SummaryMetric_PSTH.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


def PlainHeatmap_PSTH(
        BINSIZE = 10/1000,  # s
        FEATURE_RANGE = (-0.5, 1.0),  # (min, max) s
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import SymLogNorm

    plt.rcParams["font.family"] = "Arial"
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    for ax, dataset_name in zip(axs, ( "SST_JUX", "PV_JUX", "PYR_JUX",) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        bin_centers, all_spikes_histogram, _ = get_all_cellsession_PSTH_mean(dataset_name, BINSIZE, FEATURE_RANGE)
        mask = (bin_centers > 0) & (bin_centers < 0.5)
        sorted_order = np.argsort(np.mean(all_spikes_histogram[:, mask], axis=1))
        sns.heatmap(all_spikes_histogram[sorted_order],  
                    ax=ax, cbar=True, 
                    cmap='YlOrBr', norm=SymLogNorm(linthresh=10., vmin=0, vmax=150),)
        
        desired_ticks = [0, 0.5]
        tick_positions = [np.argmin(np.abs(bin_centers - tick)) for tick in desired_ticks]
        ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

        ax.invert_yaxis()
        ax.set_xlabel("Time [s]")
        ax.set_yticks([0.5, len(all_spikes_histogram)-0.5], 
                      ["1", f"{len(all_spikes_histogram)}"], rotation=0)
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
    fig.set_size_inches(14, 3)

    save_path = path.join(get_saving_path(), "PlainHeatmap_PSTH.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
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


def SummaryMetric_LFP(
        FEATURE_RANGE = (-0.02, 0.06),  # (min, max) s
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import matplotlib.patches as patches

    plt.rcParams["font.family"] = "Arial"
    x_offset, y_offset = 0.05, 0.5
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for dataset_index, dataset_name in enumerate(("PYR_JUX", "SST_JUX", "PV_JUX", ) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        LFP_t, all_LFP = get_all_cellsession_LFP_mean(dataset_name, FEATURE_RANGE)
        oreo_plot(ax, AdvancedTimeSeries(t=LFP_t + dataset_index * x_offset, 
                                        v=np.mean(all_LFP, axis=0) + dataset_index * y_offset, 
                                        variance=np.std(all_LFP, axis=0), 
                                        raw_array=all_LFP), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name], "lw": 1.5, "zorder": -dataset_index}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.7, "zorder": -dataset_index})
        ax.axhline(dataset_index * y_offset, color=COHORT_COLORS[dataset_name], 
                   linestyle='--', lw=1, alpha=0.5, zorder=-10)
    parallelogram = patches.Polygon(xy=list(zip([0, 0.5, 0.6, 0.1], [0, 0, 1, 1])), 
                                    alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)

    ax.add_patch(parallelogram)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("LFP [mV]")
            
    ax.set_xlim(-0.02, 0.16)
    # ax.set_ylim(-5, 155)
    fig.set_size_inches(3.5, 2.5)

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
        all_waveforms = []
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        for cell_session_node in dataset.select("cellsession"):
            potential_timeseries = cell_session_node.potential.aspect('raw')
            
            spike_timeseries = potential_timeseries.batch_segment(
                cell_session_node.potential.spikes.t, (-0.001, 0.002))
            if len(spike_timeseries) == 0:
                continue
            grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
            zscored_waveform = np.mean(numpy_kit.zscore(grouped_spike_timeseries.raw_array, axis=1), axis=0)
            all_waveforms.append(TimeSeries(v=zscored_waveform, t=grouped_spike_timeseries.t))
        all_waveforms = grouping_timeseries(all_waveforms)
        np.save(pkl_save_path, {
                "waveform": all_waveforms.raw_array, 
                "t": all_waveforms.t,
            })  # type: ignore
        logger.info("Waveform saved to " + pkl_save_path)
    tmp_save = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("Waveform loaded from " + pkl_save_path)
    return tmp_save["t"], tmp_save["waveform"]


def SummaryMetric_Waveform(
):
    

    # plotting
    import matplotlib.pyplot as plt    
    import matplotlib.patches as patches

    plt.rcParams["font.family"] = "Arial"
    x_offset, y_offset = 1, 1
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for dataset_index, dataset_name in enumerate(("PYR_JUX", "SST_JUX", "PV_JUX", ) ):        
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        waveform_t, all_waveforms = get_all_cellsession_Waveform_mean(dataset_name)
        oreo_plot(ax, AdvancedTimeSeries(t=waveform_t * 1000 + dataset_index * x_offset, 
                                        v=np.mean(all_waveforms, axis=0) + dataset_index * y_offset, 
                                        variance=np.std(all_waveforms, axis=0), 
                                        raw_array=all_waveforms), 0, 1, 
                    {"color": COHORT_COLORS[dataset_name], "lw": 1.5, "zorder": -dataset_index}, 
                    style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.7, "zorder": -dataset_index})
        ax.axhline(dataset_index * y_offset, color=COHORT_COLORS[dataset_name], 
                   linestyle='--', lw=1, alpha=0.5, zorder=-10)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.set_xlabel("Time [ms]")
    # ax.set_ylabel("Norm. Vm [a.u.]")
            
    fig.set_size_inches(2., 2.)

    save_path = path.join(get_saving_path(), "SummaryMetric_Waveform.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)
