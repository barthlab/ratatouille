import logging
from os import path
import os
from typing import Dict
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from kitchen.configs.naming import get_node_name
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator import select_trial_rules
from kitchen.operator.grouping import grouping_events_histogram, grouping_events_rate, grouping_timeseries
from kitchen.operator.select_trial_rules import PREDEFINED_PASSIVEPUFF_RULES
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.color_scheme import EARLY_SPIKE_COLOR, PUFF_COLOR, SPIKE_COLOR_SCHEME, SUSTAINED_SPIKE_COLOR
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import FLAT_X_INCHES, FLUORESCENCE_RATIO, POTENTIAL_RATIO, UNIT_Y_INCHES
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_potential, unit_plot_potential_conv
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, SPIKE_ANNOTATION_EARLY_WINDOW, WC_CONVERT_FLAG
from kitchen.structure.hierarchical_data_structure import DataSet, Mice, Session
from kitchen.structure.neural_data_structure import Events, TimeSeries
from kitchen.utils.numpy_kit import zscore



plt.rcParams["font.family"] = "Arial"


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)



FEATURE_RANGE = (-1, 1.5)
BINSIZE = 10/1000
alignment_events = ("VerticalPuffOn",)
plot_manual_spike4Hz = PlotManual(potential=4,)
plot_manual_spike300Hz = PlotManual(potential=300.)
COHORT_COLORS = {
    "SST_JUX": "#ffa000",
    "SST_WC": "#CF8200",
    "PV_JUX": "#ff0000",
    "PYR_JUX": "#0000FF",
}

def calculate_spike_shape_features(spike_waveforms, sampling_rate_hz):
    if not isinstance(spike_waveforms, np.ndarray) or spike_waveforms.ndim != 2:
        raise ValueError
    if not isinstance(sampling_rate_hz, (int, float)) or sampling_rate_hz <= 0:
        raise ValueError

    n_spikes, n_samples = spike_waveforms.shape
    peak_index = int(n_samples/2)
    
    ms_per_sample = 1000.0 / sampling_rate_hz

    spike_widths_ms = np.zeros(n_spikes)
    spike_asymmetries = np.zeros(n_spikes)

    for i in range(n_spikes):
        waveform = spike_waveforms[i, :]

        peak_amplitude = waveform[peak_index]

        post_peak_waveform = waveform[peak_index:]
        trough_index_relative = np.argmin(post_peak_waveform)
        trough_index = peak_index + trough_index_relative
        trough_amplitude = waveform[trough_index] 

        width_in_samples = trough_index - peak_index
        spike_widths_ms[i] = width_in_samples * ms_per_sample

        post_mag = float(peak_amplitude - trough_amplitude)

        pre_peak_waveform = waveform[:peak_index]
        pre_peak_min_amplitude = np.min(pre_peak_waveform)
        pre_mag = float(peak_amplitude - pre_peak_min_amplitude)
        
        denominator = pre_mag + post_mag
        if denominator > 1e-9: 
             spike_asymmetries[i] = (post_mag - pre_mag) / denominator
        else:
             spike_asymmetries[i] = np.nan

    return spike_widths_ms, spike_asymmetries



def calculate_cv2(spike_times: np.ndarray) -> float:
    """
    Calculates the CV2 for a sequence of spike times.

    CV2 is a measure of local firing regularity, defined as the average of
    2 * |ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i) for all adjacent
    inter-spike intervals (ISIs). It is less sensitive to slow changes in firing rate
    than the standard coefficient of variation (CV).

    Args:
        spike_times (np.ndarray): A 1D NumPy array of spike times in seconds.

    Returns:
        float: The calculated CV2 value. Returns np.nan if there are fewer than 3 spikes.
    """
    # CV2 requires at least 2 ISIs, which means at least 3 spikes.
    if spike_times.size < 3:
        return np.nan
        
    # Calculate inter-spike intervals (ISIs)
    isis = np.diff(spike_times)
    
    # Calculate the CV2 for each adjacent pair of ISIs
    # Numerator: 2 * |ISI_{i+1} - ISI_i|
    # Denominator: ISI_{i+1} + ISI_i
    diff_isis = np.abs(np.diff(isis))
    sum_isis = isis[:-1] + isis[1:]
    
    # Avoid division by zero if two ISIs are zero (highly unlikely with real data)
    # but good practice for robustness.
    cv2_pairs = 2 * diff_isis / sum_isis
    
    # The final CV2 is the mean of these values
    return np.mean(cv2_pairs)


def calculate_autocorrelogram_rise_time(spike_times, bin_size=0.001, max_lag=10):
    if not isinstance(spike_times, np.ndarray) or spike_times.ndim != 1:
        raise TypeError("spike_times must be a 1D NumPy array.")
    
    if len(spike_times) < 2:
        print("Warning: Not enough spikes to compute an autocorrelogram. Returning NaNs.")
        return np.nan

    # Calculate all inter-spike intervals (ISIs) up to max_lag
    lags = []
    for i in range(len(spike_times)):
        for j in range(i + 1, len(spike_times)):
            lag = spike_times[j] - spike_times[i]
            # Since spike_times is sorted, we can break early
            if lag > max_lag:
                break
            lags.append(lag)

    if not lags:
        print(f"Warning: No spike pairs found within the max_lag of {max_lag}s.")
        return np.nan

    # Create the histogram (the autocorrelogram)
    bins = np.arange(bin_size, max_lag + bin_size, bin_size)
    counts, bin_edges = np.histogram(lags, bins=bins)

    peak_bin_index = np.argmax(counts)

    rise_time = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0

    return rise_time



def extract_features(dataset: DataSet):
    session_nodes = dataset.select("cellsession", 
                                     _self=lambda x: len(dataset.subtree(x).select(
                                         "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,
                                         _empty_warning=False,
                                         )) > 0)
    all_session_features = []
    for row_idx, session_node in enumerate(session_nodes):
        subtree = dataset.subtree(session_node)
        node_name = get_node_name(session_node)

        # filter out trial types that cannot be plotted
        puff_trials_500ms = subtree.select("trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52, _empty_warning=False,)

        alignment_events = ("VerticalPuffOn",)

        if len(puff_trials_500ms) == 0:
            logger.debug(f"Cannot plot raster plot for {node_name}: no puff trials found. Skip...")
            continue

        puff_trials_500ms = sync_nodes(puff_trials_500ms, alignment_events, plot_manual_spike300Hz)
        if len(puff_trials_500ms) == 0:
            logger.debug(f"Cannot plot raster plot for {node_name}: no puff trials found after syncing. Skip...")
            continue

        LFP_peak = [one_trial.potential.aspect(4).segment(*SPIKE_ANNOTATION_EARLY_WINDOW).v.min() for one_trial in puff_trials_500ms]
        early_spikes = [one_trial.potential.spikes.filter("early_spike") for one_trial in puff_trials_500ms]
        early_spike_median_time = [np.median(early_spike.t) for early_spike in early_spikes if len(early_spike) > 0]
        early_spike_num = [len(early_spike) for early_spike in early_spikes]
        sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike") for one_trial in puff_trials_500ms]
        sustained_spike_median_time = [np.median(sustained_spike.t) for sustained_spike in sustained_spikes if len(sustained_spike) > 0]
        sustained_spike_num = [len(sustained_spike) for sustained_spike in sustained_spikes]

        all_spikes = session_node.potential.spikes.t
        potential_timeseries = session_node.potential.aspect('raw')
        spike_timeseries = potential_timeseries.batch_segment(all_spikes, CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
        if len(spike_timeseries) > 0:
            grouped_spike_timeseries = grouping_timeseries(
                [TimeSeries(v=zscore(ts.v), t=ts.t) for ts in spike_timeseries], 
                interp_method="linear")
            width_ms, asymmetry = calculate_spike_shape_features(grouped_spike_timeseries.raw_array, potential_timeseries.fs)
            CV2 = calculate_cv2(all_spikes)
            acg_rise_time_s = calculate_autocorrelogram_rise_time(all_spikes)
        else:
            width_ms = np.nan
            asymmetry = np.nan
            CV2 = np.nan
            acg_rise_time_s = np.nan
        def collapse_list(tmp_list):
            if isinstance(tmp_list, list | np.ndarray):
                return np.nanmean(tmp_list) if len(tmp_list) > 0 else np.nan
            else:
                return tmp_list
        all_session_features.append({
            "node_name": str(session_node.coordinate),
            "LFP_peak": collapse_list(LFP_peak),
            "early_spike_median_time": collapse_list(early_spike_median_time),
            "early_spike_num": collapse_list(early_spike_num),
            "sustained_spike_median_time": collapse_list(sustained_spike_median_time),
            "sustained_spike_num": collapse_list(sustained_spike_num),
            "width_ms": collapse_list(width_ms),
            "asymmetry": collapse_list(asymmetry),
            "CV2": collapse_list(CV2),
            "acg_rise_time_s": collapse_list(acg_rise_time_s),
            "cohort": dataset.name,
        })
    return all_session_features


def LFP_extraction(dataset: DataSet):
    session_nodes = dataset.select("cellsession", 
                                     _self=lambda x: len(dataset.subtree(x).select(
                                         "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,
                                         _empty_warning=False,
                                         )) > 0)
    all_session_LFP = []
    for row_idx, session_node in enumerate(session_nodes):
        subtree = dataset.subtree(session_node)
        node_name = get_node_name(session_node)

        # filter out trial types that cannot be plotted
        puff_trials_500ms = subtree.select("trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52, _empty_warning=False,)

        alignment_events = ("VerticalPuffOn",)

        if len(puff_trials_500ms) == 0:
            logger.debug(f"Cannot plot raster plot for {node_name}: no puff trials found. Skip...")
            continue

        puff_trials_500ms = sync_nodes(puff_trials_500ms, alignment_events, plot_manual_spike300Hz)
        if len(puff_trials_500ms) == 0:
            logger.debug(f"Cannot plot raster plot for {node_name}: no puff trials found after syncing. Skip...")
            continue
        LFP_timeseries = [one_trial.potential.aspect(4) for one_trial in puff_trials_500ms]
        grouped_LFP_timeseries = grouping_timeseries(LFP_timeseries, interp_method="linear")
        all_session_LFP.append(grouped_LFP_timeseries.mean_ts)
    return all_session_LFP


def feature_summary():    
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })
    all_cell_feature_summary = []
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):        
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        all_cell_feature_summary.extend(extract_features(dataset,))
    all_cell_feature_summary = pd.DataFrame(all_cell_feature_summary)
    all_cell_feature_summary.to_csv("all_cell_feature_summary.csv", index=False)

    for feature_name in all_cell_feature_summary.columns[1:-1]:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), constrained_layout=True)
        sns.boxplot(data=all_cell_feature_summary, y="cohort", x=feature_name, ax=ax, 
                    hue="cohort", palette=COHORT_COLORS, orient="h", zorder=-10,
                    order=["PV_JUX", "SST_JUX", "PYR_JUX", ], showfliers=False,)
        sns.swarmplot(data=all_cell_feature_summary, y="cohort", x=feature_name, ax=ax, 
                      order=["PV_JUX", "SST_JUX", "PYR_JUX", ],
                      color="black", alpha=0.7, size=2, orient="h", )
        ax.set_xlabel(feature_name)
        ax.set_ylabel("")
        if feature_name == "acg_rise_time_s":
            ax.set_xscale("log")
        ax.spines[['right', 'top']].set_visible(False)
        fig.savefig(os.path.join("tmp_feature_summary", f"{feature_name}.png"), dpi=900)
        plt.close(fig)


def LFP_summary():    
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):        
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        LFP_timeseries = LFP_extraction(dataset,)
        group_LFP = grouping_timeseries(LFP_timeseries, interp_method="linear")
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), constrained_layout=True)
        oreo_plot(ax, group_LFP, 0, 1, {"color": COHORT_COLORS[dataset_name], "lw": 0.5, }, FILL_BETWEEN_STYLE)
        for individual_LFP in LFP_timeseries:
            ax.plot(individual_LFP.t, individual_LFP.v, color=COHORT_COLORS[dataset_name], alpha=0.05, lw=0.3)
        ax.set_title(dataset_name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Vm [4Hz mV]")
        ax.set_xlim(-0.1, 0.15)
        ax.axvspan(0, 0.5, alpha=0.3, color=PUFF_COLOR, lw=0, zorder=-10)
        ax.spines[['right', 'top']].set_visible(False)
        fig.savefig(os.path.join("tmp_feature_summary", f"LFP_{dataset_name}.png"), dpi=900)
        plt.close(fig)

    
def main():
    feature_summary()
    # LFP_summary()

if __name__ == "__main__":
    main()