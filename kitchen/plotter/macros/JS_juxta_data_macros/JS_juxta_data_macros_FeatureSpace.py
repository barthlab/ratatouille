import logging
from matplotlib.gridspec import GridSpec
import numpy as np
import os.path as path
import os
import pandas as pd
import _pickle as pkl

from kitchen.calculator import calculate_physiology
from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import AdvancedTimeSeries, grouping_timeseries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.utils.ordering_kit import linkage_order
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_BRIGHTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_100ms_puff_trials, get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_PSTH_mean, get_all_cellsession_Waveform_mean, get_saving_path
from kitchen.settings.potential import NARROW_SPIKE_RANGE_FOR_VISUALIZATION, SPIKE_ANNOTATION_EARLY_WINDOW
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)



def get_all_physiology_features(
        dataset_name: str,
):
               
    def collapse_mean(tmp_list):
        if isinstance(tmp_list, list | np.ndarray):
            return np.nanmean(tmp_list) if len(tmp_list) > 0 else np.nan
        else:
            return tmp_list
        
    def collapse_std(tmp_list):
        if isinstance(tmp_list, list | np.ndarray):
            return np.nanstd(tmp_list) if len(tmp_list) > 0 else np.nan
        else:
            return tmp_list

    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        "ARCHIVE_all_physiology_features.csv"
    )
    if not os.path.exists(pkl_save_path):     
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        all_physiology_features = []
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is None:
                continue

            LFP_peak = [one_trial.potential.aspect(4).segment(*SPIKE_ANNOTATION_EARLY_WINDOW).v.min() for one_trial in puff_trials_500ms]

            early_spikes = [one_trial.potential.spikes.filter("early_spike") for one_trial in puff_trials_500ms]
            early_spike_median_time = [np.median(early_spike.t) for early_spike in early_spikes if len(early_spike) > 0]
            early_spike_first_time = [np.min(early_spike.t) for early_spike in early_spikes if len(early_spike) > 0]
            early_spike_num = [len(early_spike) for early_spike in early_spikes]

            sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike") for one_trial in puff_trials_500ms]
            sustained_spike_median_time = [np.median(sustained_spike.t) for sustained_spike in sustained_spikes if len(sustained_spike) > 0]
            sustained_spike_num = [len(sustained_spike) for sustained_spike in sustained_spikes]
            sustained_spike_first_time = [np.min(sustained_spike.t) for sustained_spike in sustained_spikes if len(sustained_spike) > 0]

            pre_stim_spont_spike_num = [np.sum((-1 < one_trial.potential.spikes.t) & (one_trial.potential.spikes.t < 0)) 
                                        for one_trial in puff_trials_500ms if len(one_trial.potential.spikes) > 0]
            
            all_spikes = cell_session_node.potential.spikes.t
            potential_timeseries = cell_session_node.potential.aspect('raw')
            spike_timeseries = potential_timeseries.batch_segment(all_spikes, NARROW_SPIKE_RANGE_FOR_VISUALIZATION)
 
            if len(spike_timeseries) > 0:
                
                grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
                zscored_waveform = numpy_kit.zscore(grouped_spike_timeseries.raw_array, axis=1)
                
                width_ms, asymmetry_au = calculate_physiology.calculate_spike_width_and_asymmetry(zscored_waveform, grouped_spike_timeseries.fs)
                cv2_au = calculate_physiology.calculate_cv2(all_spikes)
                acg_rise_time_ms = calculate_physiology.calculate_autocorrelogram_rise_time(all_spikes)
            else:
                width_ms = np.nan
                asymmetry_au = np.nan
                cv2_au = np.nan
                acg_rise_time_ms = np.nan
            
            all_physiology_features.append({
                "LFP peak\nAvg [mV]": collapse_mean(LFP_peak),
                "LFP peak\nStd [mV]": collapse_std(LFP_peak),

                "Early Spike Median Timing\nAvg [ms]": collapse_mean(early_spike_median_time) * 1000,
                "Early Spike Median Timing\nStd [ms]": collapse_std(early_spike_median_time) * 1000,
                "Early Spike Num\nAvg": collapse_mean(early_spike_num),
                "Early Spike Num\nStd": collapse_std(early_spike_num),
                "Early Spike First Timing\nAvg [ms]": collapse_mean(early_spike_first_time) * 1000,
                "Early Spike First Timing\nStd [ms]": collapse_std(early_spike_first_time) * 1000,

                "Sustained Spike Median Timing\nAvg [ms]": collapse_mean(sustained_spike_median_time) * 1000,
                "Sustained Spike Median Timing\nStd [ms]": collapse_std(sustained_spike_median_time) * 1000,
                "Sustained Spike Num\nAvg": collapse_mean(sustained_spike_num),
                "Sustained Spike Num\nStd": collapse_std(sustained_spike_num),
                "Sustained Spike First Timing\nAvg [ms]": collapse_mean(sustained_spike_first_time) * 1000,
                "Sustained Spike First Timing\nStd [ms]": collapse_std(sustained_spike_first_time) * 1000,

                "Spont. FR [Hz]": collapse_mean(pre_stim_spont_spike_num),
                "Spont. FR\nStd [Hz]": collapse_std(pre_stim_spont_spike_num),

                "Spike Width [ms]": collapse_mean(width_ms),
                "Spike Width\nStd [ms]": collapse_std(width_ms),
                "Spike Asymmetry [a.u.]": collapse_mean(asymmetry_au),
                "Spike Asymmetry\nStd [a.u.]": collapse_std(asymmetry_au),
                "CV2 [a.u.]": cv2_au,
                "ACG Rise Time [ms]": acg_rise_time_ms,

                "node_name": str(cell_session_node.coordinate),
                "cohort_name": dataset_name,
            })
            
        all_physiology_features = pd.DataFrame(all_physiology_features)
        all_physiology_features.to_csv(pkl_save_path, index=False)  
        logger.info("Physiology features saved to " + pkl_save_path)

    all_physiology_features = pd.read_csv(pkl_save_path)
    logger.info("Physiology features loaded from " + pkl_save_path)
    return all_physiology_features



def get_variability_physiology_features(
        dataset_name: str,
):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        "ARCHIVE_variability_physiology_features.pkl"
    )
    if not os.path.exists(pkl_save_path):     
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        all_physiology_features = []
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is None:
                continue

            LFP_peak = [one_trial.potential.aspect(4).segment(*SPIKE_ANNOTATION_EARLY_WINDOW).v.min() for one_trial in puff_trials_500ms]

            early_spikes = [one_trial.potential.spikes.filter("early_spike").t * 1000 for one_trial in puff_trials_500ms]
 
            sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike").t * 1000 for one_trial in puff_trials_500ms]
           
            pre_stim_spont_spike_num = [np.sum((-1 < one_trial.potential.spikes.t) & (one_trial.potential.spikes.t < 0)) 
                                        if len(one_trial.potential.spikes) > 0 else 0
                                        for one_trial in puff_trials_500ms]
            
            all_physiology_features.append({
                "LFP peak [mV]": LFP_peak,
                "Early Spike [ms]": early_spikes,
                "Sustained Spike [ms]": sustained_spikes,
                "PreStim Spont. FR [Hz]": pre_stim_spont_spike_num,
                "node_name": str(cell_session_node.coordinate),
                "cohort_name": dataset_name,
            })

        pkl.dump(all_physiology_features, open(pkl_save_path, "wb"))
        logger.info("Physiology features saved to " + pkl_save_path)

    all_physiology_features = pkl.load(open(pkl_save_path, "rb"))
    logger.info("Physiology features loaded from " + pkl_save_path)
    return all_physiology_features




def get_detailed_spike_timing_features(
        dataset_name: str,
):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        "ARCHIVE_detailed_spike_timing.npy"
    )
    if not os.path.exists(pkl_save_path):     
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        all_spikes, node_names, cohort_names = [], [], []
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is None:
                continue
                        
            all_spikes.append(cell_session_node.potential.spikes.t)
            node_names.append(str(cell_session_node.coordinate))
            cohort_names.append(dataset_name)
        all_spikes = np.array(all_spikes, dtype=object)
        all_node_name = np.array(node_names)
        all_cohort_name = np.array(cohort_names)

        np.save(pkl_save_path, {
            "all_spikes": all_spikes,
            "all_node_name": all_node_name,
            "all_cohort_name": all_cohort_name
        }) # type: ignore
        logger.info("Detailed spiking features saved to " + pkl_save_path)

    detailed_spiking_features = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("Detailed spiking features loaded from " + pkl_save_path)
    return detailed_spiking_features



def get_trial_wise_spike_timing_features(
        dataset_name: str,
):
    pkl_save_path = routing.robust_path_join(
        routing.DATA_PATH,
        "PassivePuff_JuxtaCellular_FromJS_202509",
        dataset_name,
        "ARCHIVE_trial_wise_spike_timing.npy"
    )
    if not os.path.exists(pkl_save_path):     
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                                recipe="default_ephys", name=dataset_name)
        all_spikes, node_names, cohort_names, trial_types = [], [], [], []
        for cell_session_node in dataset.select("cellsession"):
            puff_trials_500ms = get_500ms_puff_trials(cell_session_node, dataset)
            if puff_trials_500ms is not None:
                all_spikes.append([one_trial.potential.spikes.t * 1000 for one_trial in puff_trials_500ms])
                node_names.append(str(cell_session_node.coordinate))
                cohort_names.append(dataset_name)
                trial_types.append("500msPuff")

            puff_trials_100ms = get_100ms_puff_trials(cell_session_node, dataset)
            if puff_trials_100ms is not None:
                all_spikes.append([one_trial.potential.spikes.t * 1000 for one_trial in puff_trials_100ms])
                node_names.append(str(cell_session_node.coordinate))
                cohort_names.append(dataset_name)
                trial_types.append("100msPuff")
            
        all_spikes = np.array(all_spikes, dtype=object)
        all_node_name = np.array(node_names)
        all_cohort_name = np.array(cohort_names)
        all_trial_type = np.array(trial_types)

        np.save(pkl_save_path, {
            "all_spikes": all_spikes,
            "all_node_name": all_node_name,
            "all_cohort_name": all_cohort_name,
            "all_trial_type": all_trial_type
        }) # type: ignore
        logger.info("Trial-wise spiking features saved to " + pkl_save_path)

    trial_wise_spiking_features = np.load(pkl_save_path, allow_pickle=True).item()
    logger.info("Trial-wise spiking features loaded from " + pkl_save_path)
    return trial_wise_spiking_features




def VisualizeACG():

    pyr_jux_spike_timing = get_detailed_spike_timing_features("PYR_JUX")
    sst_jux_spike_timing = get_detailed_spike_timing_features("SST_JUX")
    pv_jux_spike_timing = get_detailed_spike_timing_features("PV_JUX")

    import matplotlib.pyplot as plt    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.1), constrained_layout=True)

    for ax_id, cohort_name, spike_timings in zip([0, 1, 2], ["SST_JUX", "PV_JUX", "PYR_JUX"], 
                                                 [sst_jux_spike_timing, pv_jux_spike_timing, pyr_jux_spike_timing]):
        all_acg = []
        for spike_times in spike_timings["all_spikes"]:
            acg_reslt = calculate_physiology.calculate_autocorrelogram(spike_times)
            if np.isnan(acg_reslt).any():
                continue
            acg, acg_t = acg_reslt
            all_acg.append(acg)
        oreo_plot(ax, AdvancedTimeSeries(t=acg_t, 
                                                v=np.mean(all_acg, axis=0), 
                                                variance=np.std(all_acg, axis=0)/np.sqrt(len(all_acg)), 
                                                raw_array=np.array(all_acg)), 0, 1, 
                            {"color": COHORT_BRIGHTER_COLORS[cohort_name], "lw": 1., "alpha": 0.9, "zorder": -ax_id},
                            style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.3, "zorder": -ax_id})

        ax.tick_params(which='both', length=1, pad=1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.001, 1)
        ax.set_xticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
        ax.set_ylim(0.01, 1000)
        ax.set_yticks([0.01, 1, 100])
        plt.grid(axis='x', color='gray', linestyle='-', lw=0.2, which='both')

    save_path = path.join(get_saving_path(), "FeatureSpace", "ACG_ALL.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def Visualize_ISI_distribution(annotate: bool = True):
    
    pyr_jux_spike_timing = get_detailed_spike_timing_features("PYR_JUX")
    sst_jux_spike_timing = get_detailed_spike_timing_features("SST_JUX")
    pv_jux_spike_timing = get_detailed_spike_timing_features("PV_JUX")

    import matplotlib.pyplot as plt    
    import seaborn as sns
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    fig, ax = plt.subplots(1, 1, figsize=(1., 1.2), constrained_layout=True)
    ax.tick_params(length=1, pad=1)
    annotate_data = {}
    for ax_id, cohort_name, spike_timings in zip([0, 1, 2], ["SST_JUX", "PV_JUX", "PYR_JUX"], 
                                                 [sst_jux_spike_timing, pv_jux_spike_timing, pyr_jux_spike_timing]):

        all_cv2, all_isis = [], []
        for spike_times in spike_timings["all_spikes"]:            
            isis = np.diff(spike_times)
            all_isis.append(isis)
            # diff_isis = np.abs(np.diff(isis))  # Numerator: 2 * |ISI_{i+1} - ISI_i|
            # sum_isis = isis[:-1] + isis[1:]  # Denominator: ISI_{i+1} + ISI_i
            
            # cv2_pairs = 2 * diff_isis / sum_isis
            # all_cv2.append(cv2_pairs)
        # all_cv2 = np.concatenate(all_cv2)
        all_isis = np.concatenate(all_isis)
        sns.kdeplot(all_isis, ax=ax, color=COHORT_BRIGHTER_COLORS[cohort_name], 
                    log_scale=True, lw=1, fill=False, bw_adjust=.2,)
        annotate_data[cohort_name] = all_isis

    ax.set_xlabel(r'Inter-spike interval (s)', labelpad=1)
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel("")

    save_path = path.join(get_saving_path(), "FeatureSpace", f"cv2_distribution{'_annotated' if annotate else ''}.png")
    if annotate:
        basic_ttest.stats_pairwise_ks(annotate_data, ax=ax,)
    default_exit_save(fig, save_path, _transparent=True)



def get_weight_tuple_Physiology_Fingerprint(with_spont_FR: bool = True, _drop_ACG_rise_time: bool = True, 
                                            _log_ACG_rise_time: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    wanted_columns = [
        "Spike Width [ms]",
        "Spike Asymmetry [a.u.]",
        "CV2 [a.u.]",
        "ACG Rise Time [ms]",
        "node_name",
        "cohort_name",
    ]

    if with_spont_FR:
        wanted_columns.insert(0, "Spont. FR [Hz]")

    if _drop_ACG_rise_time:
        wanted_columns.remove("ACG Rise Time [ms]")        
        wanted_df = combined_physiology[wanted_columns].dropna()
    else:        
        wanted_df = combined_physiology[wanted_columns].dropna()
        if _log_ACG_rise_time:
            wanted_df["ACG Rise Time [ms]"] = np.log(wanted_df["ACG Rise Time [ms]"])

    weight_matrix = wanted_df[wanted_columns[:-2]].to_numpy()
    return weight_matrix, wanted_df["node_name"].to_numpy(), wanted_df["cohort_name"].to_numpy()


def get_combined_variability_physiology() -> list[dict]:
    all_physiology_features = []
    all_physiology_features.extend(get_variability_physiology_features("SST_WC"))
    all_physiology_features.extend(get_variability_physiology_features("SST_JUX"))
    all_physiology_features.extend(get_variability_physiology_features("PV_JUX"))
    all_physiology_features.extend(get_variability_physiology_features("PYR_JUX"))
    all_physiology_features.extend(get_variability_physiology_features("PYR_WC"))
    return all_physiology_features

def get_weight_tuple_Variability_Physiology() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    var_physiology_list = get_combined_variability_physiology()
    node_names, cohort_names = [], []
    spont_msrsd, early_msrsd, sustained_msrsd = [], [], []
    
    for data_dict in var_physiology_list:
        single_cell_early_spike_ts = data_dict["Early Spike [ms]"]
        early_spike_num = [len(early_spike_t) for early_spike_t in single_cell_early_spike_ts]

        single_cell_sustained_spike_ts = data_dict["Sustained Spike [ms]"]
        sustained_spike_num = [len(sustained_spike_t) for sustained_spike_t in single_cell_sustained_spike_ts]

        spont_spike_num = data_dict["PreStim Spont. FR [Hz]"]

        assert len(spont_spike_num) == len(early_spike_num) == len(sustained_spike_num),\
              "Mismatch in spike counts: spont={}, early={}, sustained={}".format(len(spont_spike_num), len(early_spike_num), len(sustained_spike_num))
        if len(spont_spike_num) <= 5:
            continue

        early_fr = [(single_trial_early_spike_num / (SPIKE_ANNOTATION_EARLY_WINDOW[1] - SPIKE_ANNOTATION_EARLY_WINDOW[0]))
                    for single_trial_early_spike_num, spont_fr in zip(early_spike_num, spont_spike_num)]

        sustained_fr = [(single_trial_sustained_spike_num / (0.5 - SPIKE_ANNOTATION_EARLY_WINDOW[1]))
                        for single_trial_sustained_spike_num, spont_fr in zip(sustained_spike_num, spont_spike_num)]

        single_spont_msrsd = numpy_kit.msrsd(spont_spike_num)
        single_early_msrsd = numpy_kit.msrsd(early_fr)
        single_sustained_msrsd = numpy_kit.msrsd(sustained_fr)

        if np.any(~np.isfinite(np.array([
            single_spont_msrsd, 
            single_early_msrsd, 
            single_sustained_msrsd,]))):
            print("Non-finite values encountered for cell:", data_dict["node_name"], 
                  "Values:", [single_spont_msrsd, 
                             single_early_msrsd, 
                             single_sustained_msrsd],
                             "Skipping this cell due to non-finite values.")
            continue



        spont_msrsd.append(single_spont_msrsd)
        early_msrsd.append(single_early_msrsd)
        sustained_msrsd.append(single_sustained_msrsd)

        node_names.append(data_dict["node_name"])
        cohort_names.append(data_dict["cohort_name"])


    weight_matrix = np.stack([spont_msrsd, early_msrsd, sustained_msrsd], axis=1)
    return weight_matrix, np.array(node_names), np.array(cohort_names)



def get_weight_tuple_PSTH(variant: str, binsize: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_sst_wc, psth_sst_wc, base_fr_sst_wc, node_names_sst_wc = get_all_cellsession_PSTH_mean("SST_WC", BINSIZE=binsize)
    t_sst_jux, psth_sst_jux, base_fr_sst_jux, node_names_sst_jux = get_all_cellsession_PSTH_mean("SST_JUX", BINSIZE=binsize)
    t_pv_jux, psth_pv_jux, base_fr_pv_jux, node_names_pv_jux = get_all_cellsession_PSTH_mean("PV_JUX", BINSIZE=binsize)
    t_pyr_jux, psth_pyr_jux, base_fr_pyr_jux, node_names_pyr_jux = get_all_cellsession_PSTH_mean("PYR_JUX", BINSIZE=binsize)
    t_pyr_wc, psth_pyr_wc, base_fr_pyr_wc, node_names_pyr_wc = get_all_cellsession_PSTH_mean("PYR_WC", BINSIZE=binsize)

    assert np.allclose(t_sst_wc, t_sst_jux)
    assert np.allclose(t_sst_wc, t_pyr_wc)
    assert np.allclose(t_sst_wc, t_pv_jux)
    assert np.allclose(t_sst_wc, t_pyr_jux)
    t = t_sst_wc
    psth = np.concatenate([psth_sst_wc, psth_sst_jux, psth_pv_jux, psth_pyr_jux, psth_pyr_wc], axis=0)
    base_fr = np.concatenate([base_fr_sst_wc, base_fr_sst_jux, base_fr_pv_jux, base_fr_pyr_jux, base_fr_pyr_wc], axis=0)
    node_names = np.concatenate([node_names_sst_wc, node_names_sst_jux, node_names_pv_jux, node_names_pyr_jux, node_names_pyr_wc], axis=0)
    cohort_names = np.concatenate([np.full_like(node_names_sst_wc, "SST_WC"), 
                                 np.full_like(node_names_sst_jux, "SST_JUX"), 
                                 np.full_like(node_names_pv_jux, "PV_JUX"), 
                                 np.full_like(node_names_pyr_jux, "PYR_JUX"),
                                 np.full_like(node_names_pyr_wc, "PYR_WC")], axis=0)
    if variant == 'raw':
        weight_matrix = psth
    elif variant == 'zscore':
        weight_matrix = numpy_kit.zscore(psth, axis=1)
    elif variant == 'log-scale':
        weight_matrix = np.log(psth + 1)
    elif variant == "log-zscore":
        weight_matrix = numpy_kit.zscore(np.log(psth + 1), axis=1)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return weight_matrix, node_names, cohort_names


def get_weight_tuple_Waveform():
    # t_sst_wc, waveform_sst_wc, node_names_sst_wc = get_all_cellsession_Waveform_mean("SST_WC")
    t_sst_jux, waveform_sst_jux, node_names_sst_jux = get_all_cellsession_Waveform_mean("SST_JUX")
    t_pv_jux, waveform_pv_jux, node_names_pv_jux = get_all_cellsession_Waveform_mean("PV_JUX")
    t_pyr_jux, waveform_pyr_jux, node_names_pyr_jux = get_all_cellsession_Waveform_mean("PYR_JUX")

    grouped_waveform = grouping_timeseries(
        # [TimeSeries(v=v, t=t_sst_wc) for v in waveform_sst_wc] + 
        [TimeSeries(v=v, t=t_sst_jux) for v in waveform_sst_jux] + 
        [TimeSeries(v=v, t=t_pv_jux) for v in waveform_pv_jux] + 
        [TimeSeries(v=v, t=t_pyr_jux) for v in waveform_pyr_jux])
    
    waveform = grouped_waveform.raw_array
    node_names = np.concatenate([
        # node_names_sst_wc, 
        node_names_sst_jux, 
        node_names_pv_jux, 
        node_names_pyr_jux], axis=0)
    cohort_name = np.concatenate([
        # np.full_like(node_names_sst_wc, "SST_WC"), 
        np.full_like(node_names_sst_jux, "SST_JUX"), 
        np.full_like(node_names_pv_jux, "PV_JUX"), 
        np.full_like(node_names_pyr_jux, "PYR_JUX")], axis=0)
    return waveform, node_names, cohort_name


def Visualize_Waveform():
    weight_matrix, node_names, cohort_name = get_weight_tuple_Waveform()
    import matplotlib.pyplot as plt    

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6


    sst_wc_physiology = get_all_physiology_features("SST_WC")
    sst_jux_physiology = get_all_physiology_features("SST_JUX")
    pv_jux_physiology = get_all_physiology_features("PV_JUX")
    pyr_jux_physiology = get_all_physiology_features("PYR_JUX")

    combined_physiology = pd.concat(
        [sst_wc_physiology, 
         sst_jux_physiology, 
         pv_jux_physiology, 
         pyr_jux_physiology], ignore_index=True)

    spike_asymmetry = combined_physiology["Spike Asymmetry [a.u.]"].to_numpy()
    spike_width = combined_physiology["Spike Width [ms]"].to_numpy()
    spike_feature_nodes_name = combined_physiology["node_name"].to_numpy()

    
    n_cell, n_feature = weight_matrix.shape
    x_scale, y_scale = 0.06, 0.006
    t = np.linspace(0, 1, n_feature)

    assert n_cell == len(cohort_name), f"Number of cells mismatch, got {n_cell} and {len(cohort_name)}"    
    dot_colors = [COHORT_BRIGHTER_COLORS[name] for name in cohort_name]
    
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.35), constrained_layout=True)

    for cell_idx, single_node_name, waveform_vector in zip(range(n_cell), node_names, weight_matrix):
        cell_index_in_physiology = np.where(spike_feature_nodes_name == single_node_name)[0]
        if len(cell_index_in_physiology) == 0:
            logger.warning(f"Node {single_node_name} not found in physiology features, skipping.")
            continue
        single_spike_asymmetry = float(spike_asymmetry[cell_index_in_physiology])
        single_spike_width = float(spike_width[cell_index_in_physiology])
        
        ax.plot(t * x_scale + single_spike_width, waveform_vector * y_scale + single_spike_asymmetry, 
                color=dot_colors[cell_idx], lw=0.4, alpha=0.9)

        ax.set_xlabel("Spike width (ms)", labelpad=1)
        ax.set_ylabel("Spike asymmetry (AU)", labelpad=1)
    # ax.set_yticks([0.5, 1, 1.5, 2])
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_yticks([-0.2, 0, 0.2, 0.4])
    ax.set_ylim(-0.2, 0.45)
    ax.tick_params(length=1, pad=1)
    ax.spines[['right', 'top']].set_visible(False)

    save_path = path.join(get_saving_path(), "FeatureSpace", "Scatter_Small_Waveforms.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def get_weight_tuple_Decomposition_Weights(variant: str, decomposition_method: str, n_components: int, append_baseline: bool,
                                           BINSIZE: float, _visualize: bool = False):
    from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, SparsePCA, NMF, FastICA
    if decomposition_method == 'PCA':
        solver = PCA(n_components=n_components, random_state=42)
    elif decomposition_method == 'SVD':
        solver = TruncatedSVD(n_components=n_components, random_state=42)
    elif decomposition_method == 'FA':
        solver = FactorAnalysis(n_components=n_components, random_state=42)
    elif decomposition_method == 'SparsePCA':
        solver = SparsePCA(n_components=n_components, random_state=42)
    elif decomposition_method == 'NMF':
        solver = NMF(n_components=n_components, random_state=42)
    elif decomposition_method == 'ICA':
        solver = FastICA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown decomposition method: {decomposition_method}")
    
    t_sst_wc, psth_sst_wc, base_fr_sst_wc, node_names_sst_wc = get_all_cellsession_PSTH_mean("SST_WC", BINSIZE=BINSIZE)
    t_sst_jux, psth_sst_jux, base_fr_sst_jux, node_names_sst_jux = get_all_cellsession_PSTH_mean("SST_JUX", BINSIZE=BINSIZE)
    t_pv_jux, psth_pv_jux, base_fr_pv_jux, node_names_pv_jux = get_all_cellsession_PSTH_mean("PV_JUX", BINSIZE=BINSIZE)
    t_pyr_jux, psth_pyr_jux, base_fr_pyr_jux, node_names_pyr_jux = get_all_cellsession_PSTH_mean("PYR_JUX", BINSIZE=BINSIZE)
    t_pyr_wc, psth_pyr_wc, base_fr_pyr_wc, node_names_pyr_wc = get_all_cellsession_PSTH_mean("PYR_WC", BINSIZE=BINSIZE)
    base_fr = np.concatenate([base_fr_sst_wc, base_fr_sst_jux, base_fr_pv_jux, base_fr_pyr_jux, base_fr_pyr_wc], axis=0)

    PSTH_weight_matrix, node_names, cohort_name = get_weight_tuple_PSTH(variant, binsize=BINSIZE)
    if np.min(PSTH_weight_matrix) < 0 and decomposition_method in ('NMF', ):
        raise NotImplementedError
    component_projection = solver.fit_transform(PSTH_weight_matrix)
    
    if _visualize:
        Visualize_Decomposition_Weights(variant, decomposition_method, n_components, 
                                        solver, t_sst_wc, base_fr, component_projection, node_names, cohort_name)
    if append_baseline:        
        component_projection = np.concatenate((base_fr[:, None], component_projection), axis=1)
    return component_projection, node_names, cohort_name


def Visualize_Decomposition_Weights(variant: str, decomposition_method: str, n_components: int, 
                                    solver, PC_t, base_fr, component_projection, node_names, cohort_names):
    
    import matplotlib.pyplot as plt    
    import seaborn as sns
    import matplotlib.patheffects as pe

    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(3, n_components + 1, figsize=(0.85 * (n_components + 1), 1.7), 
                            constrained_layout=True, height_ratios=[2.5, 2.5, 6])
    components_list_for_plot = [
        np.ones_like(PC_t),
    ] + [solver.components_[component_id, :] for component_id in range(n_components)]
    component_weights_for_plot = [
        base_fr
    ] + [component_projection[:, component_id] for component_id in range(n_components)]
    component_names_for_plot = [
        "Baseline",
    ] + [f"PC {component_id+1}" for component_id in range(n_components)]
    for component_id in range(2, n_components + 1):        
        # axs[0, component_id].sharey(axs[0, 1])      
        axs[0, component_id].sharey(axs[0, 1])
        axs[1, component_id].sharey(axs[1, 1])
        axs[2, component_id].sharey(axs[2, 1])

    rng = np.random.default_rng(0)
    for component_id, (component, component_weights, component_name) in enumerate(zip(
        components_list_for_plot, component_weights_for_plot, component_names_for_plot)):
        # axc = axs[:, component_id]
        axc = axs[:, component_id]

        axc_comp = axc[0]
        axc_comp.plot(PC_t, component, lw=0.8, color='black', 
                      path_effects=[pe.Stroke(linewidth=1.2, foreground='white'), pe.Normal()])
        axc_comp.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
        axc_comp.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
        axc_comp.spines[['right', 'top', 'left', 'bottom',]].set_visible(False)
        axc_comp.set_xlim(-0.5, 1.)
        axc_comp.set_xticks([])
        axc_comp.set_yticks([])
        if component_name == "Baseline":
            axc_comp.set_ylim(0, 2)
            axc_comp.set_ylabel("Baseline FR")
        # elif component_id == 1:
        #     axc_comp.set_ylabel("PC")

        axc_zoomin = axc[1]
        axc_zoomin.plot(PC_t, component, lw=0.8, color='black', 
                      path_effects=[pe.Stroke(linewidth=1.2, foreground='white'), pe.Normal()])
        axc_zoomin.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
        axc_zoomin.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
        axc_zoomin.axvline(20/1000, color='black', linestyle='--', lw=0.5, alpha=0.5, zorder=5)
        axc_zoomin.set_xlim(-50/1000, 100/1000 )
        axc_zoomin.set_xticks([])
        axc_zoomin.set_yticks([])

        
        axc_weight = axc[2]
        data = pd.DataFrame({
            "Weight": component_weights,
            "Cohort": cohort_names,
        })
        axc_weight.tick_params(
            length=1,
            pad=1
        )
        axc_weight.spines[["right", "top", "bottom"]].set_visible(False)
        
        axc_weight.set_xlim(-0.4, 2.7)
        axc_weight.set_xticks(np.arange(3) + 0.2, ["SST", "PV", "PYR"], fontsize=6)

        for label, color in zip(
            axc_weight.get_xticklabels(),
            [COHORT_COLORS["SST_JUX"],
             COHORT_COLORS["PV_JUX"],
             COHORT_COLORS["PYR_JUX"],]
        ):
            label.set_color(color)

        annotation_data = {}
        for cohort_id, tmp_cohort_names in zip([0, 1, 2, 0, 2], [["SST_JUX","SST_WC",], ["PV_JUX",], ["PYR_JUX",  "PYR_WC"]]):

            cohort_data = data[data["Cohort"].isin(tmp_cohort_names)]
            values = cohort_data["Weight"].dropna().values
            if len(values) == 0:
                continue

            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            axc_weight.scatter(
                np.full(len(values), cohort_id) + jitter, values,
                edgecolor=COHORT_COLORS[tmp_cohort_names[0]],
                facecolor="white", lw=0.5, s=4, alpha=0.9, zorder=2,
            )
            axc_weight.errorbar(cohort_id + 0.4, np.nanmean(values), 
                        yerr=np.nanstd(values),
                        fmt="o",
                        markeredgecolor="black",
                        markerfacecolor=COHORT_COLORS[tmp_cohort_names[0]],
                        markeredgewidth=0.5,
                        markersize=3,
                        ecolor='black',
                        elinewidth=0.5,
                        capsize=2,
                        capthick=0.5,
                        alpha=0.9,)
            annotation_data[cohort_id] = values
        
        axc_weight.set_xlabel("")
        axc_weight.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)

        basic_ttest.stats_anova(annotation_data, axc_weight, bon_ferroni_adj=5)

        # if component_name == "Baseline":
        #     axc_weight.set_title("Baseline FR [Hz]")
        # elif component_id == 1:
        #     axc_weight.set_title("Weights [a.u.]")

        # new_xticks = [tick_formatter(tick_text.get_text()) for tick_text in axc_weight.get_xticklabels()]
        # axc_weight.set_xticklabels(new_xticks)

    save_path = path.join(get_saving_path(), "FeatureSpace", 
                          f"PC_Overview_{variant}_{decomposition_method}_{n_components}_annotated.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    # fig = plt.figure(figsize=(3.5, 3.5), )
    # ax = fig.add_subplot(111, projection='3d')

    # for cohort in ("PV_JUX", "SST_JUX", "PYR_JUX", "SST_WC",):
    #     idx = np.argwhere(cohort_name == cohort).flatten()
    #     ax.scatter(
    #         component_projection[idx, 0], component_projection[idx, 1], base_fr[idx],
    #         facecolors=COHORT_COLORS[cohort],
    #         s=20, alpha=1, edgecolors="black", lw=0.5,
    #         )
    #     # for i, j, k in zip(component_projection[idx, 0], component_projection[idx, 1], base_fr[idx]):
    #     #     ax.plot([i, i], [j, j], [0, k], color=COHORT_COLORS[cohort], lw=1, alpha=0.75, zorder=-10)
    # ax.set_xlabel("PC 1 Weight [a.u.]", labelpad=-15)
    # ax.set_ylabel("PC 2 Weight [a.u.]", labelpad=-15)
    # ax.set_zlabel("Baseline FR [Hz]", labelpad=-5)
    # ax.set_zlim(0, None)
    # ax.view_init(elev=20, azim=205)
    # ax.xaxis.pane.fill = False # Left pane
    # ax.yaxis.pane.fill = False # Right pane

    # # Remove grid lines
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.tick_params(axis='z', which='both', pad=0)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    # ax.zaxis.set_pane_color((0.8, 0.8, 0.8, 1))

    # save_path = path.join(get_saving_path(), "FeatureSpace", 
    #                       f"Scatter3D_{variant}_{decomposition_method}_{n_components}.png")
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # fig.savefig(save_path, dpi=500, transparent=True)
    # plt.close(fig)
    # logger.info("Plot saved to " + save_path)




DEFAULT_UMAP_KWS = {'n_components': 2, 'random_state': 12,}
DEFAULT_TSNE_KWS = {'n_components': 2, 'random_state': 12,}
DEFAULT_UMAP_N_NEIGHBORS = (6, 7, 8, 9, 10,)
DEFAULT_TSNE_PERPLEXITY = (5, 10, 15, 20, 30, )
DEFAULT_DOT_KWS = {"edgecolors": 'black', "lw": 0.5, "s": 20, "alpha": 0.9}
def simple_Landscope_feature_space(weight_matrix, cohort_name, feature_space_name: str, _normalize_all_dimension: bool = True):
    from umap import UMAP
    from sklearn.manifold import TSNE    
    import matplotlib.pyplot as plt    

    plt.rcParams["font.family"] = "Arial"
    randomseed = 12
    n_cell, n_feature = weight_matrix.shape
    assert n_cell == len(cohort_name), f"Number of cells mismatch, got {n_cell} and {len(cohort_name)}"
    if _normalize_all_dimension:
        normalized_weight_matrix = numpy_kit.zscore(weight_matrix, axis=0)
    else:
        normalized_weight_matrix = weight_matrix
    dot_colors = [COHORT_COLORS[name] for name in cohort_name]
    n_col = max(len(DEFAULT_UMAP_N_NEIGHBORS), len(DEFAULT_TSNE_PERPLEXITY))
    fig, axs = plt.subplots(2, n_col, figsize=(2.5 * n_col, 2.5 * 2), constrained_layout=True)

    for ax, n_neighbors in zip(axs[0], DEFAULT_UMAP_N_NEIGHBORS):
        umap_embedding = UMAP(n_neighbors=n_neighbors, **(DEFAULT_UMAP_KWS|{"random_state": randomseed})).fit_transform(normalized_weight_matrix)
        ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                facecolors=dot_colors, **DEFAULT_DOT_KWS)
        ax.set_aspect('equal')
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP, n_neighbors={n_neighbors}")
    
    for ax, perplexity in zip(axs[1], DEFAULT_TSNE_PERPLEXITY):
        tsne_embedding = TSNE(perplexity=perplexity, **(DEFAULT_TSNE_KWS|{"random_state": randomseed})).fit_transform(normalized_weight_matrix)
        ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                facecolors=dot_colors, **DEFAULT_DOT_KWS)
        ax.set_aspect("equal")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE, perplexity={perplexity}")
    
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top']].set_visible(False)

    save_path = path.join(get_saving_path(), "FeatureSpace", f"Landscope_{feature_space_name}_randomseed{randomseed}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


def simple_pairwise_distance_distribution(weight_matrix, cohort_name, feature_space_name: str, annotate: bool = True):
    import matplotlib.pyplot as plt    
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    n_cell, n_feature = weight_matrix.shape
    print(f"Number of cells: {n_cell}, Number of features: {n_feature}")
    assert n_cell == len(cohort_name)

    normalized_weight_matrix = numpy_kit.zscore(weight_matrix, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(1.7, 1.5), constrained_layout=True)
    
    ax.tick_params( 
        length=1,
        pad=1     
    )
    annotate_data = {}
    cohort_indices = []
    for group_name in np.unique(cohort_name):
        weight_in_group = normalized_weight_matrix[cohort_name == group_name]
        if len(weight_in_group) >= 5:
            pairwise_distance = pdist(weight_in_group, metric="cosine")
            ax.plot(np.sort(1-pairwise_distance), np.arange(1, 1+len(pairwise_distance))/len(pairwise_distance), 
                    color=COHORT_BRIGHTER_COLORS[group_name], alpha=0.9, lw=1)
            annotate_data[group_name] = 1 - pairwise_distance
            cohort_indices.append(np.where(cohort_name == group_name)[0])

    all_pairwise_corr = 1 - pdist(
        normalized_weight_matrix, metric="cosine"
    )

    # with-in
    i, j = np.triu_indices(n_cell, k=1)
    within_mask = cohort_name[i] == cohort_name[j]
    cross_mask = ~within_mask

    within_corr = all_pairwise_corr[within_mask]
    cross_corr = all_pairwise_corr[cross_mask]
    all_corr = all_pairwise_corr.flatten()

    # ax.plot(np.sort(within_corr), np.linspace(0, 1, len(within_corr)), color="gray", alpha=0.9, lw=1)
    ax.plot(np.sort(cross_corr), np.arange(1, 1+len(cross_corr))/len(cross_corr), color="black", alpha=0.9, lw=1, ls="--")
    # ax.plot(np.sort(all_corr), np.arange(1, 1+len(all_corr))/len(all_corr), color="black", alpha=0.9, lw=1, ls='--')
    # annotate_data['within'] = within_corr
    # annotate_data['cross'] = cross_corr
    # annotate_data['all'] = all_pairwise_corr
    
    ax.set_xlabel("Pairwise correlation (cosine)", labelpad=1)
    ax.set_ylabel("Cumulative distribution", labelpad=1)
    ax.spines[['right', 'top',]].set_visible(False)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xticks([-1, 0, 1])
    if annotate:
        # basic_ttest.stats_pairwise_ks(annotate_data, ax=ax,)
        results = basic_ttest.stats_pairwise_permutation(
            weight_matrix=normalized_weight_matrix,
            cohort_name=[x.split("_")[0] for x in cohort_name],
            ax=ax,
        )

    save_path = path.join(get_saving_path(), "FeatureSpace", f"PairwiseDistance_{feature_space_name}{'_annotated' if annotate else ''}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    return 1 - squareform(pdist(normalized_weight_matrix, metric="cosine")), cohort_name



def simple_pairwise_correlation_matrix(
    weight_matrix,
    cohort_name,
    feature_space_name: str,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from os import path

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    n_cell, n_feature = weight_matrix.shape
    cohort_name = np.asarray(cohort_name)

    assert n_cell == len(cohort_name)
    normalized_weight_matrix = numpy_kit.zscore(weight_matrix, axis=0,)

    cohort_order = [
        "PV_JUX",
        "SST_JUX",
        "PYR_JUX",
    ]

    sort_idx = []
    for cohort in cohort_order:
        cohort_idx = np.where(cohort_name == cohort)[0]
        weight_in_cohort = normalized_weight_matrix[cohort_idx]
        local_order = linkage_order(weight_in_cohort, method="average", metric="cosine")

        ordered_idx = cohort_idx[local_order]
        sort_idx.append(ordered_idx)

    sort_idx = np.concatenate(sort_idx)

    sorted_weight_matrix = normalized_weight_matrix[sort_idx]
    sorted_cohort_name = cohort_name[sort_idx]

    from sklearn.metrics.pairwise import cosine_similarity
    correlation_matrix = cosine_similarity(sorted_weight_matrix)


    cohort_sizes = [
        np.sum(sorted_cohort_name == cohort)
        for cohort in cohort_order
    ]
    boundaries = np.cumsum(cohort_sizes)[:-1]
    print(boundaries, cohort_sizes)


    starts = np.concatenate([[0], np.cumsum(cohort_sizes)[:-1]])
    centers = starts + (np.asarray(cohort_sizes) - 1) / 2

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(1.7, 1.5),
        constrained_layout=True,
    )

    im = ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="none",
        aspect="equal",
        rasterized=True,
    )

    # Thin white lines separating cohorts
    # -0.5 aligns lines exactly between matrix pixels
    for boundary in boundaries:
        position = boundary - 0.5

        ax.axhline(
            position,
            color="white",
            linewidth=0.5,
            ls="--"
        )
        ax.axvline(
            position,
            color="white",
            linewidth=0.5,
            ls="--"
        )

    # ------------------------------------------------------------
    # Cohort labels instead of individual cell labels
    # ------------------------------------------------------------
    ax.set_xticks(centers)
    ax.set_yticks(centers)

    ax.set_xticklabels(
        ["PV", "SST", "PYR"],
        fontsize=7,
    )
    for label, color in zip(
        ax.get_xticklabels(),
        [COHORT_COLORS["PV_JUX"],
            COHORT_COLORS["SST_JUX"],
            COHORT_COLORS["PYR_JUX"],]
    ):
        label.set_color(color)

    ax.set_yticklabels(
        ["PV", "SST", "PYR"],
        fontsize=7,
        rotation=90,
        va="center", ha="right"
    )
    for label, color in zip(
        ax.get_yticklabels(),
        [COHORT_COLORS["PV_JUX"],
            COHORT_COLORS["SST_JUX"],
            COHORT_COLORS["PYR_JUX"],]
    ):
        label.set_color(color)

    ax.tick_params(
        length=0,
        pad=1,
    )

    # # Remove spines
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    save_path = path.join(
        get_saving_path(),
        "FeatureSpace",
        f"PairwiseCorrelationMatrix_{feature_space_name}.png",
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=500,
        transparent=True,
    )

    plt.close(fig)

    logger.info("Plot saved to " + save_path)

    return correlation_matrix, sort_idx


    

def simple_feature_correlation_matrix(
    weight_matrix,
    cohort_name,
    feature_space_name: str,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from os import path

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5

    n_cell, n_feature = weight_matrix.shape

    feature_name_list = [
        "Spont. FR",
        "Spike width",
        "Spike asym.",
        r"$\mathrm{CV}_2$",
        "ACG rise",
    ] if feature_space_name == "Physiology_Fingerprint" else [
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
    ]

    target_feature_name_order = [0, 2, 1, 3, 4]
    feature_corr = np.corrcoef(weight_matrix.T)
    fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.5), constrained_layout=True)
    ax.tick_params(length=1, pad=1)

    ax.imshow(
        feature_corr[np.ix_(target_feature_name_order, target_feature_name_order)],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="none",
        aspect="equal",
        rasterized=True,
    )
    ax.set_xticks(np.arange(n_feature), [feature_name_list[i] for i in target_feature_name_order], rotation=30, ha='right')
    ax.set_yticks(np.arange(n_feature), [feature_name_list[i] for i in target_feature_name_order])

    save_path = path.join(
        get_saving_path(),
        "FeatureSpace",
        f"FeatureCorrelationMatrix_{feature_space_name}.png",
    )
    default_exit_save(fig, save_path, _transparent=True)