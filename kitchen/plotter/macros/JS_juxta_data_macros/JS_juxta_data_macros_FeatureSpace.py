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
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_500ms_puff_trials
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
                acg_rise_time_s = calculate_physiology.calculate_autocorrelogram_rise_time(all_spikes)
            else:
                width_ms = np.nan
                asymmetry_au = np.nan
                cv2_au = np.nan
                acg_rise_time_s = np.nan
            
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
                "ACG Rise Time [s]": acg_rise_time_s,

                "node_name": str(cell_session_node.coordinate),
                "cohort_name": dataset_name,
            })
            
        all_physiology_features = pd.DataFrame(all_physiology_features)
        all_physiology_features.to_csv(pkl_save_path, index=False)  
        logger.info("Physiology features saved to " + pkl_save_path)

    all_physiology_features = pd.read_csv(pkl_save_path)
    logger.info("Physiology features loaded from " + pkl_save_path)
    return all_physiology_features


def VisualizeExample_Physiology_Fingerprint():
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
    for feature_name in combined_physiology.columns[:-2]:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), constrained_layout=True)
        sns.boxplot(data=combined_physiology, y="cohort_name", x=feature_name, ax=ax, 
                    hue="cohort_name", palette=COHORT_COLORS, orient="h", zorder=10,
                    order=["PV_JUX", "SST_JUX", "PYR_JUX", ], showfliers=False,
                    # showmeans=True, meanline=True,
                    medianprops={'color': 'k', 'ls': '-', 'lw': 2, },
                    # medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    showbox=False,
                    showcaps=False,
                    width=0.5,
                    )
        sns.stripplot(data=combined_physiology, y="cohort_name", x=feature_name, ax=ax, 
                      order=["PV_JUX", "SST_JUX", "PYR_JUX", ],
                      hue="cohort_name", palette=COHORT_COLORS,
                      jitter=0.25,
                      alpha=0.9, size=5, orient="h", edgecolor="white", linewidth=0.5,)
       
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("")
        ax.set_yticks([])
        if "ACG" in feature_name:
            ax.set_xscale("log")
        elif "FR" in feature_name and "Std" not in feature_name:
            ax.set_xscale('symlog', linthresh=10)
            ax.set_xlim(-5, 110)
        elif "Asymmetry" in feature_name and "Std" not in feature_name:
            ax.set_xlim(-0.6, 0.6)
        ax.spines[['right', 'top', 'left']].set_visible(False)

        valid_feature_name = feature_name.replace("\n", "_").replace("[", "").replace("]", "").replace(",", "")
        save_path = path.join(get_saving_path(), "Physiology_Fingerprint", f"{valid_feature_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)
    

def get_weight_tuple_Physiology_Fingerprint(with_spont_FR: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sst_wc_physiology = get_all_physiology_features("SST_WC")
    sst_jux_physiology = get_all_physiology_features("SST_JUX")
    pv_jux_physiology = get_all_physiology_features("PV_JUX")
    pyr_jux_physiology = get_all_physiology_features("PYR_JUX")

    combined_physiology = pd.concat(
        [sst_wc_physiology, 
         sst_jux_physiology, 
         pv_jux_physiology, 
         pyr_jux_physiology], ignore_index=True)
    
    wanted_columns = [
        "Spont. FR [Hz]",
        "Spike Width [ms]",
        "Spike Asymmetry [a.u.]",
        "CV2 [a.u.]",
        "ACG Rise Time [s]",
        "node_name",
        "cohort_name",
    ] if with_spont_FR else [
        "Spike Width [ms]",
        "Spike Asymmetry [a.u.]",
        "CV2 [a.u.]",
        "ACG Rise Time [s]",
        "node_name",
        "cohort_name",
    ]

    wanted_df = combined_physiology[wanted_columns].dropna()

    weight_matrix = wanted_df[wanted_columns[:-2]].to_numpy()
    return weight_matrix, wanted_df["node_name"].to_numpy(), wanted_df["cohort_name"].to_numpy()


def get_weight_tuple_PSTH(variant: str = 'raw', binsize: float = 10/1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_sst_wc, psth_sst_wc, base_fr_sst_wc, node_names_sst_wc = get_all_cellsession_PSTH_mean("SST_WC", BINSIZE=binsize)
    t_sst_jux, psth_sst_jux, base_fr_sst_jux, node_names_sst_jux = get_all_cellsession_PSTH_mean("SST_JUX", BINSIZE=binsize)
    t_pv_jux, psth_pv_jux, base_fr_pv_jux, node_names_pv_jux = get_all_cellsession_PSTH_mean("PV_JUX", BINSIZE=binsize)
    t_pyr_jux, psth_pyr_jux, base_fr_pyr_jux, node_names_pyr_jux = get_all_cellsession_PSTH_mean("PYR_JUX", BINSIZE=binsize)

    assert np.allclose(t_sst_wc, t_sst_jux)
    assert np.allclose(t_sst_wc, t_pv_jux)
    assert np.allclose(t_sst_wc, t_pyr_jux)
    t = t_sst_wc
    psth = np.concatenate([psth_sst_wc, psth_sst_jux, psth_pv_jux, psth_pyr_jux], axis=0)
    base_fr = np.concatenate([base_fr_sst_wc, base_fr_sst_jux, base_fr_pv_jux, base_fr_pyr_jux], axis=0)
    node_names = np.concatenate([node_names_sst_wc, node_names_sst_jux, node_names_pv_jux, node_names_pyr_jux], axis=0)
    cohort_name = np.concatenate([np.full_like(node_names_sst_wc, "SST_WC"), 
                                 np.full_like(node_names_sst_jux, "SST_JUX"), 
                                 np.full_like(node_names_pv_jux, "PV_JUX"), 
                                 np.full_like(node_names_pyr_jux, "PYR_JUX")], axis=0)
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
    return weight_matrix, node_names, cohort_name   


def get_weight_tuple_Waveform():
    t_sst_wc, waveform_sst_wc, node_names_sst_wc = get_all_cellsession_Waveform_mean("SST_WC")
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
    from umap import UMAP
    import matplotlib.pyplot as plt    

    plt.rcParams["font.family"] = "Arial"

    n_cell, n_feature = weight_matrix.shape
    x_scale, y_scale = 0.15, 0.03
    t = np.linspace(0, 1, n_feature)
    assert n_cell == len(cohort_name), f"Number of cells mismatch, got {n_cell} and {len(cohort_name)}"
    normalized_weight_matrix = weight_matrix
    dot_colors = [COHORT_COLORS[name] for name in cohort_name]
    n_col =  len(DEFAULT_UMAP_N_NEIGHBORS)
    fig, axs = plt.subplots(1, n_col, figsize=(2.5 * n_col, 2.5), constrained_layout=True)

    for ax, n_neighbors in zip(axs, DEFAULT_UMAP_N_NEIGHBORS):
        umap_embedding = UMAP(n_neighbors=n_neighbors, **DEFAULT_UMAP_KWS).fit_transform(normalized_weight_matrix)
        umap_embedding = numpy_kit.zscore(umap_embedding, axis=0)
        
        for cell_idx, (x, y), waveform_vector in zip(range(n_cell), umap_embedding, weight_matrix):
            ax.plot(t * x_scale + x, waveform_vector * y_scale + y, color=dot_colors[cell_idx], lw=0.3, alpha=0.9)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP, n_neighbors={n_neighbors}")
  
    
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top']].set_visible(False)

    save_path = path.join(get_saving_path(), "FeatureSpace", "Scatter_Small_Waveforms.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def get_weight_tuple_Decomposition_Weights(variant: str, decomposition_method: str, n_components: int, append_baseline: bool, _visualize: bool = False):
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
    
    t_sst_wc, psth_sst_wc, base_fr_sst_wc, node_names_sst_wc = get_all_cellsession_PSTH_mean("SST_WC")
    t_sst_jux, psth_sst_jux, base_fr_sst_jux, node_names_sst_jux = get_all_cellsession_PSTH_mean("SST_JUX")
    t_pv_jux, psth_pv_jux, base_fr_pv_jux, node_names_pv_jux = get_all_cellsession_PSTH_mean("PV_JUX")
    t_pyr_jux, psth_pyr_jux, base_fr_pyr_jux, node_names_pyr_jux = get_all_cellsession_PSTH_mean("PYR_JUX")
    base_fr = np.concatenate([base_fr_sst_wc, base_fr_sst_jux, base_fr_pv_jux, base_fr_pyr_jux], axis=0)

    PSTH_weight_matrix, node_names, cohort_name = get_weight_tuple_PSTH(variant)
    if np.min(PSTH_weight_matrix) < 0 and decomposition_method in ('NMF', ):
        return None
    component_projection = solver.fit_transform(PSTH_weight_matrix)
    
    if _visualize:
        Visualize_Decomposition_Weights(variant, decomposition_method, n_components, 
                                        solver, t_sst_wc, base_fr, component_projection, node_names, cohort_name)
    if append_baseline:        
        component_projection = np.concatenate((base_fr[:, None], component_projection), axis=1)
    return component_projection, node_names, cohort_name


def Visualize_Decomposition_Weights(variant: str, decomposition_method: str, n_components: int, 
                                    solver, basis_t, base_fr, component_projection, node_names, cohort_name):
    
    import matplotlib.pyplot as plt    
    import seaborn as sns
    import matplotlib.patheffects as pe

    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(2, n_components + 1, figsize=(2.5 * (n_components + 1), 3.), 
                            constrained_layout=True, height_ratios=[4, 6])
    components_list_for_plot = [
        np.ones_like(basis_t),
    ] + [solver.components_[component_id, :] for component_id in range(n_components)]
    component_weights_for_plot = [
        base_fr
    ] + [component_projection[:, component_id] for component_id in range(n_components)]
    component_names_for_plot = [
        "Baseline",
    ] + [f"Basis {component_id+1}" for component_id in range(n_components)]
    for component_id in range(2, n_components + 1):        
        axs[0, component_id].sharey(axs[0, 1])

    for component_id, (component, component_weights, component_name) in enumerate(zip(
        components_list_for_plot, component_weights_for_plot, component_names_for_plot)):
        axc = axs[:, component_id]

        axc_comp = axc[0]
        axc_comp.plot(basis_t, component, lw=1.5, color='black', 
                      path_effects=[pe.Stroke(linewidth=2.5, foreground='white'), pe.Normal()])
        axc_comp.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
        axc_comp.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
        axc_comp.spines[['right', 'top', 'left', 'bottom',]].set_visible(False)
        axc_comp.set_xlim(-0.5, 1.)
        axc_comp.set_xticks([])
        # axc_comp.set_yticks([])
        if component_name == "Baseline":
            axc_comp.set_ylim(0, 2)
            axc_comp.set_ylabel("Baseline FR")
        elif component_id == 1:
            axc_comp.set_ylabel("Basis")
        
        axc_weight = axc[1]
        data = pd.DataFrame({
            "Weight": component_weights,
            "Cohort": cohort_name,
        })
        sns.boxplot(data=data, x="Weight", y="Cohort", hue="Cohort", 
                    orient="h",
                    ax=axc_weight, 
                    palette=COHORT_COLORS,  
                    order=["PV_JUX", "SST_JUX", "PYR_JUX", "SST_WC", ],
                    showfliers=False, zorder=10,
                    medianprops={'color': 'k', 'ls': '-', 'lw': 2, },
                    # medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    showbox=False,
                    showcaps=False,
                    width=0.5,
                    )
        sns.stripplot(data=data, x="Weight", y="Cohort", hue="Cohort", 
                      ax=axc_weight, orient="h",
                    palette=COHORT_COLORS, alpha=0.9, jitter=0.25, size=5,
                    edgecolor="black", linewidth=0.5,
                    order=sorted(set(cohort_name)))
        axc_weight.spines[['right', 'top', 'left']].set_visible(False)
        axc_weight.set_xlabel("Weight [a.u.]" if component_id != 0 else "Baseline FR [Hz]")
        axc_weight.set_yticks([])
        axc_weight.axvline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
        axc_weight.set_ylabel("")
        # new_xticks = [tick_formatter(tick_text.get_text()) for tick_text in axc_weight.get_xticklabels()]
        # axc_weight.set_xticklabels(new_xticks)

    save_path = path.join(get_saving_path(), "FeatureSpace", 
                          f"Basis_Overview_{variant}_{decomposition_method}_{n_components}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    fig = plt.figure(figsize=(3.5, 3.5), )
    ax = fig.add_subplot(111, projection='3d')

    for cohort in ("PV_JUX", "SST_JUX", "PYR_JUX", "SST_WC",):
        idx = np.argwhere(cohort_name == cohort).flatten()
        ax.scatter(
            component_projection[idx, 0], component_projection[idx, 1], base_fr[idx],
            facecolors=COHORT_COLORS[cohort],
            s=15, alpha=1, edgecolors="black", lw=0.5,
            )
        for i, j, k in zip(component_projection[idx, 0], component_projection[idx, 1], base_fr[idx]):
            ax.plot([i, i], [j, j], [0, k], color=COHORT_COLORS[cohort], lw=1, alpha=0.75, zorder=-10)
    ax.set_xlabel("Basis 1 Weight [a.u.]", labelpad=-15)
    ax.set_ylabel("Basis 2 Weight [a.u.]", labelpad=-15)
    ax.set_zlabel("Baseline FR [Hz]", labelpad=-5)
    ax.set_zlim(0, None)
    ax.view_init(elev=30, azim=205)
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane

    # Remove grid lines
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='z', which='both', pad=0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    save_path = path.join(get_saving_path(), "FeatureSpace", 
                          f"Scatter3D_{variant}_{decomposition_method}_{n_components}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)




DEFAULT_UMAP_KWS = {'n_components': 2, 'random_state': 42,}
DEFAULT_TSNE_KWS = {'n_components': 2, 'random_state': 42,}
DEFAULT_UMAP_N_NEIGHBORS = (6, 7, 8, 9, 10,)
DEFAULT_TSNE_PERPLEXITY = (5, 10, 15, 20, 30, )
DEFAULT_DOT_KWS = {"edgecolors": 'lightgray', "lw": 0.5, "s": 15, "alpha": 0.9}
def simple_Landscope_feature_space(weight_matrix, cohort_name, feature_space_name: str, _normalize_all_dimension: bool = True):
    from umap import UMAP
    from sklearn.manifold import TSNE    
    import matplotlib.pyplot as plt    

    plt.rcParams["font.family"] = "Arial"

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
        umap_embedding = UMAP(n_neighbors=n_neighbors, **DEFAULT_UMAP_KWS).fit_transform(normalized_weight_matrix)
        ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                   facecolors=dot_colors, **DEFAULT_DOT_KWS)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP, n_neighbors={n_neighbors}")
    
    for ax, perplexity in zip(axs[1], DEFAULT_TSNE_PERPLEXITY):
        tsne_embedding = TSNE(perplexity=perplexity, **DEFAULT_TSNE_KWS).fit_transform(normalized_weight_matrix)
        ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], 
                   facecolors=dot_colors, **DEFAULT_DOT_KWS)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE, perplexity={perplexity}")
    
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top']].set_visible(False)

    save_path = path.join(get_saving_path(), "FeatureSpace", f"Landscope_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


def simple_pairwise_distance_distribution(weight_matrix, cohort_name, feature_space_name: str, _normalize_all_dimension: bool = True):
    import matplotlib.pyplot as plt    
    import seaborn as sns
    from scipy.spatial.distance import pdist

    plt.rcParams["font.family"] = "Arial"

    n_cell, n_feature = weight_matrix.shape
    assert n_cell == len(cohort_name)

    if _normalize_all_dimension:
        normalized_weight_matrix = numpy_kit.zscore(weight_matrix, axis=0)
    else:
        normalized_weight_matrix = weight_matrix
    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), constrained_layout=True)

    summarized_pairwise_distance = []
    for group_name in np.unique(cohort_name):
        weight_in_group = normalized_weight_matrix[cohort_name == group_name]
        if len(weight_in_group) >= 5:
            pairwise_distance = pdist(weight_in_group, metric="correlation")
            summarized_pairwise_distance.append(pd.DataFrame({'distance': 1 - pairwise_distance, 'group': group_name}))
    df = pd.concat(summarized_pairwise_distance, ignore_index=True)
    sns.kdeplot(data=df, x="distance", hue="group", multiple="layer", 
                palette=COHORT_COLORS, hue_order=["SST_JUX", "PYR_JUX", "PV_JUX", ],
                ax=ax, legend=False, fill=True, alpha=0.7, common_norm=False,)
    # sns.histplot(pairwise_distance, ax=ax, kde=True, stat="density", binwidth=0.05
    #              fill=True, alpha=0.4, color=COHORT_COLORS[group_name])
    ax.set_xlabel("Pairwise Correlation [R]")
    ax.set_ylabel("Density [a.u.]")
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left']].set_visible(False)

    save_path = path.join(get_saving_path(), "FeatureSpace", f"PairwiseDistance_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)