import logging
from os import path
import os
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
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, WC_CONVERT_FLAG
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
CLUSTER_COLORS = {
    0: "#991CE3",
    "cluster 0": "#991CE3",
    1: "#66E31C",
    "cluster 1": "#66E31C",
    2: "#E31C66",
    "cluster 2": "#E31C66",
}

bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
bin_centers = (bins[:-1] + bins[1:]) / 2


def load_all_datasets_pkl():
    all_spikes_histogram, all_cohort_names = [], []

    if not path.exists("tmp_save.npy"):        
        dataset_jux_sst = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="SST_JUX", 
                                    recipe="default_ephys", name="SST_JUX")
        dataset_wc_sst = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="SST_WC", 
                                    recipe="default_ephys", name="SST_WC")
        dataset_jux_pv = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="PV_JUX", 
                                    recipe="default_ephys", name="PV_JUX")
        dataset_jux_pyr = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="PYR_JUX", 
                                    recipe="default_ephys", name="PYR_JUX")
        all_datasets = [dataset_jux_sst, dataset_wc_sst, dataset_jux_pv, dataset_jux_pyr]
        for dataset_idx, dataset in enumerate(all_datasets):
            for cell_node in dataset.select("cellsession"):
                subtree = dataset.subtree(cell_node)
                puff_trials_500ms = subtree.select(
                    "trial", timeline=lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52, _empty_warning=False,)
                puff_trials_500ms = sync_nodes(puff_trials_500ms, alignment_events, plot_manual_spike300Hz)
                if len(puff_trials_500ms) == 0:
                    continue
                spikes_histogram = grouping_events_histogram([single_trial.potential.spikes.segment(*FEATURE_RANGE)
                                                            for single_trial in puff_trials_500ms],
                                                            bins=bins, use_event_value_as_weight=False)
                if spikes_histogram.data_num < 5 or (spikes_histogram.mean.min() == spikes_histogram.mean.max()):
                    continue
                all_spikes_histogram.append(spikes_histogram.mean)
                all_cohort_names.append(dataset.name)

        raw_feature_matrix = np.stack(all_spikes_histogram, axis=0)
        np.save("tmp_save.npy", {"feature_matrix": raw_feature_matrix, "cohort_names": all_cohort_names})

    logger.info("tmp_save.npy exists, loading...")
    tmp_save = np.load("tmp_save.npy", allow_pickle=True).item()
    return tmp_save["feature_matrix"], tmp_save["cohort_names"]


def get_fa(feature_matrix):
    from sklearn.decomposition import FactorAnalysis
    n_components = 5
    fa = FactorAnalysis(n_components=n_components, rotation='varimax', random_state=42)
    fa.fit(feature_matrix)
    return fa


def plot_factor_overview(fa, feature_matrix, cohort_names):
    components = fa.components_  # (n_factors, n_features)
    n_factors, n_features = fa.components_.shape

    factor_scores = fa.transform(feature_matrix)

    # calculate eigenvalues
    corr_matrix = np.corrcoef(feature_matrix.T)
    eigenvalues, _ = np.linalg.eig(corr_matrix)
    eigenvalues = sorted(eigenvalues, reverse=True)

    fig = plt.figure(figsize=(15, 4), constrained_layout=True)
    axs = [fig.add_subplot(1, 3, 1),
           fig.add_subplot(1, 3, 2),
           fig.add_subplot(1, 3, 3, projection='3d'),]
    top_eig = 10
    axs[0].scatter(range(1, top_eig + 1), eigenvalues[:top_eig])
    axs[0].plot(range(1, top_eig + 1), eigenvalues[:top_eig])
    # axs[0].axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue=1)')
    axs[0].set_title('Scree Plot')
    axs[0].set_xlabel('Principal Component')
    axs[0].set_ylabel('Eigenvalue')
    # axs[0].legend()

    for cohort_name in sorted(set(cohort_names)):
        idx = [i for i, name in enumerate(cohort_names) if name == cohort_name]
        axs[1].scatter(factor_scores[idx, 0], factor_scores[idx, 1],
                        color=COHORT_COLORS.get(cohort_name, 'gray'),
                        label=cohort_name, s=8, alpha=0.6)
        
        axs[2].scatter(factor_scores[idx, 0], factor_scores[idx, 1], factor_scores[idx, 2],
                        color=COHORT_COLORS.get(cohort_name, 'gray'),
                        label=cohort_name, s=8, alpha=0.6)
    for ax in axs[1:]:
        ax.set_xlabel('Factor 1 Score')
        ax.set_ylabel('Factor 2 Score')
        ax.legend(frameon=False, loc='best')
        # ax.set_aspect('equal')
    for ax in axs[:2]:
        ax.spines[['right', 'top']].set_visible(False)
    axs[1].set_title('Factor Scores (Factor 1 vs Factor 2)')
    axs[1].axvline(0, color='black', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
    axs[1].axhline(0, color='black', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
    axs[2].set_title('Factor Scores (Factor 1 vs Factor 2 vs Factor 3)')
    axs[2].set_zlabel('Factor 3 Score')

    fig.savefig("factor_overview.png", dpi=900)
    plt.close(fig)


def get_linkage_order(fa, feature_matrix, cohort_names):    
    from scipy.cluster.hierarchy import linkage, dendrogram
    fa_scores = fa.transform(feature_matrix)  # (n_samples, n_factors)
    linked = linkage(fa_scores[:, :3], method='ward', metric='euclidean')
    dendro_info = dendrogram(linked, orientation='left')


    fig, ax = plt.subplots(1, 1, figsize=(0.8, 3))
    matplotlib.rcParams['lines.linewidth'] = 0.5
    dendrogram(linked, ax=ax, orientation='left', )
    ax.tick_params(axis='y', which='both', length=0, labelright=False, )
    tick_positions = np.arange(len(cohort_names)) * 10 + 5
    ax.set_yticks(tick_positions + 0.5, [cohort_names[i] for i in dendro_info['leaves']], rotation=0, fontsize=1.)
    ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        tick_label.set_color("white")
        tick_label.set_bbox({'facecolor': COHORT_COLORS[text], 'alpha': 1.0, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.3'})
    ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.invert_yaxis()
    fig.savefig("dendrogram.png", dpi=900)
    plt.close(fig)
    matplotlib.rcParams['lines.linewidth'] = 1
    return dendro_info['leaves']


def plot_factor_detail(fa, feature_matrix, cohort_names):
    components = fa.components_  # (n_factors, n_features)
    n_factors, n_features = fa.components_.shape

    factor_scores = fa.transform(feature_matrix)
    top_factor = 3
    sorted_idx = get_linkage_order(fa, feature_matrix, cohort_names)
    fig, axs = plt.subplots(6, top_factor, figsize=(4*top_factor, 15), constrained_layout=True,
                            height_ratios=[1, 2, 2, 2, 2, 2], sharey='row')
    for factor_id in range(top_factor):
        ax_factor = axs[:, factor_id]
        factor_loadings = fa.components_[factor_id, :]
        project_values = factor_scores[:, factor_id]
        
        # Reconstruct the feature matrix based on the identified factors
        projected_feature = np.outer(project_values, factor_loadings)
        cumulative_reconstructed = np.dot(factor_scores[:, :factor_id+1], fa.components_[:factor_id+1, :]) + fa.mean_
        cumulative_error = np.abs(feature_matrix - cumulative_reconstructed)

        # Plot 1: Factor Loadings over time
        ax_factor[0].plot(bin_centers, factor_loadings, lw=1, color='black')
        ax_factor[0].set_title(f"Factor Component {factor_id+1}")
        ax_factor[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
        ax_factor[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        ax_factor[0].spines[['right', 'top',]].set_visible(False)
        ax_factor[0].set_xlabel("Time [s]")
        ax_factor[0].set_ylabel("FR [Hz]")

        # Plot 2: Factor Scores grouped by cohort
        data = pd.DataFrame({
            f"Factor {factor_id+1}": project_values,
            "Cohort": cohort_names,
        })
        sns.barplot(data=data, x="Cohort", y=f"Factor {factor_id+1}", hue="Cohort", ax=ax_factor[1],
                    palette=COHORT_COLORS, errorbar="se", alpha=0.7,
                    order=sorted(set(cohort_names)))
        sns.stripplot(data=data, x="Cohort", y=f"Factor {factor_id+1}", hue="Cohort", ax=ax_factor[1],
                      palette=COHORT_COLORS, alpha=0.7, jitter=0.2, size=3,
                      order=sorted(set(cohort_names)))
        ax_factor[1].spines[['right', 'top',]].set_visible(False)
        ax_factor[1].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        # ax_factor[1].set_xlabel("")
        ax_factor[1].set_ylabel(f"Factor Scores")
        
        # Heatmaps for reconstruction and error analysis
        heatmap_kws = {"cmap": 'viridis',}
        sns.heatmap(projected_feature[sorted_idx, :], ax=ax_factor[2], 
                    cbar=False, cmap='bwr', center=0., vmin=-100, vmax=100,)
        ax_factor[2].set_title(f"Reconstruction from Factor {factor_id+1}")
        sns.heatmap(cumulative_reconstructed[sorted_idx, :], ax=ax_factor[3], vmin=0, vmax=200,
                    cbar=False, **heatmap_kws)
        ax_factor[3].set_title(f"Cumulative Reconstruction")
        sns.heatmap(feature_matrix[sorted_idx, :], ax=ax_factor[4], vmin=0, vmax=200,
                    cbar=False, **heatmap_kws)
        ax_factor[4].set_title(f"Original Data (Sorted)")
        sns.heatmap(cumulative_error[sorted_idx, :], ax=ax_factor[5], vmin=0, vmax=100,
                    cbar=False, cmap='Reds', alpha=0.5)
        ax_factor[5].set_title(f"Cumulative Error")
        
        # Plot formatting for heatmaps (unchanged logic)
        for ax in ax_factor[2:]:
            ax.set_xlabel("Time [s]")
            plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
            ax.set_xticks(plot_tick_indices, [0, 0.5], rotation=0, )
            tick_positions = np.arange(len(cohort_names))
            ax.set_yticks(tick_positions + 0.5, [cohort_names[i] for i in sorted_idx], rotation=0, fontsize=1.)
            ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
            for tick_label in ax.get_yticklabels():
                text = tick_label.get_text()
                tick_label.set_color("white")
                tick_label.set_bbox({'facecolor': COHORT_COLORS[text], 'alpha': 1.0, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.3'})

    fig.savefig("factor_detail.png", dpi=900)
    plt.close(fig)

def plot_factor_score_heatmap(fa, feature_matrix, cohort_names):   
    factor_scores = fa.transform(feature_matrix)
    top_factor = 3
    sorted_idx = get_linkage_order(fa, feature_matrix, cohort_names)
    fig, ax = plt.subplots(1, 1, figsize=(0.8, 3), constrained_layout=True) 
    sns.heatmap(factor_scores[sorted_idx, :top_factor], ax=ax, cbar=False, cmap='bwr', center=0., vmin=-5, vmax=5,)
    ax.set_title("Factor Scores")
    ax.set_xlabel("Factor ID")
    
    ax.set_xticks(np.arange(top_factor) + 0.5, [f"{i+1}" for i in range(top_factor)])
    tick_positions = np.arange(len(cohort_names))
    ax.set_yticks(tick_positions + 0.5, [cohort_names[i] for i in sorted_idx], rotation=0, fontsize=1.)
    ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        tick_label.set_color("white")
        tick_label.set_bbox({'facecolor': COHORT_COLORS[text], 'alpha': 1.0, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.3'})

    fig.savefig("factor_score_heatmap.png", dpi=900)
    plt.close(fig)


def get_umap_embedding(fa, feature_matrix, cohort_names):    
    from umap import UMAP
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.cluster import SpectralClustering
    
    fa_scores = fa.transform(feature_matrix)  # (n_samples, n_factors)
    linked = linkage(fa_scores[:, :3], method='ward', metric='euclidean')
    dendro_info = dendrogram(linked, orientation='left')

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=6)
    embedding = reducer.fit_transform(fa_scores[:, ])

    cluster_id = SpectralClustering(n_clusters=2, n_neighbors=6, random_state=42).fit_predict(embedding)
    cluster_names = [f"cluster {i}" for i in cluster_id]



    fig, ax = plt.subplots(1, 1, figsize=(0.8, 3))
    matplotlib.rcParams['lines.linewidth'] = 0.5
    dendrogram(linked, ax=ax, orientation='left', )
    ax.tick_params(axis='y', which='both', length=0, labelright=False, )
    tick_positions = np.arange(len(cluster_names)) * 10 + 5
    ax.set_yticks(tick_positions + 0.5, [cluster_names[i] for i in dendro_info['leaves']], rotation=0, fontsize=1.)
    ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        tick_label.set_color("white")
        tick_label.set_bbox({'facecolor': CLUSTER_COLORS[text], 'alpha': 1.0, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.3'})
    ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.invert_yaxis()
    fig.savefig("cluster_dendrogram.png", dpi=900)
    plt.close(fig)
    matplotlib.rcParams['lines.linewidth'] = 1
    

    def draw_ellipse(points, ax, **kwargs):
        from matplotlib.patches import Ellipse
        position = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False)

        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        n_std = 2.0
        ell = Ellipse(xy=position, width=n_std * width, height=n_std * height, angle=angle, **kwargs)
        
        ax.add_patch(ell)
        return ell

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, width_ratios=[1, 1, 1])

    legend_handles = []
    for cluster_id in range(2):
        idx = [i for i, name in enumerate(cluster_names) if name == f"cluster {cluster_id}"]
        dot_cohort_colors = [COHORT_COLORS[cohort_names[i]] for i in idx]
        dot_cluster_colors = [CLUSTER_COLORS[f"cluster {cluster_id}"] for i in idx]
        axs[0].scatter(embedding[idx, 0], embedding[idx, 1],
                       facecolors=dot_cohort_colors, edgecolors=dot_cluster_colors, lw=0.5, s=8, alpha=0.6)
        draw_ellipse(embedding[idx, :], axs[0], edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"], 
                     facecolor='none', lw=2, ls='--', alpha=0.8, zorder=-10)
        legend_patch = mpatches.Patch(color=CLUSTER_COLORS[f"cluster {cluster_id}"], alpha=0.5, lw=0, label=f'cluster {cluster_id}')
        legend_handles.append(legend_patch)
        
    axs[0].legend(handles=legend_handles, title="Cluster", loc='best', frameon=False)
    axs[0].set_title("UMAP of Factor Scores")
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel("UMAP 1")
    axs[0].set_ylabel("UMAP 2")

    df = pd.DataFrame({
        "Cohort": cohort_names,
        "Cluster": cluster_names,
    })
    cohort_per_cluster = df.groupby(['Cluster', 'Cohort']).size().unstack(fill_value=0)
    cohort_per_cluster.plot(kind='bar', stacked=True, ax=axs[1], color=COHORT_COLORS)
    axs[1].set_title("Cohort Composition per Cluster")
    axs[1].set_xticklabels(cohort_per_cluster.index, rotation=0)
    axs[1].set_xlabel("Cluster")
    axs[1].set_ylabel("Number of Cohorts")
    axs[1].legend(title="Cohort", loc='best', frameon=False)
    axs[1].spines[['right', 'top']].set_visible(False)

    cluster_per_cohort = df.groupby(['Cohort', 'Cluster']).size().unstack(fill_value=0)
    cluster_per_cohort.plot(kind='bar', stacked=True, ax=axs[2], color=CLUSTER_COLORS)
    axs[2].set_title("Cluster Composition per Cohort")
    axs[2].set_xticklabels(cluster_per_cohort.index, rotation=0)
    axs[2].set_xlabel("Cohort")
    axs[2].set_ylabel("Number of Clusters")
    axs[2].legend(title="Cluster", loc='best', frameon=False)
    axs[2].spines[['right', 'top']].set_visible(False)

    fig.savefig("umap_embedding.png", dpi=900)
    plt.close(fig)
    

def answer_of_everything(dataset: DataSet, fa):
    total_session_nodes = dataset.select("cellsession", 
                                   _self=lambda x: len(dataset.subtree(x).select(
                                       "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,
                                       _empty_warning=False,
                                       )) > 0)
    total_n_session = len(total_session_nodes)
    batch_session_size = 6
    n_batch = int(np.ceil(total_n_session / batch_session_size))
    for batch_idx in range(n_batch):
        session_nodes = total_session_nodes.nodes[batch_idx * batch_session_size: (batch_idx + 1) * batch_session_size]
        n_session = len(session_nodes)

        n_sub_plots = 8
        fig, axs = plt.subplots(n_session, n_sub_plots, 
                                figsize=(12, 1.5 * n_session), 
                                width_ratios=[
                                    5,  # raster plot
                                    5,  # PSTH
                                    5,  # Factor Reconstruction
                                    5,  # Factor components
                                    3,  # convolved potential
                                    3,  # LFP potential, zomed in
                                    5,  # LFP potential, zoomed out
                                    # 3,  # early spike distribution
                                    # 3,  # sustained spike distribution
                                    3,  # waveform
                                    ], 
                                sharex='col', constrained_layout=True)
        if n_session == 1:
            axs = np.expand_dims(axs, axis=0)
        for row_idx, session_node in enumerate(session_nodes):
            for col_idx in range(n_sub_plots):
                axs[row_idx, col_idx].tick_params(labelbottom=True)

        # node features
        for row_idx, session_node in enumerate(session_nodes):
            axp = axs[row_idx, :]
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
            
            def raster_block(ax):
                for trial_id, single_trial in enumerate(puff_trials_500ms):
                    spikes_in_trial_by_type = {}
                    for spike_time, spike_type in zip(single_trial.potential.spikes.t, single_trial.potential.spikes.v):
                        spikes_in_trial_by_type.setdefault(spike_type, []).append(spike_time)
                        
                    for spike_type, spike_times in spikes_in_trial_by_type.items():
                        ax.eventplot(spike_times,                    
                                        lineoffsets=trial_id,   
                                        colors=SPIKE_COLOR_SCHEME[spike_type],
                                        alpha=0.9,
                                        linewidths=0.5,
                                        linelengths=0.8)   
                    
                    puff_on_t = single_trial.data.timeline.filter("VerticalPuffOn").t[0]
                    puff_off_t = single_trial.data.timeline.filter("VerticalPuffOff").t[0]
                    ax.add_patch(mpatches.Rectangle((puff_on_t, (trial_id - 0.5)), puff_off_t - puff_on_t, 1,
                                                    facecolor=PUFF_COLOR, alpha=0.5, edgecolor='none', zorder=-10))
                
                ax.spines[['right', 'top', 'left']].set_visible(False)
                ax.set_xlabel(f"Time [s]")
                ax.set_ylabel(f"Trial ID")
            
                ax.set_yticks([])
                ax.yaxis.set_major_locator(MultipleLocator(5))

                ax.set_xlim(-0.5, 1.0)     
                ax.set_ylim(-0.5, len(puff_trials_500ms)-0.5)
                ax.set_title(f"{node_name}")  

            def PSTH_block(ax):
                group_spikes = grouping_events_rate([single_trial.potential.spikes for single_trial in puff_trials_500ms],
                                                    bin_size=BINSIZE, use_event_value_as_weight=False)
                if len(group_spikes.t) > 0:
                    bin_edges = np.concatenate([group_spikes.t, [group_spikes.t[-1] + BINSIZE]]) - BINSIZE/2
                    ax.stairs(group_spikes.mean, bin_edges, fill=True, color='black')
                ax.axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)

                ax.spines[['right', 'top', 'left']].set_visible(False)
                ax.set_xlabel(f"Time [s]")
                ax.set_title(f"Firing Rate [Hz]") 
                ax.set_xlim(-0.5, 1.0)        
                ax.set_ylim(0, 150)
            
            def factor_components(ax1, ax2):
                PSTH = grouping_events_histogram([single_trial.potential.spikes.segment(*FEATURE_RANGE)
                                                            for single_trial in puff_trials_500ms],
                                                            bins=bins, use_event_value_as_weight=False).mean
                fa_scores = fa.transform(PSTH.reshape(1, -1))
                fa_projections = [np.outer(fa_scores[:, i], fa.components_[i, :]) for i in range(3)]
                reconstructed_PSTH = np.dot(fa_scores[:, :3], fa.components_[:3, :]) + fa.mean_

                ax1.plot(bin_centers, PSTH, label='PSTH', color='red', lw=0.5, alpha=0.7)
                ax1.plot(bin_centers, reconstructed_PSTH[0, :], label='Reconstructed PSTH', color='blue', lw=0.5, alpha=0.7)
                
                ax2.plot(bin_centers, fa.mean_, label='Mean', color='gray', lw=0.5, alpha=0.7)
                ax2.plot(bin_centers, fa_projections[0][0, :], label='Factor 1', lw=0.5, alpha=0.7)
                ax2.plot(bin_centers, fa_projections[1][0, :], label='Factor 2', lw=0.5, alpha=0.7)
                ax2.plot(bin_centers, fa_projections[2][0, :], label='Factor 3', lw=0.5, alpha=0.7)
                ax2.set_title(f"Factor Loadings")
                for ax in [ax1, ax2]:
                    ax.set_xlim(-0.5, 1)    
                    ax.spines[['right', 'top', ]].set_visible(False)
                    ax.axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
                    ax.axvspan(0, 0.5, alpha=0.2, color=PUFF_COLOR, lw=0, zorder=-10)
                    ax.set_xlabel(f"Time [s]")
                    ax.set_ylabel(f"Firing Rate [Hz]")                    
                    ax.legend(frameon=False, loc='best')

            raster_block(axp[0])
            PSTH_block(axp[1])
            factor_components(axp[2], axp[3])


            if session_node.potential is None:
                logger.debug(f"Cannot plot convolved potential for {node_name}: no potential found. Skip...")
                continue
            
            def convolved_potential(ax):
                # conv potential
                ax.set_yticks([])
                unit_plot_potential_conv(potential=[one_trial.data.potential for one_trial in puff_trials_500ms], 
                                        ax=ax, y_offset=0, ratio=FLUORESCENCE_RATIO, individual_trace_flag=True, spike_mark=False)
                ax.spines[['right', 'top',]].set_visible(False)
                ax.set_xlabel(f"Time [s]")
                ax.set_title(f"Conv. {DF_F0_SIGN}")
                ax.axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)

            def LFP_example(ax1, ax2):
                for ax in [ax1, ax2]:
                    # LFP potential 
                    ax.set_yticks([])
                    unit_plot_potential(potential=[one_trial.data.potential for one_trial in puff_trials_500ms], 
                                        ax=ax, y_offset=0, ratio=POTENTIAL_RATIO, aspect=4, wc_flag=WC_CONVERT_FLAG(session_node))
                    ax.spines[['right', 'top',]].set_visible(False)
                    ax.set_xlabel(f"Time [s]")   
                    ax.set_title(f"Vm [4Hz, mV]")
                    ax.axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)    
                ax1.set_xlim(-0.1, 0.1)    
                ax2.set_xlim(-0.2, 0.7)    

            def spike_distribution(ax1, ax2):
                # spike distribution
                early_spikes = [one_trial.potential.spikes.filter("early_spike") for one_trial in puff_trials_500ms]
                sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike") for one_trial in puff_trials_500ms]
                def unit_scatter_spike(spikes: list[Events], ax: plt.Axes, **kwargs):
                    median_time = [np.median(spike.t) if len(spike) > 0 else np.nan for spike in spikes]
                    spike_num = [len(spike) for spike in spikes]
                    ax.scatter(median_time, spike_num, **kwargs)
                unit_scatter_spike(early_spikes, ax1, facecolors=EARLY_SPIKE_COLOR, s=1, alpha=0.6, edgecolors='none')
                unit_scatter_spike(sustained_spikes, ax2, facecolors=SUSTAINED_SPIKE_COLOR, s=1, alpha=0.6, edgecolors='none')
                for ax in [ax1, ax2]:
                    ax.spines[['right', 'top',]].set_visible(False)
                    ax.set_xlabel(f"Time [s]")
                    ax.set_ylabel(f"Num. of Spikes")
                    ax.axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)    
                ax1.set_title(f"Early Spikes")
                ax2.set_title(f"Sustained Spikes")
                ax1.set_xlim(-0.05, 0.05)    
                ax2.set_xlim(-0.1, 0.6)    


            def spike_waveform(ax):                
                potential_timeseries = session_node.potential.aspect('raw')
                spike_timeseries = potential_timeseries.batch_segment(
                    session_node.potential.spikes.t, CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
                if len(spike_timeseries) == 0:
                    return
                grouped_spike_timeseries = grouping_timeseries(
                    [TimeSeries(v=zscore(ts.v), t=ts.t) for ts in spike_timeseries], 
                    interp_method="linear")

                oreo_plot(ax, grouped_spike_timeseries, 0, 1, {"color": SPIKE_COLOR_SCHEME['spike']}, FILL_BETWEEN_STYLE)
                ax.set_title("Waveform")
                ax.spines[['right', 'top',]].set_visible(False)
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("z-scored amplitude")

            convolved_potential(axp[4])
            LFP_example(axp[5], axp[6])
            # spike_distribution(axp[7], axp[8])
            spike_waveform(axp[7])
            
            
        
        fig.savefig(os.path.join("tmp_42", f"{dataset.name}_{batch_idx}.png"), dpi=900)
        plt.close(fig)


def plot_all_answer(fa):    
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
        answer_of_everything(dataset, fa)


def main():
    feature_matrix, cohort_names = load_all_datasets_pkl()
    print(feature_matrix.shape, cohort_names)
    fa = get_fa(feature_matrix)
    # plot_factor_overview(fa, feature_matrix, cohort_names)
    # plot_factor_detail(fa, feature_matrix, cohort_names)
    plot_factor_score_heatmap(fa, feature_matrix, cohort_names)
    # get_umap_embedding(fa, feature_matrix, cohort_names)
    # plot_all_answer(fa)

if __name__ == "__main__":
    main()