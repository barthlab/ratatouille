from collections import defaultdict
import os
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator import select_trial_rules
from kitchen.operator.grouping import grouping_events_histogram, grouping_events_rate, grouping_timeseries
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_PASSIVEPUFF_RULES, PREDEFINED_TRIAL_RULES
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.color_scheme import EARLY_SPIKE_COLOR, PUFF_COLOR, SPIKE_COLOR_SCHEME, SUSTAINED_SPIKE_COLOR
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_potential
from kitchen.plotter.utils.fill_plot import oreo_plot, sushi_plot
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, SPIKE_ANNOTATION_EARLY_WINDOW, SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, WC_CONVERT_FLAG
from kitchen.structure.hierarchical_data_structure import Node, DataSet
from kitchen.structure.neural_data_structure import Events, TimeSeries
from kitchen.utils.sequence_kit import select_from_value
from kitchen.utils.numpy_kit import zscore
from kitchen.settings.timeline import ALIGNMENT_STYLE
from kitchen.plotter.decorators.default_decorators import default_style
from kitchen.plotter.ax_plotter.basic_plot import beam_view, stack_view
from kitchen.configs.naming import get_node_name
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, PARALLEL_Y_INCHES, POTENTIAL_RATIO, UNIT_X_INCHES, UNIT_Y_INCHES, ZOOMED_Y_INCHES
from kitchen.configs import routing

from typing import Optional
from functools import partial
import logging

logger = logging.getLogger(__name__)

# plotting manual
plot_manual_spike4Hz = PlotManual(potential=4,)
plot_manual_spike300Hz = PlotManual(potential=300.)


def single_cell_session_parallel_view_jux(
        node: Node,
        dataset: DataSet,
        prefix_keyword: Optional[str] = None,
) -> None:
    subtree = dataset.subtree(node)
    node_name = get_node_name(node)

    # filter out trial types that cannot be plotted
    trial_types = select_from_value(subtree.rule_based_selection(PREDEFINED_PASSIVEPUFF_RULES),
                                    _self = lambda x: CHECK_PLOT_MANUAL(x, plot_manual_spike300Hz))

    n_trial_types = len(trial_types)
    if n_trial_types == 0:
        logger.debug(f"Cannot plot trial parallel for {node_name}: no trial type found")
        return
    
    prefix_str = f"{prefix_keyword}_ParallelSummary" if prefix_keyword is not None else "ParallelSummary"
    alignment_events = ("VerticalPuffOn",)
    default_style(
        mosaic_style=[[f"{node_name}\nHPF 4Hz\n{trial_type}",
                       f"{node_name}\nHPF 300Hz\n{trial_type}",
                       ] for trial_type in trial_types.keys()],
        content_dict={
            f"{node_name}\nHPF 4Hz\n{trial_type}": (
                partial(beam_view, plot_manual=plot_manual_spike4Hz, sync_events=alignment_events),
                trial_dataset)
            for trial_type, trial_dataset in trial_types.items()
            } | {
            f"{node_name}\nHPF 300Hz\n{trial_type}": (
                partial(beam_view, plot_manual=plot_manual_spike300Hz, sync_events=alignment_events),
                trial_dataset)
            for trial_type, trial_dataset in trial_types.items()
            },
        figsize=(n_trial_types * 2 * UNIT_X_INCHES, PARALLEL_Y_INCHES),
        save_path=routing.default_fig_path(node, prefix_str + "_{}.png"),
        plot_settings={
            f"{node_name}\nHPF 4Hz\n{trial_type}": {"set_xlim": (-0.3, 0.8)}
            for trial_type in trial_types.keys()
        } | {
            f"{node_name}\nHPF 300Hz\n{trial_type}": {"set_xlim": (-1.2, 2.7)}
            for trial_type in trial_types.keys()
        },
        sharex=False,
        auto_yscale=False,
    )


def subtree_summary_trial_avg_jux(
        root_node: Node,
        dataset: DataSet,
        prefix_keyword: Optional[str] = None,
) -> None:
    """
    Generate trial-averaged plots organized by FOV days and trial types.

    Creates multi-panel figures in a grid layout where rows represent FOV days
    and columns represent trial types.

    Args:
        fov_node (Fov): FOV node defining the spatial scope for day and trial selection.
        dataset (DataSet): Complete dataset used to find FOV days and trials.
        plot_manual (PlotManual): Configuration specifying which data modalities to include.

    Example:
        >>> from kitchen.plotter.plotting_manual import PlotManual
        >>> plot_config = PlotManual(timeline=True, fluorescence=True, lick=True)
        >>> fov_trial_avg_default(my_fov_node, complete_dataset, plot_config)
    """
    max_n_col = 0
    enumerate_nodes = dataset.subtree(root_node).select("cellsession")

    prefix_str = f"{prefix_keyword}_SubtreeSummary" if prefix_keyword is not None else "SubtreeSummary"
    alignment_events = ("VerticalPuffOn",)


    total_mosaic, content_dict, plot_settings = [], {}, {}
    for row_node in enumerate_nodes:
        subtree = dataset.subtree(row_node)                
        all_possible_trial_types = subtree.rule_based_selection(PREDEFINED_PASSIVEPUFF_RULES)

        # filter out trial types that cannot be plotted
        trial_types = select_from_value(all_possible_trial_types,
                                        _self = lambda dataset: CHECK_PLOT_MANUAL(dataset, plot_manual_spike4Hz))

        # update total_mosaic and content_dict
        total_mosaic.append([f"{get_node_name(row_node)}\nHPF 4Hz\n{trial_type}"
                            for trial_type in trial_types.keys()] + 
                            [f"{get_node_name(row_node)}\nHPF 300Hz\n{trial_type}"
                            for trial_type in trial_types.keys()])
        content_dict.update({
            f"{get_node_name(row_node)}\nHPF 4Hz\n{trial_type}": (
                partial(stack_view, plot_manual=plot_manual_spike4Hz, sync_events=alignment_events),
                trial_dataset)
            for trial_type, trial_dataset in trial_types.items()
            } | {
            f"{get_node_name(row_node)}\nHPF 300Hz\n{trial_type}": (
                partial(stack_view, plot_manual=plot_manual_spike300Hz, sync_events=alignment_events),
                trial_dataset)
            for trial_type, trial_dataset in trial_types.items()
            })
        plot_settings.update({
            f"{get_node_name(row_node)}\nHPF 4Hz\n{trial_type}": {"set_xlim": (-0.3, 0.8)}
            for trial_type in trial_types.keys()
        } | {
            f"{get_node_name(row_node)}\nHPF 300Hz\n{trial_type}": {"set_xlim": (-1.2, 2.7)}
            for trial_type in trial_types.keys()
        })
        max_n_col = max(max_n_col, len(trial_types) * 2)
        
    for i in range(len(total_mosaic)):
        total_mosaic[i] += ["."] * (max_n_col - len(total_mosaic[i]))

    default_style(
        mosaic_style=total_mosaic,
        content_dict=content_dict,
        figsize=(max_n_col * UNIT_X_INCHES, ZOOMED_Y_INCHES * len(total_mosaic)),
        save_path=routing.default_fig_path(root_node, prefix_str + "_{}.png"),
        plot_settings=plot_settings,
        sharex=False,
    )


def raster_plot(        
        node: Node,
        dataset: DataSet,
        prefix_keyword: Optional[str] = None,

        BINSIZE = 20/1000,  # s
):
    subtree = dataset.subtree(node)
    node_name = get_node_name(node)

    # filter out trial types that cannot be plotted
    puff_trials_500ms = subtree.select("trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,)

    prefix_str = f"{prefix_keyword}_RasterPlot" if prefix_keyword is not None else "RasterPlot"
    alignment_events = ("VerticalPuffOn",)

    if len(puff_trials_500ms) == 0:
        logger.debug(f"Cannot plot raster plot for {node_name}: no puff trials found. Skip...")
        return

    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })
    fig, axs = plt.subplots(2, 1, sharex='col', sharey='row', height_ratios=[3, 1], constrained_layout=True)

    puff_trials_500ms = sync_nodes(puff_trials_500ms, alignment_events, plot_manual_spike300Hz)
    if len(puff_trials_500ms) == 0:
        logger.debug(f"Cannot plot raster plot for {node_name}: no puff trials found after syncing. Skip...")
        return
    
    for trial_id, single_trial in enumerate(puff_trials_500ms):
        spikes_in_trial_by_type = {}
        for spike_time, spike_type in zip(single_trial.potential.spikes.t, single_trial.potential.spikes.v):
            spikes_in_trial_by_type.setdefault(spike_type, []).append(spike_time)
            
        for spike_type, spike_times in spikes_in_trial_by_type.items():
            axs[0].eventplot(spike_times,                    
                            lineoffsets=trial_id,   
                            colors=SPIKE_COLOR_SCHEME[spike_type],
                            alpha=0.9,
                            linewidths=0.5,
                            linelengths=0.8)   
        
        puff_on_t = single_trial.data.timeline.filter("VerticalPuffOn").t[0]
        puff_off_t = single_trial.data.timeline.filter("VerticalPuffOff").t[0]
        axs[0].add_patch(ptchs.Rectangle((puff_on_t, (trial_id - 0.5)), puff_off_t - puff_on_t, 1,
                                         facecolor=PUFF_COLOR, alpha=0.5, edgecolor='none', zorder=-10))
    
    group_spikes = grouping_events_rate([single_trial.potential.spikes for single_trial in puff_trials_500ms],
                                        bin_size=BINSIZE, use_event_value_as_weight=False)

    # Only plot if we have spike data
    if len(group_spikes.t) > 0:
        bin_edges = np.concatenate([group_spikes.t, [group_spikes.t[-1] + BINSIZE]]) - BINSIZE/2
        axs[1].stairs(group_spikes.mean, bin_edges, fill=True, color='black')
    # upper_bound = mean_rate + group_spikes.variance
    # lower_bound = mean_rate - group_spikes.variance
    # axs[1].fill_between(bin_edges[:-1], lower_bound, upper_bound, alpha=0.4, lw=0, color='gray', step='post')
    axs[1].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)

    axs[1].spines[['right', 'top', 'left']].set_visible(False)
    axs[0].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    axs[1].set_xlabel(f"Time [s]")
    axs[0].set_xlim(-0.5, 1.0)
    
    axs[0].set_yticks([])
    axs[0].yaxis.set_major_locator(MultipleLocator(5))
    axs[0].set_xticks([])
    
    axs[1].set_ylim(0, 150)
    axs[0].set_ylim(-0.5, len(puff_trials_500ms)-0.5)
    axs[0].set_title(f"{node_name}")
    fig.set_size_inches(1.5, 1.5)
    
    fig.savefig(routing.default_fig_path(node, prefix_str + "_{}.png"), dpi=900, transparent=True)
    plt.close(fig)
    
    logger.info("Plot saved to" + routing.default_fig_path(node, prefix_str + "_{}.png"))



def multiple_dataset_dendrogram_plot(        
        datasets: list[DataSet],
        prefix_keyword: str,
        save_name: Optional[str] = None,

        BINSIZE = 25/1000,  # s
        FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
        LINKAGE_METHOD = 'single',  # 'single', 'complete', 'average', 'ward', 'centroid', 'median', 'weighted'
        LINKAGE_METRIC = 'cosine',  # 'cosine', 'euclidean', 'correlation', 
        PREPROCESSING_METHOD = 'log-scale',  # 'raw', 'log-scale', 'z-score', 'fold-change' 
):
    alignment_events = ("VerticalPuffOn",)

    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import matplotlib.colors
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5,     # Figure title (suptitle)
        'lines.linewidth': 0.5,    # Line width
    })
    all_spikes_histogram, all_cohort_names, all_baseline_fr = [], [], []
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for dataset_idx, dataset in enumerate(datasets):
        for cell_node in dataset.select("cellsession"):        
            subtree = dataset.subtree(cell_node)
            
            puff_trials_500ms = subtree.select(
                "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52, _empty_warning=False,)
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
            all_baseline_fr.append(np.mean(spikes_histogram.mean[spikes_histogram.t < 0.]))

    raw_feature_matrix = np.stack(all_spikes_histogram, axis=0)

    all_baseline_fr = np.array(all_baseline_fr)
    if PREPROCESSING_METHOD == 'log-scale':
        feature_matrix = np.log(raw_feature_matrix + 1)
        heatmap_kws = {"cmap": 'viridis',}
    elif PREPROCESSING_METHOD == 'z-score':
        feature_matrix = zscore(raw_feature_matrix, axis=1)
        heatmap_kws = {"cmap": 'coolwarm', "center": 0.,}
    elif PREPROCESSING_METHOD == 'baseline-subtraction':
        feature_matrix = raw_feature_matrix - all_baseline_fr[:, None]
        heatmap_kws = {"cmap": 'coolwarm', "center": 0.,}
    elif PREPROCESSING_METHOD == 'raw':
        feature_matrix = raw_feature_matrix
        heatmap_kws = {"cmap": 'viridis',}
    elif PREPROCESSING_METHOD == 'baseline-rescaling':
        feature_matrix = np.asinh(raw_feature_matrix - all_baseline_fr[:, None])
        heatmap_kws = {"cmap": 'coolwarm', "center": 0.,}
    elif PREPROCESSING_METHOD == 'baseline-normalization':
        feature_matrix = zscore(raw_feature_matrix - all_baseline_fr[:, None], axis=1)
        heatmap_kws = {"cmap": 'viridis', }
    else:
        raise ValueError(f"Unknown preprocessing method: {PREPROCESSING_METHOD}")
        
    linked = linkage(feature_matrix, method=LINKAGE_METHOD, metric=LINKAGE_METRIC)

    fig, axs = plt.subplots(1, 3, width_ratios=[0.5, 1, 0.2], constrained_layout=True)
    
    set_link_color_palette(['red', 'blue', '#b2df8a', '#33a02c'])
    dendro_info = dendrogram(linked, ax=axs[0], orientation='left')
    axs[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    axs[0].tick_params(axis='y', which='both', length=0, labelright=False, )
    axs[0].set_xticks([])
    axs[0].set_ylabel('Hierarchical Dendrogram')
    axs[0].set_xlabel(f'Distance ({LINKAGE_METHOD})')

    sorted_feature_matrix = feature_matrix[dendro_info['leaves'], :]   
    sorted_cohort_names = [all_cohort_names[i] for i in dendro_info['leaves']]
    print(f"Data range being plotted: {sorted_feature_matrix.min()} to {sorted_feature_matrix.max()}")

    # sorted_feature_matrix = feature_matrix[dendro_info['leaves'], :]
    sns.heatmap(sorted_feature_matrix.repeat(10, axis=0), 
                ax=axs[1], cbar_ax=axs[2], 
                **heatmap_kws,
                cbar_kws={'label': f"Firing Rate ({PREPROCESSING_METHOD})"},
                )
    axs[1].set_xlabel("Time [s]")
    axs[1].sharey(axs[0])

    plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
    axs[1].set_xticks(plot_tick_indices, [0, 0.5])
    
    # set y ticks to cohort names
    tick_positions = np.arange(len(sorted_cohort_names)) * 10 + 5
    axs[1].set_yticks(tick_positions, sorted_cohort_names, rotation=0, fontsize=1.)
    axs[1].tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
    for tick_label in axs[1].get_yticklabels():
        text = tick_label.get_text()
        tick_label.set_color(COHORT_COLORS[text])

    fig.suptitle(f"{prefix_keyword}\n {save_name}")
    fig.set_size_inches(3, 2)
    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])), 
                             prefix_keyword, f"Dendrogram_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to" + save_path)




def multiple_dataset_decomposition_plot(     
        prefix_keyword: str,
        save_name: str,
        dir_save_path: str,

        BINSIZE = 25/1000,  # s
        FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
        PREPROCESSING_METHOD = 'log-scale',  # 'raw', 'log-scale', 'z-score', 'fold-change' 
        DECOMPOSITION_METHOD = 'PCA',  # 'PCA', 'SVD', 'FA', 'SparsePCA', 'NMF', 'ICA'
        n_components = 5,
):
    alignment_events = ("VerticalPuffOn",)

    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import matplotlib.colors
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
    import pandas as pd
    from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, SparsePCA, NMF, FastICA

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5,     # Figure title (suptitle)
        'lines.linewidth': 0.5,    # Line width
    })
    
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    pkl_name = f"feature_matrix_{FEATURE_RANGE[0]}_{FEATURE_RANGE[1]}_{BINSIZE}.npy"
    pkl_path = os.path.join(dir_save_path, "feature_matrix_pkl", pkl_name)
    if not os.path.exists(pkl_path):
        all_spikes_histogram, all_cohort_names, all_baseline_fr = [], [], []


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
                    "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52, _empty_warning=False,)
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
                all_baseline_fr.append(np.mean(spikes_histogram.mean[spikes_histogram.t < 0.]))

        raw_feature_matrix = np.stack(all_spikes_histogram, axis=0)
        all_baseline_fr = np.array(all_baseline_fr)
        np.save(pkl_path, {"feature_matrix": raw_feature_matrix, "cohort_names": all_cohort_names, "baseline_fr": all_baseline_fr})
        logger.info("Feature matrix saved to " + pkl_path)
    tmp_save = np.load(pkl_path, allow_pickle=True).item()
    raw_feature_matrix, all_cohort_names, all_baseline_fr = tmp_save["feature_matrix"], tmp_save["cohort_names"], tmp_save["baseline_fr"]
    print(f"Successfully loaded feature matrix from {pkl_path}")
    print(f"Feature matrix shape: {raw_feature_matrix.shape}")

    if PREPROCESSING_METHOD == 'log-scale':
        feature_matrix = np.log(raw_feature_matrix + 1)
        heatmap_kws = {"cmap": 'viridis',}
    elif PREPROCESSING_METHOD == 'z-score':
        feature_matrix = zscore(raw_feature_matrix, axis=1)
        heatmap_kws = {"cmap": 'viridis', "vmin": -1, "vmax": 3,}
    elif PREPROCESSING_METHOD == 'baseline-subtraction':
        feature_matrix = raw_feature_matrix - all_baseline_fr[:, None]
        heatmap_kws = {"cmap": 'viridis',}
    elif PREPROCESSING_METHOD == 'raw':
        feature_matrix = raw_feature_matrix
        heatmap_kws = {"cmap": 'viridis', "vmin": 0, "vmax": 200,}
    elif PREPROCESSING_METHOD == 'baseline-rescaling':
        feature_matrix = np.asinh(raw_feature_matrix - all_baseline_fr[:, None])
        heatmap_kws = {"cmap": 'viridis', }
    else:
        raise ValueError(f"Unknown preprocessing method: {PREPROCESSING_METHOD}")
    

    if DECOMPOSITION_METHOD == 'PCA':
        solver = PCA(n_components=n_components, random_state=42)
    elif DECOMPOSITION_METHOD == 'SVD':
        solver = TruncatedSVD(n_components=n_components, random_state=42)
    elif DECOMPOSITION_METHOD == 'FA':
        solver = FactorAnalysis(n_components=n_components, rotation='varimax', random_state=42)
    elif DECOMPOSITION_METHOD == 'SparsePCA':
        solver = SparsePCA(n_components=n_components, random_state=42)
    elif DECOMPOSITION_METHOD == 'NMF':
        solver = NMF(n_components=n_components, random_state=42)
    elif DECOMPOSITION_METHOD == 'ICA':
        solver = FastICA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown decomposition method: {DECOMPOSITION_METHOD}")
    
    if np.min(feature_matrix) < 0 and DECOMPOSITION_METHOD in ('NMF', ):
        return
    component_projection = solver.fit_transform(feature_matrix)
    total_reconstruction = np.dot(component_projection, solver.components_)

    # --- Figure 1: Detailed Component Breakdown ---
    fig, axs = plt.subplots(6, n_components, figsize=(2*n_components, 8), constrained_layout=True,
                            height_ratios=[1.5, 1.5, 2, 2, 2, 2], )
    for component_id in range(n_components):
        axc = axs[:, component_id]
        component = solver.components_[component_id, :]
        component_weights = component_projection[:, component_id]

        component_reconstruction = np.outer(component_weights, component)
        cumulative_reconstruction = np.dot(component_projection[:, :component_id+1], solver.components_[:component_id+1, :])
        cumulative_error = np.abs(feature_matrix - cumulative_reconstruction)

        axc_comp = axc[0]
        axc_comp.plot(bin_centers, component, lw=0.75, color='black')
        axc_comp.set_title(f"Component {component_id+1}")
        axc_comp.axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
        axc_comp.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
        axc_comp.spines[['right', 'top',]].set_visible(False)
        axc_comp.set_xlabel("Time [s]")
        axc_comp.set_ylabel("Component Amplitude [a.u.]")
        
        axc_weight = axc[1]
        data = pd.DataFrame({
            "Weight": component_weights,
            "Cohort": all_cohort_names,
        })
        sns.boxplot(data=data, x="Cohort", y="Weight", hue="Cohort", ax=axc_weight, 
                    palette=COHORT_COLORS,  
                    order=sorted(set(all_cohort_names)),
                    showfliers=False, zorder=10,
                    medianprops={'color': 'k', 'ls': '-', 'lw': 1.5, },
                    # medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    showbox=False,
                    showcaps=False,
                    width=0.5,
                    )
        sns.stripplot(data=data, x="Cohort", y="Weight", hue="Cohort", ax=axc_weight, 
                    palette=COHORT_COLORS, alpha=0.7, jitter=0.25, size=4,
                    order=sorted(set(all_cohort_names)))
        axc_weight.spines[['right', 'top',]].set_visible(False)
        axc_weight.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)

        sorted_idx = np.argsort(component_weights)
        sns.heatmap(component_reconstruction[sorted_idx, :], ax=axc[2], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(cumulative_reconstruction[sorted_idx, :], ax=axc[3], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(feature_matrix[sorted_idx, :], ax=axc[4], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(cumulative_error[sorted_idx, :], ax=axc[5], 
                    cbar=False, cmap='Reds', alpha=0.5)
        
        for ax in axc[2:]:
            ax.set_xlabel("Time [s]")

            plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
            ax.set_xticks(plot_tick_indices, [0, 0.5], rotation=0, )
            
            # set y ticks to cohort names
            tick_positions = np.arange(len(all_cohort_names))
            ax.set_yticks(tick_positions + 0.5, [all_cohort_names[i] for i in sorted_idx], rotation=0, fontsize=0.5)
            ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
            for tick_label in ax.get_yticklabels():
                text = tick_label.get_text()
                tick_label.set_color("white")
                tick_label.set_bbox({
                    'facecolor': COHORT_COLORS[text], 
                    'alpha': 1.0,             
                    'edgecolor': 'none',        
                    'boxstyle': 'square,pad=0.3' 
                })

    save_path = os.path.join(dir_save_path, prefix_keyword, f"Components_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



    # --- Figure 2: Summary Plots ---
    fig = plt.figure(figsize=(5, 2.5), constrained_layout=True, )
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2, projection='3d')]

    ax_weight_2d = axs[0]
    ax_weight_3d = axs[1]
    unique_cohorts = sorted(list(set(all_cohort_names)))
    for cohort in unique_cohorts:
        idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
        ax_weight_2d.scatter(
            component_projection[idx, 0], component_projection[idx, 1],
            color=COHORT_COLORS.get(cohort, 'gray'),
            label=cohort, s=15, alpha=0.8, edgecolors=COHORT_edgecolors.get(cohort, 'none'), lw=0.2,
        )
        ax_weight_3d.scatter(
            component_projection[idx, 0], component_projection[idx, 1], all_baseline_fr[idx],
            color=COHORT_COLORS.get(cohort, 'gray'),
            label=cohort, s=8, alpha=0.8, edgecolors=COHORT_edgecolors.get(cohort, 'none'), lw=0.2,
            )
    ax_weight_2d.set_xlabel("Component 1 [a.u.]")
    ax_weight_2d.set_ylabel("Component 2 [a.u.]")
    ax_weight_3d.set_xlabel("Component 1 [a.u.]")
    ax_weight_3d.set_ylabel("Component 2 [a.u.]")
    ax_weight_3d.set_zlabel("Baseline FR [Hz]")

    ax_weight_2d.set_aspect('equal')
    ax_weight_3d.view_init(elev=20, azim=225)
    ax_weight_2d.legend(title="Cohort", loc='best', frameon=False)
    ax_weight_3d.legend(title="Cohort", loc='best', frameon=False)
    ax_weight_2d.spines[['right', 'top']].set_visible(False)
    save_path = os.path.join(dir_save_path, prefix_keyword, f"Overview_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    # --- Figure 3: UMAP Decomposition ---
    from umap import UMAP
    n_umap_examples = 5
    n_neighbor_options = (5, 6, 7, 8, 9)
    fig, axs = plt.subplots(2*len(n_neighbor_options), n_umap_examples, 
                            figsize=(2*n_umap_examples, 4*len(n_neighbor_options)), constrained_layout=True)
    for random_state in range(n_umap_examples):
        for row_idx,n_neighbors in enumerate(n_neighbor_options):
            axwo = axs[row_idx, random_state]
            axw = axs[row_idx + len(n_neighbor_options), random_state]
            reducer_wo_baseline = UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, n_jobs=-1)
            reducer_w_baseline = UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, n_jobs=-1)
            weights_wo_baseline = zscore(component_projection[:, :n_components], axis=0)
            weights_w_baseline = zscore(np.concatenate((component_projection[:, :n_components], all_baseline_fr[:, None]), axis=1), axis=0)
            embedding_wo_baseline = reducer_wo_baseline.fit_transform(weights_wo_baseline)
            embedding_w_baseline = reducer_w_baseline.fit_transform(weights_w_baseline, axis=1)
            
            for cohort in unique_cohorts:
                idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
                axwo.scatter(
                    embedding_wo_baseline[idx, 0], embedding_wo_baseline[idx, 1],
                    color=COHORT_COLORS.get(cohort, 'gray'),
                    label=cohort, s=8, alpha=0.8, edgecolors=COHORT_edgecolors.get(cohort, 'none'), lw=0.2,
                )
                axw.scatter(
                    embedding_w_baseline[idx, 0], embedding_w_baseline[idx, 1],
                    color=COHORT_COLORS.get(cohort, 'gray'),
                    label=cohort, s=8, alpha=0.8, edgecolors=COHORT_edgecolors.get(cohort, 'none'), lw=0.2,
                )
                axwo.set_title(f"UMAP (w/o Baseline, RS={random_state} n_neighbors={n_neighbors})")
                axw.set_title(f"UMAP (w/ Baseline, RS={random_state} n_neighbors={n_neighbors})")
        for ax in axs[:, random_state]:
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(title="Cohort", loc='best', frameon=False)
            ax.spines[['right', 'top']].set_visible(False)
 
    save_path = os.path.join(dir_save_path, prefix_keyword, f"UMAP_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    # --- Figure 4: Linkage Dendrogram ---
    from scipy.cluster.hierarchy import linkage, dendrogram
    fig, axs = plt.subplots(2, 4, figsize=(6, 4), constrained_layout=True, width_ratios=[0.3, 1, 1, 0.2])

    weights_wo_baseline = zscore(component_projection[:, :n_components], axis=0)
    weights_w_baseline = zscore(np.concatenate((component_projection[:, :n_components], all_baseline_fr[:, None]), axis=1), axis=0)
    linked_wo_baseline = linkage(weights_wo_baseline, method='ward', metric='euclidean')
    linked_w_baseline = linkage(weights_w_baseline, method='ward', metric='euclidean')

    for axh, linked, weights, xticklabels, tag in zip(
        [axs[0, :], axs[1, :]], 
        [linked_wo_baseline, linked_w_baseline], 
        [weights_wo_baseline, weights_w_baseline],
        [[f"Comp {i+1}" for i in range(n_components)], 
        [f"Comp {i+1}" for i in range(n_components)] + ["Baseline FR",]],
        ["wo_baseline", "w/_baseline"],
        ):
        dendro_info = dendrogram(linked, ax=axh[0], orientation='left')
        axh[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        axh[0].tick_params(axis='y', which='both', length=0, labelright=False, )
        axh[0].set_xticks([])
        axh[0].set_ylabel('Hierarchical Dendrogram')
        axh[0].set_xlabel('Ward Distance')

        sorted_idx = dendro_info['leaves']
        sns.heatmap(feature_matrix[sorted_idx, :].repeat(10, axis=0), 
                    ax=axh[1], **heatmap_kws,
                    )
        sns.heatmap(total_reconstruction[sorted_idx, :].repeat(10, axis=0), 
                    ax=axh[2], **heatmap_kws,
                    )
        axh[1].sharey(axh[0])
        axh[2].sharey(axh[0])
        axh[1].set_title(f'Original Data ({tag})')
        axh[2].set_title(f'Reconstructed Data ({tag})')
        for ax in axh[1:3]:
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xlabel("Time [s]")
            plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
            ax.set_xticks(plot_tick_indices, [0, 0.5], rotation=0, )

        sns.heatmap(weights[sorted_idx, :].repeat(10, axis=0), 
                    ax=axh[3], cmap='bwr', center=0.,
                    )
        axh[3].sharey(axh[0])
        axh[3].set_ylabel('')
        axh[3].set_xticks(np.arange(len(xticklabels)) + 0.5, xticklabels, rotation=45, ha='right',)

        # set y ticks to cohort names
        for ax in axh[1:]:
            tick_positions = np.arange(len(all_cohort_names)) * 10 + 5
            ax.set_yticks(tick_positions + 0.5, [all_cohort_names[i] for i in sorted_idx], rotation=0, fontsize=0.2)
            ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
            for tick_label in ax.get_yticklabels():
                text = tick_label.get_text()
                tick_label.set_color("white")
                tick_label.set_bbox({
                    'facecolor': COHORT_COLORS[text], 
                    'alpha': 1.0,             
                    'edgecolor': 'none',        
                    'boxstyle': 'square,pad=0.3' 
                })
    fig.suptitle(f"{prefix_keyword}\n {save_name}")
    save_path = os.path.join(dir_save_path, prefix_keyword, f"Dendrogram_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


   

    # --- Figure 6: UMAP clustering ---
    from sklearn.cluster import SpectralClustering
    import matplotlib.patches as mpatches
    if n_components == 2:
        umap_kws = {'n_components': 2, 'random_state': 0, 'n_neighbors': 7}
        n_clusters = 4
    elif n_components == 5:
        umap_kws = {'n_components': 2, 'random_state': 3, 'n_neighbors': 6}
        n_clusters = 6
    else:
        raise ValueError(f"n_components {n_components} not supported")
    weights_w_baseline = zscore(np.concatenate((component_projection[:, :n_components], all_baseline_fr[:, None]), axis=1), axis=0)
    embedding = UMAP(**umap_kws).fit_transform(weights_w_baseline)
    cluster_id = SpectralClustering(n_clusters=n_clusters, n_neighbors=umap_kws['n_neighbors'], random_state=42).fit_predict(embedding)

    cluster_names = [f"cluster {i}" for i in cluster_id]   

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

        ell = Ellipse(xy=position, width=4 * width, height=2.5 * height, angle=angle, **kwargs)
        
        ax.add_patch(ell)
        return ell

    fig, axs = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True, width_ratios=[1, 1, 1])

    legend_handles = []
    for cluster_id in range(n_clusters):
        idx = [i for i, name in enumerate(cluster_names) if name == f"cluster {cluster_id}"]
        dot_cohort_colors = [COHORT_COLORS[all_cohort_names[i]] for i in idx]
        dot_cluster_colors = [CLUSTER_COLORS[f"cluster {cluster_id}"] for i in idx]
        axs[0].scatter(embedding[idx, 0], embedding[idx, 1],
                       facecolors=dot_cohort_colors, edgecolors=dot_cluster_colors, lw=0.1, s=3, alpha=0.8)
        draw_ellipse(embedding[idx, :], axs[0], edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"], 
                     facecolor='none', lw=1, ls='--', alpha=0.8, zorder=-10)
        legend_patch = mpatches.Patch(color=CLUSTER_COLORS[f"cluster {cluster_id}"], alpha=0.5, lw=0, label=f'cluster {cluster_id}')
        legend_handles.append(legend_patch)
        
    axs[0].legend(handles=legend_handles, title="Cluster", loc='best', frameon=False)
    axs[0].set_title("UMAP of Factor Scores")
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel("UMAP 1")
    axs[0].set_ylabel("UMAP 2")

    df = pd.DataFrame({
        "Cohort": all_cohort_names,
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

    save_path = os.path.join(dir_save_path, prefix_keyword, f"Clusters_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)
    


    # --- Figure 5: Another Heatmap ---
    fig, axs = plt.subplots(2, 3, figsize=(4, 4), constrained_layout=True, width_ratios=[0.3, 1, 0.2])

    weights_w_baseline = zscore(np.concatenate((component_projection[:, :n_components], all_baseline_fr[:, None]), axis=1), axis=0)
    linked_w_baseline = linkage(weights_w_baseline, method='ward', metric='euclidean')
    dendro_info = dendrogram(linked_w_baseline, ax=axs[0, 0], orientation='left')
    dendro_info = dendrogram(linked_w_baseline, ax=axs[1, 0], orientation='left')
    for ax in axs[:, 0]:
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='y', which='both', length=0, labelright=False, )
        ax.set_xticks([])
        ax.set_ylabel('Hierarchical Dendrogram')
        ax.set_xlabel('Ward Distance')
        
        tick_positions = np.arange(len(all_cohort_names)) * 10 + 5
        ax.set_yticks(tick_positions, [cluster_names[i] for i in dendro_info['leaves']], rotation=0, fontsize=0.2)
        ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
        for tick_label in ax.get_yticklabels():
            text = tick_label.get_text()
            tick_label.set_color("white")
            tick_label.set_bbox({'facecolor': CLUSTER_COLORS[text], 'alpha': 1.0, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.3'})

    sorted_idx = dendro_info['leaves']
    sns.heatmap(feature_matrix[sorted_idx, :].repeat(10, axis=0), 
                ax=axs[1, 1], **heatmap_kws,
                )
    sns.heatmap(raw_feature_matrix[sorted_idx, :].repeat(10, axis=0), 
                ax=axs[0, 1], cmap='viridis', vmin=0, vmax=200,
                )
    axs[0, 1].sharey(axs[0, 0])
    axs[1, 1].sharey(axs[1, 0])
    axs[0, 1].set_title('Raw FR')
    axs[1, 1].set_title('Z-Scored FR')
    for ax in axs[:, 1]:
        ax.set_ylabel('')
        ax.set_xlabel("Time [s]")
        plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
        ax.set_xticks(plot_tick_indices, [0, 0.5], rotation=0, )

    sns.heatmap(weights_w_baseline[sorted_idx, :].repeat(10, axis=0), 
                ax=axs[1, 2], cmap='bwr', center=0.,
                )
    sns.heatmap(np.concatenate((component_projection[:, :n_components], 
                                all_baseline_fr[:, None]), axis=1)[sorted_idx, :].repeat(10, axis=0), 
                ax=axs[0, 2], cmap='bwr', center=0.,
                )
    axs[0, 2].sharey(axs[0, 0])
    axs[1, 2].sharey(axs[1, 0])
    for ax in axs[:, 2]:
        ax.set_ylabel('')
        ax.set_xticks(np.arange(n_components + 1) + 0.5, [f"Comp {i+1}" for i in range(n_components)] + ["Baseline FR",], rotation=45, ha='right')
    axs[0, 2].set_title('Raw Weights')
    axs[1, 2].set_title('Z-Scored Weights')
    

    # set y ticks to cohort names
    for ax in [axs[1, 1], axs[1, 2], axs[0, 1], axs[0, 2]]:
        tick_positions = np.arange(len(all_cohort_names)) * 10 + 5
        ax.set_yticks(tick_positions + 0.5, [all_cohort_names[i] for i in sorted_idx], rotation=0, fontsize=0.2)
        ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
        for tick_label in ax.get_yticklabels():
            text = tick_label.get_text()
            tick_label.set_color("white")
            tick_label.set_bbox({
                'facecolor': COHORT_COLORS[text], 
                'alpha': 1.0,             
                'edgecolor': 'none',        
                'boxstyle': 'square,pad=0.3' 
            })
    fig.suptitle(f"{prefix_keyword}\n {save_name}")
    save_path = os.path.join(dir_save_path, prefix_keyword, f"Compared2Raw_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

def waveform_plot(        
        dataset: DataSet,
        prefix_keyword: Optional[str] = None,

    
):
    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import matplotlib.colors
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })

    all_waveforms = defaultdict(list)
    for cell_node in dataset.select("cellsession"):    
        potential_timeseries = cell_node.potential.aspect('raw')
        
        spike_timeseries = potential_timeseries.batch_segment(
            cell_node.potential.spikes.t, CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
        if len(spike_timeseries) == 0:
            continue
        grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
        zscored_waveform = np.mean(zscore(grouped_spike_timeseries.raw_array, axis=1), axis=0)
        all_waveforms['spike'].append(TimeSeries(v=zscored_waveform, t=grouped_spike_timeseries.t))

        for spike_type, spike_times in cell_node.potential.spikes.groupby():
            spike_timeseries = potential_timeseries.batch_segment(
                spike_times, CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
            grouped_spike_timeseries = grouping_timeseries(spike_timeseries, interp_method="linear")
            zscored_waveform = np.mean(zscore(grouped_spike_timeseries.raw_array, axis=1), axis=0)
            all_waveforms[spike_type].append(TimeSeries(v=zscored_waveform, t=grouped_spike_timeseries.t))

    fig, axs = plt.subplots(1, len(all_waveforms), sharex=True, sharey=True, constrained_layout=True)
    for ax, spike_type in zip(axs, ("spike", "early_spike", "sustained_spike", "regular_spike")):
        waveforms = all_waveforms[spike_type]
        grouped_waveform = grouping_timeseries(waveforms, interp_method="linear")
        oreo_plot(ax, grouped_waveform, 0, 1, {"color": SPIKE_COLOR_SCHEME[spike_type]}, FILL_BETWEEN_STYLE)
        ax.set_title(spike_type)
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-scored amplitude")
    fig.suptitle(f"{dataset.name}")
    fig.set_size_inches(6, 1.5)
    save_path = routing.smart_path_append(routing.default_fig_path(dataset), "Waveform_{}" + f"_{prefix_keyword}.png")
    
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to" + save_path)



COHORT_COLORS = {
    "SST_JUX": "#ffa000",
    "SST_WC": "#CF8300A4",
    "PV_JUX": "#ff0000",
    "PYR_JUX": "#0000FF",
}
COHORT_edgecolors = {
    "SST_JUX": "none",
    "SST_WC": "black",
    "PV_JUX": "none",
    "PYR_JUX": "none",
}


CLUSTER_COLORS = {
    0: "#991CE3",
    "cluster 0": "#991CE3",
    1: "#66E31C",
    "cluster 1": "#66E31C",
    2: "#E31C66",
    "cluster 2": "#E31C66",
    3: "#E3661C",
    "cluster 3": "#E3661C",
    4: "#1CE3E3",
    "cluster 4": "#1CE3E3",
    5: "#E3E31C",
    "cluster 5": "#E3E31C",
    6: "#584835",
    "cluster 6": "#584835",
    7: "#C73F93",
    "cluster 7": "#C73F93",
}
def feature_overview(
    datasets: list[DataSet],
    prefix_keyword: str,
):
    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import matplotlib.colors
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
    import os # Added import for os
    
    # Assuming 'routing' and 'logger' are defined elsewhere in your project
    # from your_project import routing, logger 
    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 5,      # Legend font size (increased for visibility)
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5,     # Figure title (suptitle)
        'lines.linewidth': 0.5,    # Line width
    })
    all_early_spike_timing, all_cohort_name = [], []
    for dataset_idx, dataset in enumerate(datasets):
        for cell_node in dataset.select("cellsession"):
            tmp_cell_early_spike_timing = []
            for trial_node in dataset.subtree(cell_node).select("trial"):
                aligned_trial_node = trial_node.aligned_to(trial_node.timeline.filter("VerticalPuffOn").t[0])
                early_spike_times = aligned_trial_node.potential.spikes.filter("early_spike").t
                if len(early_spike_times) == 0:
                    continue
                tmp_cell_early_spike_timing.append(np.min(early_spike_times))
            if len(tmp_cell_early_spike_timing) == 0:
                continue
            all_early_spike_timing.append(np.array(tmp_cell_early_spike_timing))
            all_cohort_name.append(dataset.name)
    
    avg_early_spike_timing = np.array([np.mean(x) for x in all_early_spike_timing])
    all_cohort_name = np.array(all_cohort_name)
    sorted_cell_id = np.argsort(avg_early_spike_timing)
    sorted_avg_timing = avg_early_spike_timing[sorted_cell_id]
    sorted_spike_timing = [all_early_spike_timing[i] for i in sorted_cell_id]
    sorted_cohort_name = all_cohort_name[sorted_cell_id]

    # --- Calculate the error (Standard Deviation) for each cell ---
    error_values = np.array([np.std(x) if len(x) > 0 else 0 for x in sorted_spike_timing])
    
    # --- NEW: Create a list of colors corresponding to the sorted cohort names ---
    plot_colors = [COHORT_COLORS[name] for name in sorted_cohort_name]


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    y_pos = np.arange(len(sorted_avg_timing))

    # 1. Plot the horizontal bars with error bars, using cohort colors
    ax.barh(
        y_pos,
        sorted_avg_timing,
        xerr=error_values,
        height=0.9,
        align='center',
        color=plot_colors, # <-- MODIFIED: Use the cohort-specific color list
        alpha=0.7,
        capsize=3,
        error_kw={'ecolor': 'gray', 'elinewidth': 1.5}
    )

    # 2. Plot the individual data points (dots) with jitter, using cohort colors
    jitter_strength = 0.2
    for i, timings in enumerate(sorted_spike_timing):
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(timings))
        ax.scatter(
            timings,
            np.repeat(y_pos[i], len(timings)) + jitter,
            color='black', # <-- MODIFIED: Use the cohort-specific color
            s=2,
            alpha=0.7,
            zorder=2
        )

    # --- Styling and Labels ---
    # --- MODIFIED: Remove y-axis text labels but keep the bar positions ---
    ax.set_yticks(y_pos) # Keep ticks for positioning the bars
    ax.set_yticklabels([]) # Set the labels to an empty list to hide them
    
    # This line is good to keep as it removes the tick marks themselves
    ax.tick_params(axis='y', which='both', length=0)
    
    # The loop for coloring tick labels is no longer needed and has been removed.

    # --- NEW: Add a custom legend for cohort colors ---
    legend_handles = [ptchs.Patch(color=color, label=name) for name, color in COHORT_COLORS.items()]
    ax.legend(handles=legend_handles, title="Cohort", loc='upper right')

    ax.set_xlabel('Early Spike Timing [s]', fontsize=12)

    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
 
    fig.suptitle(f"{prefix_keyword}")
    fig.set_size_inches(7, 6)
    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])), f"{prefix_keyword}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to" + save_path)


def SPONT_FR_EVOKED_FR_SCATTER(
        datasets: list[DataSet]
):

    
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import matplotlib.colors
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
    import os 
    from matplotlib.ticker import AutoMinorLocator, LogLocator

    from kitchen.plotter.plotting_manual import PlotManual
    from kitchen.plotter.ax_plotter.basic_plot import flat_view
    from kitchen.plotter.plotting_params import FLAT_X_INCHES, UNIT_Y_INCHES, DPI
    from kitchen.configs import routing
    from kitchen.settings.fluorescence import DF_F0_SIGN
    from kitchen.configs.naming import get_node_name
    from kitchen.plotter.unit_plotter.unit_trace import  unit_plot_potential_conv
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })

    fig1, ax1 = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)  # spontaneous vs evoked firing rate
    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)  # spontaneous vs early firing rate
    fig3, ax3 = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)  # spontaneous vs sustained firing rate

    fig4, ax4 = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)  # early firing median time vs early firing spike number
    fig5, ax5 = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)  # sustained firing median time vs sustained firing spike number

    for dataset in datasets:
        
        session_nodes = dataset.select("cellsession", 
                                    _self=lambda x: len(dataset.subtree(x).select(
                                        "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,
                                        _empty_warning=False,
                                        )) > 0)
        n_session = len(session_nodes)
        
        for session_node in session_nodes:
            # calculate the number of early, sustained, and regular spikes
            n_early_spikes = np.sum(session_node.potential.spikes.v == "early_spike") 
            n_sustained_spikes = np.sum(session_node.potential.spikes.v == "sustained_spike") 
            n_regular_spikes = np.sum(session_node.potential.spikes.v == "regular_spike") 
            n_evoked_spikes = n_early_spikes + n_sustained_spikes 
            # calculate the total airpuff duration
            airpuff_stim_duration = session_node.data.timeline.filter("VerticalPuffOff").t - session_node.data.timeline.filter("VerticalPuffOn").t            
            total_airpuff_duration = np.sum(airpuff_stim_duration)
            early_window_duration = len(airpuff_stim_duration) * SPIKE_ANNOTATION_EARLY_WINDOW[1]
            # calculate the evoked firing rate
            evoked_fr = n_evoked_spikes / total_airpuff_duration
            early_fr = n_early_spikes / early_window_duration
            sustained_fr = n_sustained_spikes / (total_airpuff_duration - early_window_duration)
            task_start, task_end = session_node.data.timeline.task_time()
            spontaneous_fr = n_regular_spikes / (task_end - task_start - total_airpuff_duration)

            def spikes_temporal_distribution(spikes: list[Events], ax: plt.Axes, **kwargs):
                median_time = [np.median(spike.t) if len(spike) > 0 else np.nan for spike in spikes]
                spike_num = [len(spike) for spike in spikes]
                ax.errorbar(np.nanmean(median_time), np.nanmean(spike_num) + 1, 
                            xerr=np.nanstd(median_time) / np.sqrt(len(median_time)), 
                            yerr=np.nanstd(spike_num) / np.sqrt(len(spike_num)), **kwargs)
                ax.set_yscale("log")

            ax1.scatter(spontaneous_fr, evoked_fr, label=dataset.name, facecolors=COHORT_COLORS[dataset.name], edgecolors='none',
                       s=8, alpha=0.8, clip_on=False)
            ax2.scatter(spontaneous_fr, early_fr, label=dataset.name, facecolors=COHORT_COLORS[dataset.name], edgecolors='none',
                       s=8, alpha=0.8, clip_on=False)
            ax3.scatter(spontaneous_fr, sustained_fr, label=dataset.name, facecolors=COHORT_COLORS[dataset.name], edgecolors='none',
                       s=8, alpha=0.8, clip_on=False)
            
            puff_trials_500ms = dataset.subtree(session_node).select(
                "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52, _empty_warning=False,)
            puff_trials_500ms = sync_nodes(puff_trials_500ms, ("VerticalPuffOn",), plot_manual_spike300Hz)
            early_spikes = [one_trial.potential.spikes.filter("early_spike") for one_trial in puff_trials_500ms]
            sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike") for one_trial in puff_trials_500ms]
            spikes_temporal_distribution(early_spikes, ax4, 
                                         elinewidth=0.5, capsize=0.5, marker='.', markeredgecolor='none', markersize=5,
                                         capthick=0.5,
                                         color=COHORT_COLORS[dataset.name], alpha=0.6, clip_on=False)
            spikes_temporal_distribution(sustained_spikes, ax5,
                                         elinewidth=0.5, capsize=0.5, marker='.', markeredgecolor='none', markersize=5,
                                         capthick=0.5,
                                         color=COHORT_COLORS[dataset.name], alpha=0.6, clip_on=False)

    for ax in [ax1, ax2, ax3]:
        ax.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=-10)
        ax.plot(np.linspace(0.1, 10, 1000), np.linspace(0.1, 10, 1000) * 10, color='gray', linestyle='--', linewidth=1, alpha=0.2, zorder=-10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("Spontaneous Firing Rate [Hz]")
        ax.set_ylabel("Evoked Firing Rate [Hz]")
        ax.set_xscale("symlog", linthresh=1, linscale=0.5)
        ax.set_yscale("symlog", linthresh=1, linscale=0.5)
        ax.set_aspect('equal')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)    
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(LogLocator(subs='all', numticks=30))
        ax.yaxis.set_minor_locator(LogLocator(subs='all', numticks=30))

        legend_handles = [ptchs.Patch(color=color, label=name) for name, color in COHORT_COLORS.items()]
        ax.legend(handles=legend_handles, title="Cohort", loc='best', frameon=False)
    
    for ax in [ax4, ax5]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel("Median Spike Time [s]")
        ax.set_ylabel("Spike Number + 1")
        ax.set_xlim(0, None)
        ax.set_ylim(1, None)
        ax.minorticks_on()

        legend_handles = [ptchs.Patch(color=color, label=name) for name, color in COHORT_COLORS.items()]
        ax.legend(handles=legend_handles, title="Cohort", loc='best', frameon=False)

    save_path = os.path.join(routing.default_fig_path(datasets[0]), "..", )
    fig1.savefig(os.path.join(save_path, "FR_scatter_evoked.png"), dpi=DPI)
    fig2.savefig(os.path.join(save_path, "FR_scatter_early.png"), dpi=DPI)
    fig3.savefig(os.path.join(save_path, "FR_scatter_sustained.png"), dpi=DPI)
    fig4.savefig(os.path.join(save_path, "TVN_early_spike_time_vs_num.png"), dpi=DPI)
    fig5.savefig(os.path.join(save_path, "TVN_sustained_spike_time_vs_num.png"), dpi=DPI)
    logger.info(f"Plot saved to {save_path}")
    
    # Close the figure
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)

    
def AnwerOfEverything(
        dataset: DataSet,
):
    

    
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import matplotlib.colors
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
    import os 

    from kitchen.plotter.plotting_manual import PlotManual
    from kitchen.plotter.ax_plotter.basic_plot import flat_view
    from kitchen.plotter.plotting_params import FLAT_X_INCHES, UNIT_Y_INCHES, DPI
    from kitchen.configs import routing
    from kitchen.settings.fluorescence import DF_F0_SIGN
    from kitchen.configs.naming import get_node_name
    from kitchen.plotter.unit_plotter.unit_trace import  unit_plot_potential_conv

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })


    
    total_session_nodes = dataset.select("cellsession", 
                                   _self=lambda x: len(dataset.subtree(x).select(
                                       "trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,
                                       _empty_warning=False,
                                       )) > 0)
    total_n_session = len(total_session_nodes)
    batch_session_size = 9
    n_batch = int(np.ceil(total_n_session / batch_session_size))
    for batch_idx in range(n_batch):
        session_nodes = total_session_nodes.nodes[batch_idx * batch_session_size: (batch_idx + 1) * batch_session_size]
        n_session = len(session_nodes)

        n_sub_plots = 8
        fig, axs = plt.subplots(n_session, 8, 
                                figsize=(FLAT_X_INCHES * 0.8, UNIT_Y_INCHES * n_session), 
                                width_ratios=[65, ] + [5,] * (n_sub_plots - 1),
                                sharex='col', constrained_layout=True)
        if n_session == 1:
            axs = np.expand_dims(axs, axis=0)
        for row_idx, session_node in enumerate(session_nodes):
            axs[row_idx, 0].spines[['top', 'right']].set_visible(False)
            axs[row_idx, 0].set_title(f"{session_node.session_id}")
            axs[row_idx, 0].set_yticks([])
            axs[row_idx, 0].set_xlabel("Time [s]")
            for col_idx in range(n_sub_plots):
                axs[row_idx, col_idx].tick_params(labelbottom=True)

        plot_coroutines = {flat_view(ax=axs[row_idx, 0], datasets=DataSet(name=session_node.session_id, nodes=[session_node]), 
                            plot_manual=PlotManual(potential_conv=True)): f"session_{row_idx}" 
                        for row_idx, session_node in enumerate(session_nodes)}

        active_coroutines = {}
        for plot_coro, name in plot_coroutines.items():
            try:
                initial_progress = next(plot_coro)
                active_coroutines[plot_coro] = initial_progress
            except StopIteration:
                pass
            except Exception as e:
                raise ValueError(f"Error in {name} at initial progress: {e}")
        
        # Progressive plot functions
        progress = 0
        while active_coroutines:
            plot_heights = list(active_coroutines.values())
            progress += max(plot_heights) + 0.1

            # update the alive coroutines
            next_step_active_coroutines = {}
            for plot_coro, name in active_coroutines.items():
                try:
                    next_progress = plot_coro.send(progress)
                    next_step_active_coroutines[plot_coro] = next_progress
                except StopIteration:
                    pass
                except Exception as e:
                    raise ValueError(f"Error in {name} at progress {progress}: {e}")
            active_coroutines = next_step_active_coroutines    

        # node features
        for row_idx, session_node in enumerate(session_nodes):
            axp = axs[row_idx, 1:]
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
        
            for trial_id, single_trial in enumerate(puff_trials_500ms):
                spikes_in_trial_by_type = {}
                for spike_time, spike_type in zip(single_trial.potential.spikes.t, single_trial.potential.spikes.v):
                    spikes_in_trial_by_type.setdefault(spike_type, []).append(spike_time)
                    
                for spike_type, spike_times in spikes_in_trial_by_type.items():
                    axp[0].eventplot(spike_times,                    
                                    lineoffsets=trial_id,   
                                    colors=SPIKE_COLOR_SCHEME[spike_type],
                                    alpha=0.9,
                                    linewidths=0.5,
                                    linelengths=0.8)   
                
                puff_on_t = single_trial.data.timeline.filter("VerticalPuffOn").t[0]
                puff_off_t = single_trial.data.timeline.filter("VerticalPuffOff").t[0]
                axp[0].add_patch(ptchs.Rectangle((puff_on_t, (trial_id - 0.5)), puff_off_t - puff_on_t, 1,
                                                facecolor=PUFF_COLOR, alpha=0.5, edgecolor='none', zorder=-10))
            BINSIZE = 20/1000
            group_spikes = grouping_events_rate([single_trial.potential.spikes for single_trial in puff_trials_500ms],
                                                bin_size=BINSIZE, use_event_value_as_weight=False)
            if len(group_spikes.t) > 0:
                bin_edges = np.concatenate([group_spikes.t, [group_spikes.t[-1] + BINSIZE]]) - BINSIZE/2
                axp[1].stairs(group_spikes.mean, bin_edges, fill=True, color='black')
            axp[1].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)

            axp[1].spines[['right', 'top', 'left']].set_visible(False)
            axp[0].spines[['right', 'top', 'left']].set_visible(False)
            axp[0].set_xlabel(f"Time [s]")
            axp[1].set_xlabel(f"Time [s]")
            axp[0].set_ylabel(f"Trial ID")
            axp[1].set_title(f"Firing Rate [Hz]")
            
            axp[0].set_yticks([])
            axp[0].yaxis.set_major_locator(MultipleLocator(5))
            axp[0].set_xticks([])

            axp[0].set_xlim(-0.5, 1.0)        
            axp[1].set_xlim(-0.5, 1.0)        
            axp[1].set_ylim(0, 150)
            axp[0].set_ylim(-0.5, len(puff_trials_500ms)-0.5)
            axp[0].set_title(f"{node_name}")

            if session_node.potential is None:
                logger.debug(f"Cannot plot convolved potential for {node_name}: no potential found. Skip...")
                continue

            # conv potential
            axp[2].set_yticks([])
            unit_plot_potential_conv(potential=[one_trial.data.potential for one_trial in puff_trials_500ms], 
                                    ax=axp[2], y_offset=0, ratio=FLUORESCENCE_RATIO, individual_trace_flag=True, spike_mark=False)
            axp[2].spines[['right', 'top',]].set_visible(False)
            axp[2].set_xlabel(f"Time [s]")
            axp[2].set_title(f"Conv. {DF_F0_SIGN}")
            axp[2].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)


            # LFP potential 
            axp[3].set_yticks([])
            unit_plot_potential(potential=[one_trial.data.potential for one_trial in puff_trials_500ms], 
                                ax=axp[3], y_offset=0, ratio=POTENTIAL_RATIO, aspect=4, wc_flag=WC_CONVERT_FLAG(session_node))
            axp[3].spines[['right', 'top',]].set_visible(False)
            axp[3].set_xlabel(f"Time [s]")   
            axp[3].set_xlim(-0.1, 0.1)    
            axp[3].set_title(f"Vm [4Hz, mV]")
            axp[3].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)    


            # spike distribution
            early_spikes = [one_trial.potential.spikes.filter("early_spike") for one_trial in puff_trials_500ms]
            sustained_spikes = [one_trial.potential.spikes.filter("sustained_spike") for one_trial in puff_trials_500ms]
            def unit_scatter_spike(spikes: list[Events], ax: plt.Axes, **kwargs):
                median_time = [np.median(spike.t) if len(spike) > 0 else np.nan for spike in spikes]
                spike_num = [len(spike) for spike in spikes]
                ax.scatter(median_time, spike_num, **kwargs)
            unit_scatter_spike(early_spikes, axp[4], facecolors=EARLY_SPIKE_COLOR, s=1, alpha=0.6, edgecolors='none')
            unit_scatter_spike(sustained_spikes, axp[5], facecolors=SUSTAINED_SPIKE_COLOR, s=1, alpha=0.6, edgecolors='none')
            for ax in axp[4:6]:
                ax.spines[['right', 'top',]].set_visible(False)
                ax.set_xlabel(f"Time [s]")
                ax.set_ylabel(f"Num. of Spikes")
                ax.axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)    
            axp[4].set_title(f"Early Spikes")
            axp[5].set_title(f"Sustained Spikes")
            axp[4].set_xlim(-0.05, 0.05)    
            axp[5].set_xlim(-0.1, 0.6)    

            


        # waveform
        for row_idx, session_node in enumerate(session_nodes):
            ax= axs[row_idx, -1]
            
            potential_timeseries = session_node.potential.aspect('raw')
            all_waveforms = {}
            spike_timeseries = potential_timeseries.batch_segment(
                session_node.potential.spikes.t, CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT)
            if len(spike_timeseries) == 0:
                continue
            grouped_spike_timeseries = grouping_timeseries(
                [TimeSeries(v=zscore(ts.v), t=ts.t) for ts in spike_timeseries], 
                interp_method="linear")

            oreo_plot(ax, grouped_spike_timeseries, 0, 1, {"color": SPIKE_COLOR_SCHEME['spike']}, FILL_BETWEEN_STYLE)
            ax.set_title("Waveform")
            ax.spines[['right', 'top',]].set_visible(False)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("z-scored amplitude")
        
        save_path = routing.default_fig_path(dataset, "42_{}" + f"_{batch_idx}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI)
        logger.info(f"Plot saved to {save_path}")
        
        # Close the figure
        plt.close(fig)
