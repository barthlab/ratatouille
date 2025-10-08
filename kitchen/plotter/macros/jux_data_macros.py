from collections import defaultdict
import os
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


def multiple_dataset_pca_plot(        
        datasets: list[DataSet],
        prefix_keyword: str,
        save_name: Optional[str] = None,

        BINSIZE = 25/1000,  # s
        FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
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
    from sklearn.decomposition import PCA

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
    
    n_components = 5
    pca = PCA(n_components=n_components)
    PC_projection = pca.fit_transform(feature_matrix)
    
    fig, axs = plt.subplots(6, n_components, figsize=(4*n_components, 15), constrained_layout=True,
                            height_ratios=[1, 1, 2, 2, 2, 2], sharey='row')
    for pc_id in range(n_components):
        axpc = axs[:, pc_id]
        pc = pca.components_[pc_id, :]
        project_values = PC_projection[:, pc_id]
        projected_feature = np.outer(project_values, pc)
        culmulative_reconstructed = np.dot(PC_projection[:, :pc_id+1], pca.components_[:pc_id+1, :]) + pca.mean_
        culmulative_error = np.abs(feature_matrix - culmulative_reconstructed)

        axpc[0].plot(bin_centers, pc, lw=1, color='black')
        axpc[0].set_title(f"PC{pc_id+1}")
        axpc[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
        axpc[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        axpc[0].spines[['right', 'top',]].set_visible(False)

        data = pd.DataFrame({
            f"PC{pc_id+1}": project_values,
            "Cohort": all_cohort_names,
        })
        sns.barplot(data=data, x="Cohort", y=f"PC{pc_id+1}", hue="Cohort", ax=axpc[1], 
                    palette=COHORT_COLORS, errorbar="se", alpha=0.7, 
                    order=sorted(set(all_cohort_names)))
        sns.stripplot(data=data, x="Cohort", y=f"PC{pc_id+1}", hue="Cohort", ax=axpc[1], 
                    palette=COHORT_COLORS, alpha=0.7, jitter=0.2, size=3,
                    order=sorted(set(all_cohort_names)))
        axpc[1].spines[['right', 'top',]].set_visible(False)
        axpc[1].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)

        sorted_idx = np.argsort(project_values)
        sns.heatmap(projected_feature[sorted_idx, :], ax=axpc[2], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(culmulative_reconstructed[sorted_idx, :], ax=axpc[3], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(feature_matrix[sorted_idx, :], ax=axpc[4], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(culmulative_error[sorted_idx, :], ax=axpc[5], 
                    cbar=False, cmap='Reds', alpha=0.5)
        
        for ax in axpc[2:]:
            ax.set_xlabel("Time [s]")

            plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
            ax.set_xticks(plot_tick_indices, [0, 0.5])
            
            # set y ticks to cohort names
            tick_positions = np.arange(len(all_cohort_names))
            ax.set_yticks(tick_positions + 0.5, [all_cohort_names[i] for i in sorted_idx], rotation=0, fontsize=2.)
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

    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])), 
                             prefix_keyword, f"PCA_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to" + save_path)




    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    axs = [fig.add_subplot(1, 4, 1),
           fig.add_subplot(1, 4, 2),
           fig.add_subplot(1, 4, 3, projection='3d'),
           fig.add_subplot(1, 4, 4)]
    
    for pc_id in range(n_components):
        axs[0].plot(bin_centers, pca.components_[pc_id, :], lw=1, label=f"PC{pc_id+1}")
    axs[0].set_title("Top 5 Principal Components")
    axs[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
    axs[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Component Weight")
    axs[0].legend()


    # --- Plot 2: Explained Variance Ratio (axs[1]) ---
    pc_indices = np.arange(1, n_components + 1)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    axs[1].bar(pc_indices, explained_variance, alpha=0.7, color='steelblue', label='Individual Variance')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(pc_indices, cumulative_variance, 'o-', color='darkred', label='Cumulative Variance')

    axs[1].set_xlabel("Principal Component")
    axs[1].set_ylabel("Explained Variance Ratio")
    ax1_twin.set_ylabel("Cumulative Variance Ratio")
    axs[1].set_xticks(pc_indices)
    axs[1].set_title("Explained Variance by Component")
    axs[1].spines[['top']].set_visible(False)
    ax1_twin.spines[['top']].set_visible(False)
    # Combine legends from both y-axes
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='best')


    # --- Plot 3: 3D Scatter Plot of Top 3 PCs (axs[2]) ---
    unique_cohorts = sorted(list(set(all_cohort_names)))
    for cohort in unique_cohorts:
        idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
        axs[2].scatter(
            PC_projection[idx, 0], PC_projection[idx, 1], PC_projection[idx, 2],
            color=COHORT_COLORS.get(cohort, 'gray'),
            label=cohort, s=15, alpha=0.8
        )
    axs[2].set_xlabel("PC1")
    axs[2].set_ylabel("PC2")
    axs[2].set_zlabel("PC3")
    axs[2].set_title("3D PCA Projection")
    axs[2].legend(title="Cohort")


    # --- Plot 4: UMAP Decomposition of Top 5 PCs (axs[3]) ---
    try:
        import umap
    except ImportError:
        axs[3].text(0.5, 0.5, 'umap-learn is not installed.\npip install umap-learn',
                    horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=6)
        embedding = reducer.fit_transform(PC_projection[:, :n_components])
        
        for cohort in unique_cohorts:
            idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
            axs[3].scatter(
                embedding[idx, 0], embedding[idx, 1],
                color=COHORT_COLORS.get(cohort, 'gray'),
                label=cohort, s=15, alpha=0.8
            )
        axs[3].set_xlabel("UMAP 1")
        axs[3].set_ylabel("UMAP 2")
        axs[3].set_title("UMAP of Top 5 PCs")
        axs[3].legend(title="Cohort")
        axs[3].spines[['right', 'top']].set_visible(False)
    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])), 
                            prefix_keyword, f"PC_Scatter_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to" + save_path)



def multiple_dataset_svd_plot(
    datasets: list[DataSet],
    prefix_keyword: str,
    save_name: Optional[str] = None,

    BINSIZE = 25/1000,  # s
    FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
    PREPROCESSING_METHOD = 'log-scale',  # 'raw', 'log-scale', 'z-score', 'fold-change'
):
    alignment_events = ("VerticalPuffOn",)

    # plotting
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from scipy.stats import zscore
    from sklearn.decomposition import TruncatedSVD # MODIFIED: Changed from PCA

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

    # --- Data Loading and Preprocessing (Unchanged) ---
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

    # --- SVD Analysis (Modified from PCA) ---
    n_components = 5
    svd = TruncatedSVD(n_components=n_components, random_state=42) # MODIFIED
    component_projection = svd.fit_transform(feature_matrix) # MODIFIED

    # --- Figure 1: Detailed Component Breakdown ---
    fig, axs = plt.subplots(6, n_components, figsize=(4*n_components, 15), constrained_layout=True,
                            height_ratios=[1, 1, 2, 2, 2, 2], sharey='row')
    for comp_id in range(n_components):
        ax_comp = axs[:, comp_id]
        component = svd.components_[comp_id, :] # MODIFIED
        project_values = component_projection[:, comp_id] # MODIFIED
        projected_feature = np.outer(project_values, component)

        # KEY CHANGE: TruncatedSVD reconstruction does not involve the mean.
        culmulative_reconstructed = np.dot(component_projection[:, :comp_id+1], svd.components_[:comp_id+1, :]) # MODIFIED
        culmulative_error = np.abs(feature_matrix - culmulative_reconstructed)

        ax_comp[0].plot(bin_centers, component, lw=1, color='black')
        ax_comp[0].set_title(f"Component {comp_id+1}") # MODIFIED
        ax_comp[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
        ax_comp[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        ax_comp[0].spines[['right', 'top',]].set_visible(False)

        data = pd.DataFrame({
            f"Component {comp_id+1}": project_values, # MODIFIED
            "Cohort": all_cohort_names,
        })
        sns.barplot(data=data, x="Cohort", y=f"Component {comp_id+1}", hue="Cohort", ax=ax_comp[1],
                    palette=COHORT_COLORS, errorbar="se", alpha=0.7,
                    order=sorted(set(all_cohort_names)))
        sns.stripplot(data=data, x="Cohort", y=f"Component {comp_id+1}", hue="Cohort", ax=ax_comp[1],
                      palette=COHORT_COLORS, alpha=0.7, jitter=0.2, size=3,
                      order=sorted(set(all_cohort_names)))
        ax_comp[1].spines[['right', 'top',]].set_visible(False)
        ax_comp[1].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)

        sorted_idx = np.argsort(project_values)
        sns.heatmap(projected_feature[sorted_idx, :], ax=ax_comp[2], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(culmulative_reconstructed[sorted_idx, :], ax=ax_comp[3], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(feature_matrix[sorted_idx, :], ax=ax_comp[4], 
                    cbar=False, **heatmap_kws)
        sns.heatmap(culmulative_error[sorted_idx, :], ax=ax_comp[5], 
                    cbar=False, cmap='Reds', alpha=0.5)

        for ax in ax_comp[2:]:
            ax.set_xlabel("Time [s]")
            plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
            ax.set_xticks(plot_tick_indices, [0, 0.5])

            tick_positions = np.arange(len(all_cohort_names))
            ax.set_yticks(tick_positions + 0.5, [all_cohort_names[i] for i in sorted_idx], rotation=0, fontsize=2.)
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

    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])),
                             prefix_keyword, f"SVD_{save_name}.png") # MODIFIED
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    # --- Figure 2: Summary Plots ---
    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    axs = [fig.add_subplot(1, 4, 1),
           fig.add_subplot(1, 4, 2),
           fig.add_subplot(1, 4, 3, projection='3d'),
           fig.add_subplot(1, 4, 4)]

    for comp_id in range(n_components):
        axs[0].plot(bin_centers, svd.components_[comp_id, :], lw=1, label=f"Component {comp_id+1}") # MODIFIED
    axs[0].set_title("Top 5 SVD Components") # MODIFIED
    axs[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
    axs[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel("Time [s]"); axs[0].set_ylabel("Component Weight"); axs[0].legend()

    # --- Plot 2: Explained Variance Ratio ---
    comp_indices = np.arange(1, n_components + 1)
    explained_variance = svd.explained_variance_ratio_ # MODIFIED
    cumulative_variance = np.cumsum(explained_variance)
    axs[1].bar(comp_indices, explained_variance, alpha=0.7, color='steelblue', label='Individual Variance')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(comp_indices, cumulative_variance, 'o-', color='darkred', label='Cumulative Variance')
    axs[1].set_xlabel("SVD Component") # MODIFIED
    axs[1].set_ylabel("Explained Variance Ratio"); ax1_twin.set_ylabel("Cumulative Variance Ratio")
    axs[1].set_xticks(comp_indices); axs[1].set_title("Explained Variance by Component")
    axs[1].spines[['top']].set_visible(False); ax1_twin.spines[['top']].set_visible(False)
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='best')

    # --- Plot 3: 3D Scatter Plot ---
    unique_cohorts = sorted(list(set(all_cohort_names)))
    for cohort in unique_cohorts:
        idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
        axs[2].scatter(
            component_projection[idx, 0], component_projection[idx, 1], component_projection[idx, 2], # MODIFIED
            color=COHORT_COLORS.get(cohort, 'gray'),
            label=cohort, s=15, alpha=0.8
        )
    axs[2].set_xlabel("Component 1"); axs[2].set_ylabel("Component 2"); axs[2].set_zlabel("Component 3") # MODIFIED
    axs[2].set_title("3D SVD Projection"); axs[2].legend(title="Cohort") # MODIFIED

    # --- Plot 4: UMAP Decomposition ---
    try:
        import umap
    except ImportError:
        axs[3].text(0.5, 0.5, 'umap-learn is not installed.\npip install umap-learn',
                    horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
        embedding = reducer.fit_transform(component_projection[:, :n_components]) # MODIFIED

        for cohort in unique_cohorts:
            idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
            axs[3].scatter(
                embedding[idx, 0], embedding[idx, 1],
                color=COHORT_COLORS.get(cohort, 'gray'),
                label=cohort, s=15, alpha=0.8
            )
        axs[3].set_xlabel("UMAP 1"); axs[3].set_ylabel("UMAP 2")
        axs[3].set_title("UMAP of Top 5 SVD Components") # MODIFIED
        axs[3].legend(title="Cohort"); axs[3].spines[['right', 'top']].set_visible(False)

    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])),
                             prefix_keyword, f"SVD_Scatter_{save_name}.png") # MODIFIED
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


def multiple_dataset_fa_plot(
    datasets: list[DataSet],
    prefix_keyword: str,
    save_name: Optional[str] = None,
    BINSIZE=25/1000,  # s
    FEATURE_RANGE=(-0.25, 0.75),  # (min, max) s
    PREPROCESSING_METHOD='log-scale',  # 'raw', 'log-scale', 'z-score', 'fold-change'
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
    # --- CHANGE 1: Import FactorAnalysis instead of PCA ---
    from sklearn.decomposition import FactorAnalysis
    from scipy.stats import zscore # Ensure zscore is imported if used in preprocessing

    # --- Plotting parameters are unchanged ---
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

    # --- Data loading and feature matrix creation are unchanged ---
    all_spikes_histogram, all_cohort_names, all_baseline_fr = [], [], []
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for dataset_idx, dataset in enumerate(datasets):
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
            all_baseline_fr.append(np.mean(spikes_histogram.mean[spikes_histogram.t < 0.]))

    raw_feature_matrix = np.stack(all_spikes_histogram, axis=0)

    # --- Data preprocessing is unchanged ---

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

    # --- CHANGE 2: Apply Factor Analysis model ---
    n_components = 5
    # Use FactorAnalysis with a varimax rotation for better interpretability
    fa = FactorAnalysis(n_components=n_components, rotation='varimax', random_state=42)
    # The `fit_transform` method returns the factor scores for each sample
    factor_scores = fa.fit_transform(feature_matrix)

    # --- First Figure: Detailed analysis per factor ---
    fig, axs = plt.subplots(6, n_components, figsize=(4*n_components, 15), constrained_layout=True,
                            height_ratios=[1, 1, 2, 2, 2, 2], sharey='row')
    
    for factor_id in range(n_components):
        ax_factor = axs[:, factor_id]
        # In FA, `components_` are the factor loadings (weights of each feature on the factor)
        factor_loadings = fa.components_[factor_id, :]
        project_values = factor_scores[:, factor_id]
        
        # Reconstruct the feature matrix based on the identified factors
        projected_feature = np.outer(project_values, factor_loadings)
        cumulative_reconstructed = np.dot(factor_scores[:, :factor_id+1], fa.components_[:factor_id+1, :]) + fa.mean_
        cumulative_error = np.abs(feature_matrix - cumulative_reconstructed)

        # Plot 1: Factor Loadings over time
        ax_factor[0].plot(bin_centers, factor_loadings, lw=1, color='black')
        ax_factor[0].set_title(f"Factor {factor_id+1}")
        ax_factor[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
        ax_factor[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        ax_factor[0].spines[['right', 'top',]].set_visible(False)

        # Plot 2: Factor Scores grouped by cohort
        data = pd.DataFrame({
            f"Factor {factor_id+1}": project_values,
            "Cohort": all_cohort_names,
        })
        sns.barplot(data=data, x="Cohort", y=f"Factor {factor_id+1}", hue="Cohort", ax=ax_factor[1],
                    palette=COHORT_COLORS, errorbar="se", alpha=0.7,
                    order=sorted(set(all_cohort_names)))
        sns.stripplot(data=data, x="Cohort", y=f"Factor {factor_id+1}", hue="Cohort", ax=ax_factor[1],
                      palette=COHORT_COLORS, alpha=0.7, jitter=0.2, size=3,
                      order=sorted(set(all_cohort_names)))
        ax_factor[1].spines[['right', 'top',]].set_visible(False)
        ax_factor[1].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
        
        # Heatmaps for reconstruction and error analysis
        sorted_idx = np.argsort(project_values)
        sns.heatmap(projected_feature[sorted_idx, :], ax=ax_factor[2], 
                    cbar=False, **heatmap_kws)
        ax_factor[2].set_title(f"Reconstruction from Factor {factor_id+1}")
        sns.heatmap(cumulative_reconstructed[sorted_idx, :], ax=ax_factor[3], 
                    cbar=False, **heatmap_kws)
        ax_factor[3].set_title(f"Cumulative Reconstruction")
        sns.heatmap(feature_matrix[sorted_idx, :], ax=ax_factor[4], 
                    cbar=False, **heatmap_kws)
        ax_factor[4].set_title(f"Original Data (Sorted)")
        sns.heatmap(cumulative_error[sorted_idx, :], ax=ax_factor[5], 
                    cbar=False, cmap='Reds', alpha=0.5)
        ax_factor[5].set_title(f"Cumulative Error")
        
        # Plot formatting for heatmaps (unchanged logic)
        for ax in ax_factor[2:]:
            ax.set_xlabel("Time [s]")
            plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
            ax.set_xticks(plot_tick_indices, [0, 0.5])
            tick_positions = np.arange(len(all_cohort_names))
            ax.set_yticks(tick_positions + 0.5, [all_cohort_names[i] for i in sorted_idx], rotation=0, fontsize=2.)
            ax.tick_params(axis='y', which='both', length=0, labelleft=False, labelright=True, pad=1)
            for tick_label in ax.get_yticklabels():
                text = tick_label.get_text()
                tick_label.set_color("white")
                tick_label.set_bbox({'facecolor': COHORT_COLORS[text], 'alpha': 1.0, 'edgecolor': 'none', 'boxstyle': 'square,pad=0.3'})

    # --- Save the first figure ---
    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])),
                             prefix_keyword, f"FA_{save_name}.png") # Changed from PCA to FA
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    # --- Second Figure: Summary plots ---
    fig = plt.figure(figsize=(12, 3), constrained_layout=True)
    axs = [fig.add_subplot(1, 4, 1),
           fig.add_subplot(1, 4, 2),
           fig.add_subplot(1, 4, 3, projection='3d'),
           fig.add_subplot(1, 4, 4)]
    
    # --- Plot 1: Top Factor Loadings ---
    for factor_id in range(n_components):
        axs[0].plot(bin_centers, fa.components_[factor_id, :], lw=1, label=f"Factor {factor_id+1}")
    axs[0].set_title("Top 5 Factor Loadings")
    axs[0].axvspan(0, 0.5, alpha=0.5, color=PUFF_COLOR, lw=0, zorder=-10)
    axs[0].axhline(0, color='gray', linestyle='--', lw=0.5, alpha=0.5, zorder=-10)
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Factor Loading") # More accurate term than "Component Weight"
    axs[0].legend()

    # --- CHANGE 3: Plot 2: Explained Variance Calculation for FA ---
    factor_indices = np.arange(1, n_components + 1)
    # In FA, variance explained by a factor is the sum of its squared loadings.
    factor_variance = np.sum(fa.components_**2, axis=1)
    # For context, we can show this as a proportion of the total variance in the original data.
    total_variance = np.sum(feature_matrix.var(axis=0))
    cumulative_variance_prop = np.cumsum(factor_variance) / total_variance

    axs[1].bar(factor_indices, factor_variance, alpha=0.7, color='steelblue', label='Factor Variance')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(factor_indices, cumulative_variance_prop, 'o-', color='darkred', label='Cumulative Variance Prop.')

    axs[1].set_xlabel("Factor")
    axs[1].set_ylabel("Variance Explained (Sum of Squared Loadings)")
    ax1_twin.set_ylabel("Cumulative Proportion of Total Variance")
    axs[1].set_xticks(factor_indices)
    axs[1].set_title("Variance Explained by Factor")
    axs[1].spines[['top']].set_visible(False)
    ax1_twin.spines[['top']].set_visible(False)
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='best')

    # --- Plot 3: 3D Scatter Plot of Top 3 Factor Scores ---
    unique_cohorts = sorted(list(set(all_cohort_names)))
    for cohort in unique_cohorts:
        idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
        axs[2].scatter(
            factor_scores[idx, 0], factor_scores[idx, 1], factor_scores[idx, 2],
            color=COHORT_COLORS.get(cohort, 'gray'),
            label=cohort, s=15, alpha=0.8
        )
    axs[2].set_xlabel("Factor 1 Score")
    axs[2].set_ylabel("Factor 2 Score")
    axs[2].set_zlabel("Factor 3 Score")
    axs[2].set_title("3D Factor Score Projection")
    axs[2].legend(title="Cohort")

    # --- Plot 4: UMAP Decomposition of Top 5 Factor Scores ---
    try:
        import umap
    except ImportError:
        axs[3].text(0.5, 0.5, 'umap-learn is not installed.\npip install umap-learn',
                      horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=6) # Safer n_neighbors
        embedding = reducer.fit_transform(factor_scores[:, :n_components])
        
        for cohort in unique_cohorts:
            idx = [i for i, name in enumerate(all_cohort_names) if name == cohort]
            axs[3].scatter(
                embedding[idx, 0], embedding[idx, 1],
                color=COHORT_COLORS.get(cohort, 'gray'),
                label=cohort, s=15, alpha=0.8
            )
        axs[3].set_xlabel("UMAP 1")
        axs[3].set_ylabel("UMAP 2")
        axs[3].set_title("UMAP of Top 5 Factors")
        axs[3].legend(title="Cohort")
        axs[3].spines[['right', 'top']].set_visible(False)

    # --- Save the second figure ---
    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])),
                             prefix_keyword, f"FA_Scatter_{save_name}.png") # Changed from PC to FA
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
    "SST_WC": "#CF8200",
    "PV_JUX": "#ff0000",
    "PYR_JUX": "#0000FF",
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
