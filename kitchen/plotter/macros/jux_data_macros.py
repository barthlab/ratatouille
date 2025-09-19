from kitchen.operator import select_trial_rules
from kitchen.operator.grouping import grouping_events_histogram, grouping_events_rate
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_PASSIVEPUFF_RULES, PREDEFINED_TRIAL_RULES
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.color_scheme import PUFF_COLOR, SPIKE_COLOR_SCHEME
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Node, DataSet
from kitchen.utils.sequence_kit import select_from_value
from kitchen.settings.timeline import ALIGNMENT_STYLE
from kitchen.plotter.decorators.default_decorators import default_style
from kitchen.plotter.ax_plotter.basic_plot import beam_view, stack_view
from kitchen.configs.naming import get_node_name
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL
from kitchen.plotter.plotting_params import PARALLEL_Y_INCHES, UNIT_X_INCHES, UNIT_Y_INCHES, ZOOMED_Y_INCHES
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
    bin_edges = np.concatenate([group_spikes.t, [group_spikes.t[-1] + BINSIZE]]) - BINSIZE/2

    axs[1].stairs(group_spikes.mean, bin_edges, fill=True, color='black')    
    # upper_bound = group_spikes.mean + group_spikes.variance
    # lower_bound = group_spikes.mean - group_spikes.variance
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


def dendrogram_plot(        
        dataset: DataSet,
        prefix_keyword: Optional[str] = None,

        BINSIZE = 25/1000,  # s
        FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
        LINKAGE_METHOD = 'single',  # 'single', 'complete', 'average', 'ward', 'centroid', 'median', 'weighted'
        LINKAGE_METRIC = 'cosine',  # 'cosine', 'euclidean', 'correlation', 
):
    prefix_str = f"{prefix_keyword}_Dendrogram" if prefix_keyword is not None else "Dendrogram"
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
        'figure.titlesize': 5      # Figure title (suptitle)
    })
    
    all_spikes_histogram = []
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for cell_node in dataset.select("cellsession"):        
        subtree = dataset.subtree(cell_node)
        
        puff_trials_500ms = subtree.select("trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,)
        puff_trials_500ms = sync_nodes(puff_trials_500ms, alignment_events, plot_manual_spike300Hz)
        if len(puff_trials_500ms) == 0:
            continue
        spikes_histogram = grouping_events_histogram([single_trial.potential.spikes.segment(*FEATURE_RANGE) 
                                                      for single_trial in puff_trials_500ms], 
                                                      bins=bins, use_event_value_as_weight=False)
        all_spikes_histogram.append(spikes_histogram.mean)
        
    feature_matrix = np.log(np.stack(all_spikes_histogram, axis=0) + 1)
    linked = linkage(feature_matrix, method=LINKAGE_METHOD, metric=LINKAGE_METRIC)

    fig, axs = plt.subplots(1, 3, width_ratios=[0.5, 1, 0.2], constrained_layout=True)
    
    set_link_color_palette(['red', 'blue', '#b2df8a', '#33a02c'])
    dendro_info = dendrogram(linked, ax=axs[0], orientation='left')
    axs[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    axs[0].tick_params(axis='y', which='both', length=0, labelright=False, )
    axs[0].set_xticks([])
    axs[0].set_ylabel('Hierarchical Dendrogram')
    axs[0].set_xlabel(f'Distance ({LINKAGE_METHOD})')

    sorted_feature_matrix = np.exp(feature_matrix[dendro_info['leaves'], :])    
    
    # sorted_feature_matrix = feature_matrix[dendro_info['leaves'], :]
    sns.heatmap(sorted_feature_matrix.repeat(10, axis=0), 
                ax=axs[1], cmap='viridis', cbar_ax=axs[2], 
                cbar_kws={'label': "Firing Rate (Hz)"},
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=100)
                # norm=matplotlib.colors.Normalize()
                )
    axs[1].set_xlabel("Time [s]")
    axs[1].sharey(axs[0])

    plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
    axs[1].set_xticks(plot_tick_indices, [0, 0.5])
    
    axs[1].tick_params(axis='y', which='both', length=0, labelleft=False, )

    fig.set_size_inches(3, 2)
    fig.savefig(routing.default_fig_path(dataset, prefix_str + "_{}.png"), dpi=900, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to" + routing.default_fig_path(dataset, prefix_str + "_{}.png"))




def multiple_dataset_dendrogram_plot(        
        datasets: list[DataSet],
        prefix_keyword: Optional[str] = None,

        BINSIZE = 25/1000,  # s
        FEATURE_RANGE = (-0.25, 0.75),  # (min, max) s
        LINKAGE_METHOD = 'single',  # 'single', 'complete', 'average', 'ward', 'centroid', 'median', 'weighted'
        LINKAGE_METRIC = 'cosine',  # 'cosine', 'euclidean', 'correlation', 
):
    prefix_str = f"{prefix_keyword}_Dendrogram" if prefix_keyword is not None else "Dendrogram"
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
    COHORT_COLORS = {
        "SST_JUX": "#FF5F00",
        "SST_WC": "#FE0109",
        "PV_JUX": "#00FF5F",
        "PYR_JUX": "#5F00FF",
    }
    all_spikes_histogram, all_cohort_names = [], []
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for dataset_idx, dataset in enumerate(datasets):
        for cell_node in dataset.select("cellsession"):        
            subtree = dataset.subtree(cell_node)
            
            puff_trials_500ms = subtree.select("trial", timeline = lambda x: 0.48 < select_trial_rules._puff_duration(x) < 0.52,)
            puff_trials_500ms = sync_nodes(puff_trials_500ms, alignment_events, plot_manual_spike300Hz)
            if len(puff_trials_500ms) == 0:
                continue
            spikes_histogram = grouping_events_histogram([single_trial.potential.spikes.segment(*FEATURE_RANGE) 
                                                        for single_trial in puff_trials_500ms], 
                                                        bins=bins, use_event_value_as_weight=False)
            all_spikes_histogram.append(spikes_histogram.mean)
            all_cohort_names.append(dataset.name)
        
    feature_matrix = np.log(np.stack(all_spikes_histogram, axis=0) + 1)
    linked = linkage(feature_matrix, method=LINKAGE_METHOD, metric=LINKAGE_METRIC)

    fig, axs = plt.subplots(1, 3, width_ratios=[0.5, 1, 0.2], constrained_layout=True)
    
    set_link_color_palette(['red', 'blue', '#b2df8a', '#33a02c'])
    dendro_info = dendrogram(linked, ax=axs[0], orientation='left')
    axs[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    axs[0].tick_params(axis='y', which='both', length=0, labelright=False, )
    axs[0].set_xticks([])
    axs[0].set_ylabel('Hierarchical Dendrogram')
    axs[0].set_xlabel(f'Distance ({LINKAGE_METHOD})')

    sorted_feature_matrix = np.exp(feature_matrix[dendro_info['leaves'], :])    
    sorted_cohort_names = [all_cohort_names[i] for i in dendro_info['leaves']]


    # sorted_feature_matrix = feature_matrix[dendro_info['leaves'], :]
    sns.heatmap(sorted_feature_matrix.repeat(10, axis=0), 
                ax=axs[1], cmap='viridis', cbar_ax=axs[2], 
                cbar_kws={'label': "Firing Rate (Hz)"},
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=100)
                # norm=matplotlib.colors.Normalize()
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

    fig.set_size_inches(3, 2)
    fig.savefig(routing.default_fig_path(datasets[0], prefix_str + "_{}.png"), dpi=900, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to" + routing.default_fig_path(datasets[0], prefix_str + "_{}.png"))
