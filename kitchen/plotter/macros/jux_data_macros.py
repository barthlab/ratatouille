from collections import defaultdict
import os
from kitchen.operator import select_trial_rules
from kitchen.operator.grouping import grouping_events_histogram, grouping_events_rate, grouping_timeseries
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_PASSIVEPUFF_RULES, PREDEFINED_TRIAL_RULES
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.color_scheme import PUFF_COLOR, SPIKE_COLOR_SCHEME
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE
from kitchen.plotter.utils.fill_plot import oreo_plot, sushi_plot
from kitchen.settings.potential import CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, SPIKE_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.structure.hierarchical_data_structure import Node, DataSet
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils.sequence_kit import select_from_value
from kitchen.utils.numpy_kit import zscore
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
    COHORT_COLORS = {
        "SST_JUX": "#FF5F00",
        "SST_WC": "#FE0109",
        "PV_JUX": "#00FF5F",
        "PYR_JUX": "#5F00FF",
    }
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
    elif PREPROCESSING_METHOD == 'z-score':
        feature_matrix = zscore(raw_feature_matrix, axis=1)
    elif PREPROCESSING_METHOD == 'fold-change':
        feature_matrix = (raw_feature_matrix - all_baseline_fr[:, None]) / (all_baseline_fr[:, None] + 1)
    elif PREPROCESSING_METHOD == 'raw':
        feature_matrix = raw_feature_matrix
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

    # sorted_feature_matrix = feature_matrix[dendro_info['leaves'], :]
    sns.heatmap(sorted_feature_matrix.repeat(10, axis=0), 
                ax=axs[1], cmap='viridis', cbar_ax=axs[2], 
                cbar_kws={'label': f"Firing Rate ({PREPROCESSING_METHOD})"},
                # norm=matplotlib.colors.LogNorm(vmin=1, vmax=100)
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

    fig.suptitle(f"{prefix_keyword}\n {save_name}")
    fig.set_size_inches(3, 2)
    save_path = os.path.join(os.path.dirname(routing.default_fig_path(datasets[0])), 
                             prefix_keyword, f"Dendrogram_{save_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=900, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to" + save_path)



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
    COHORT_COLORS = {
        "SST_JUX": "#FF5F00",
        "SST_WC": "#FE0109",
        "PV_JUX": "#00FF5F",
        "PYR_JUX": "#5F00FF",
    }
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