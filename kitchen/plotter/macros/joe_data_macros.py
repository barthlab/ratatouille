from collections import defaultdict
import os
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator import select_trial_rules
from kitchen.operator.grouping import grouping_events_histogram, grouping_events_rate, grouping_timeseries
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_NAIVE_RULES, PREDEFINED_PASSIVEPUFF_RULES, PREDEFINED_TRIAL_RULES
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.color_scheme import EARLY_SPIKE_COLOR, PUFF_COLOR, SPIKE_COLOR_SCHEME, STIM_COLOR, SUSTAINED_SPIKE_COLOR
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
    trial_types = select_from_value(subtree.rule_based_selection(PREDEFINED_NAIVE_RULES),
                                    _self = lambda x: CHECK_PLOT_MANUAL(x, plot_manual_spike300Hz))

    n_trial_types = len(trial_types)
    if n_trial_types == 0:
        logger.debug(f"Cannot plot trial parallel for {node_name}: no trial type found")
        return
    
    prefix_str = f"{prefix_keyword}_ParallelSummary" if prefix_keyword is not None else "ParallelSummary"
    alignment_events = ("TrialOn",)
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
            f"{node_name}\nHPF 4Hz\n{trial_type}": {"set_xlim": (-2, 3)}
            for trial_type in trial_types.keys()
        } | {
            f"{node_name}\nHPF 300Hz\n{trial_type}": {"set_xlim": (-2, 3)}
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
    alignment_events = ("TrialOn",)


    total_mosaic, content_dict, plot_settings = [], {}, {}
    for row_node in enumerate_nodes:
        subtree = dataset.subtree(row_node)                
        all_possible_trial_types = subtree.rule_based_selection(PREDEFINED_NAIVE_RULES)

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
    all_trials = subtree.select("trial")

    prefix_str = f"{prefix_keyword}_RasterPlot" if prefix_keyword is not None else "RasterPlot"
    alignment_events = ("TrialOn",)

    if len(all_trials) == 0:
        logger.debug(f"Cannot plot raster plot for {node_name}: no trials found. Skip...")
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

    all_trials = sync_nodes(all_trials, alignment_events, plot_manual_spike300Hz)
    if len(all_trials) == 0:
        logger.debug(f"Cannot plot raster plot for {node_name}: no trials found after syncing. Skip...")
        return
    
    for trial_id, single_trial in enumerate(all_trials):
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
        
        stim_on_t = single_trial.data.timeline.filter("StimOn").t
        stim_off_t = single_trial.data.timeline.filter("StimOff").t
        for single_stim_on_t, single_stim_off_t in zip(stim_on_t, stim_off_t):
            axs[0].add_patch(ptchs.Rectangle((single_stim_on_t, (trial_id - 0.5)), single_stim_off_t - single_stim_on_t, 1,
                                            facecolor=STIM_COLOR, alpha=0.5, edgecolor='none', zorder=-10))
    
    group_spikes = grouping_events_rate([single_trial.potential.spikes for single_trial in all_trials],
                                        bin_size=BINSIZE, use_event_value_as_weight=False)

    # Only plot if we have spike data
    if len(group_spikes.t) > 0:
        bin_edges = np.concatenate([group_spikes.t, [group_spikes.t[-1] + BINSIZE]]) - BINSIZE/2
        axs[1].stairs(group_spikes.mean, bin_edges, fill=True, color='black')
    # upper_bound = mean_rate + group_spikes.variance
    # lower_bound = mean_rate - group_spikes.variance
    # axs[1].fill_between(bin_edges[:-1], lower_bound, upper_bound, alpha=0.4, lw=0, color='gray', step='post')
    opto_start = 2.0  # s
    pulse_duration = 0.005  # 5 ms
    pulse_interval = 0.05   # 50 ms
    num_pulses = 5
    for i in range(num_pulses):
        start = opto_start + i * pulse_interval
        end = start + pulse_duration
        axs[1].axvspan(start, end, alpha=0.5, color=STIM_COLOR, lw=0, zorder=-10)

    axs[1].spines[['right', 'top', 'left']].set_visible(False)
    axs[0].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    axs[1].set_xlabel(f"Time [s]")
    axs[0].set_xlim(-2, 3)
    
    axs[0].set_yticks([])
    axs[0].yaxis.set_major_locator(MultipleLocator(5))
    axs[0].set_xticks([])
    
    axs[1].set_ylim(0, 150)
    axs[0].set_ylim(-0.5, len(all_trials)-0.5)
    axs[0].set_title(f"{node_name}")
    fig.set_size_inches(1.5, 1.5)
    
    fig.savefig(routing.default_fig_path(node, prefix_str + "_{}.png"), dpi=900, transparent=True)
    plt.close(fig)
    
    logger.info("Plot saved to" + routing.default_fig_path(node, prefix_str + "_{}.png"))


