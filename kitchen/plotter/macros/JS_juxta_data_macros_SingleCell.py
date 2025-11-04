
from kitchen.configs import routing
from kitchen.operator.grouping import grouping_events_rate
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import style_dicts
from kitchen.plotter.ax_plotter.basic_plot import beam_view
import kitchen.plotter.color_scheme as color_scheme
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.unit_plotter import unit_trace
from kitchen.settings.potential import WC_CONVERT_FLAG
from kitchen.structure.hierarchical_data_structure import Node, DataSet
from kitchen.utils.sequence_kit import select_from_value, select_truthy_items
from kitchen.plotter.decorators.default_decorators import default_style
from kitchen.configs.naming import get_node_name
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL
from kitchen.plotter.plotting_params import PARALLEL_Y_INCHES, POTENTIAL_RATIO, UNIT_X_INCHES

from functools import partial
import logging

logger = logging.getLogger(__name__)

# plotting 
plot_manual_spike4Hz = PlotManual(potential=4,)
plot_manual_spike300Hz = PlotManual(potential=300.)


def SingleCell_4HZ_300HZ_BeamView(
        node: Node,
        dataset: DataSet,
) -> None:
    node_name = get_node_name(node)

    # filter out trial types that cannot be plotted
    all_trial_nodes = dataset.subtree(node).select("trial", _empty_warning=False)
    trial_types = select_from_value(
                all_trial_nodes.rule_based_group_by(lambda x: x.info.get("trial_type")),
                _self = partial(CHECK_PLOT_MANUAL, plot_manual=plot_manual_spike4Hz)
            )
    n_trial_types = len(trial_types)
    if n_trial_types == 0:
        logger.debug(f"Cannot plot trial parallel for {node_name}: no trial type found")
        return
    
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
        save_path=routing.default_fig_path(node, "BeamView_{}.png"),
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

def get_500ms_puff_trials(node: Node, dataset: DataSet):
    all_trial_nodes = dataset.subtree(node).select("trial", _empty_warning=False)
    trial_types = select_from_value(
                all_trial_nodes.rule_based_group_by(lambda x: x.info.get("trial_type")),
                _self = partial(CHECK_PLOT_MANUAL, plot_manual=plot_manual_spike4Hz)
            )
    if "500msPuff" not in trial_types:
        logger.debug(f"Cannot plot raster plot for {get_node_name(node)}: no 500ms puff trials found. Skip...")
        return None
    trials_500ms_puff = trial_types["500msPuff"]
    if len(trials_500ms_puff) == 0:
        logger.debug(f"Cannot plot raster plot for {get_node_name(node)}: no puff trials found. Skip...")
        return None
    return sync_nodes(trials_500ms_puff, ("VerticalPuffOn",), plot_manual_spike300Hz)


def SingleCell_4HZ_ZOOMINOUT_StackView(        
        node: Node,
        dataset: DataSet,
) -> None:
    node_name = get_node_name(node)
    
    puff_trials_500ms = get_500ms_puff_trials(node, dataset)
    if puff_trials_500ms is None:
        return

    # plotting
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(2, 1, height_ratios=[1, 1], constrained_layout=True)

    axs[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axvspan(0, 0.5, alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
    for ax in axs:
        max_height = unit_trace.unit_plot_potential(potential=select_truthy_items([node.data.potential for node in puff_trials_500ms]), 
                            ax=ax, y_offset=0, ratio=POTENTIAL_RATIO, aspect=4, wc_flag=WC_CONVERT_FLAG(node), spike_mark=2,
                            emphasize_rule="best" if not WC_CONVERT_FLAG(node) else "median",)
        # ax.set_ylim(-0.8*max_height, 0.8*max_height)
    
    axs[1].set_xlabel("Time [s]")
    axs[0].set_xlim(-0.7, 1.2)
    axs[1].set_xlim(-0.05, 0.15)

    # axs[0].set_title(f"{node_name}")
    fig.set_size_inches(4, 2.5)
    
    save_path = routing.default_fig_path(node, "StackViewExample_{}.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    
    logger.info("Plot saved to" + save_path)


def SingleCell_4HZ_BeamView_Beautify(        
        node: Node,
        dataset: DataSet,
) -> None:
    node_name = get_node_name(node)
    
    puff_trials_500ms = get_500ms_puff_trials(node, dataset)
    if puff_trials_500ms is None:
        return

    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    x_offset, y_offset = 0.5 / len(puff_trials_500ms), 1
    ax.spines[['right', 'top',]].set_visible(False)
    ax.set_yticks([])
    style_dicts.EMPHASIZED_POTENTIAL_LINE_ADD_STYLE["lw"] = 0.5
    style_dicts.EMPHASIZED_POTENTIAL_LINE_ADD_STYLE["alpha"] = 0.7
    for spike_type in style_dicts.SPIKE_POTENTIAL_TRACE_STYLE.keys():
        style_dicts.SPIKE_POTENTIAL_TRACE_STYLE[spike_type]["lw"] = 1
        style_dicts.SPIKE_POTENTIAL_TRACE_STYLE[spike_type]["alpha"] = 0.9
    for trial_id, trial_node in enumerate(puff_trials_500ms):
        offseted_potntial = trial_node.data.potential.aligned_to(-trial_id * x_offset)
        unit_trace.unit_plot_potential(potential=offseted_potntial, 
                    ax=ax, y_offset=trial_id * y_offset, ratio=POTENTIAL_RATIO, aspect=4, wc_flag=WC_CONVERT_FLAG(node), spike_mark=1,
                    yticks_flag=trial_id == 0,)
        # ax.set_ylim(-0.8*max_height, 0.8*max_height)
    
    parallelogram = patches.Polygon(xy=list(zip([0, 0.5, 0.5 + x_offset * (len(puff_trials_500ms) - 1), x_offset * (len(puff_trials_500ms) - 1)], 
                                                [0, 0, y_offset * (len(puff_trials_500ms) - 1), y_offset * (len(puff_trials_500ms) - 1)])), 
                                    alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)

    ax.add_patch(parallelogram)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(-0.7, 1.6)

    # axs[0].set_title(f"{node_name}")
    fig.set_size_inches(6, 3.5)
    
    save_path = routing.default_fig_path(node, "BeautifyBeamView_{}.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    
    logger.info("Plot saved to" + save_path)


def SingleCell_RasterPlot(        
        node: Node,
        dataset: DataSet,

        BINSIZE = 20/1000,  # s
):
    node_name = get_node_name(node)

    puff_trials_500ms = get_500ms_puff_trials(node, dataset)
    if puff_trials_500ms is None:
        return
    
    # plotting
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptchs
    import numpy as np
    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(2, 1, sharex='col', sharey='row', height_ratios=[3, 1], constrained_layout=True)    
    for trial_id, single_trial in enumerate(puff_trials_500ms):
        spikes_in_trial_by_type = {}
        for spike_time, spike_type in zip(single_trial.potential.spikes.t, single_trial.potential.spikes.v):
            spikes_in_trial_by_type.setdefault(spike_type, []).append(spike_time)
            
        for spike_type, spike_times in spikes_in_trial_by_type.items():
            axs[0].eventplot(spike_times,                    
                            lineoffsets=trial_id,   
                            colors=color_scheme.SPIKE_COLOR_SCHEME[spike_type],
                            alpha=0.9,
                            # linewidths=1,
                            linelengths=0.8)   
        
        puff_on_t = single_trial.data.timeline.filter("VerticalPuffOn").t[0]
        puff_off_t = single_trial.data.timeline.filter("VerticalPuffOff").t[0]
        axs[0].add_patch(ptchs.Rectangle((puff_on_t, (trial_id - 0.5)), puff_off_t - puff_on_t, 1,
                                         facecolor=color_scheme.PUFF_COLOR, alpha=0.3, edgecolor='none', zorder=-10))
        if trial_id >= 20:
            break
    
    group_spikes = grouping_events_rate([single_trial.potential.spikes for single_trial in puff_trials_500ms],
                                        bin_size=BINSIZE, use_event_value_as_weight=False)

    # Only plot if we have spike data
    if len(group_spikes.t) > 0:
        bin_edges = np.concatenate([group_spikes.t, [group_spikes.t[-1] + BINSIZE]]) - BINSIZE/2
        axs[1].stairs(group_spikes.mean, bin_edges, fill=True, color='black')
    # upper_bound = mean_rate + group_spikes.variance
    # lower_bound = mean_rate - group_spikes.variance
    # axs[1].fill_between(bin_edges[:-1], lower_bound, upper_bound, alpha=0.4, lw=0, color='gray', step='post')
    axs[1].axvspan(0, 0.5, alpha=0.3, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)

    axs[1].spines[['right', 'top', 'left']].set_visible(False)
    axs[0].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    axs[1].set_xlabel("Time [s]")
    axs[0].set_xlim(-0.5, 1.0)
    
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    
    axs[1].set_ylim(0, 101)
    axs[0].set_ylim(-0.5, min(20, len(puff_trials_500ms))-0.5)
    # axs[0].set_title(f"{node_name}")
    fig.set_size_inches(2, 2.5)
    
    save_path = routing.default_fig_path(node, "RasterPlot_{}.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    
    logger.info("Plot saved to" + save_path)
