"""
Basic plotting macros for neuroscience data visualization.

High-level plotting functions that generate standardized visualizations
for different levels of the hierarchical data structure.

Functions:
    session_overview: Generate flat view plots for individual sessions
    fov_overview: Generate flat view plots for all sessions within a FOV
    node_trial_avg_default: Generate trial-averaged plots for any node type
    fov_trial_avg_default: Generate trial-averaged plots organized by FOV days
"""

from functools import partial
import warnings

from kitchen.configs import routing
from kitchen.configs.naming import get_node_name
from kitchen.plotter.ax_plotter.advance_plot import subtract_view
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL, PlotManual
from kitchen.plotter.plotting_params import UNIT_X_INCHES, UNIT_Y_INCHES
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL
from kitchen.settings.timeline import ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import Fov, Node, Session, DataSet
from kitchen.plotter.decorators.default_decorators import default_style
from kitchen.plotter.ax_plotter.basic_plot import flat_view, stack_view
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_TRIAL_RULES
from kitchen.utils.sequence_kit import select_from_value


def session_overview(
        session_node: Session,
        plot_manual: PlotManual,
) -> None:
    """
    Generate a flat view overview plot for a single session.

    Creates a single-panel figure showing data modalities in a flat temporal view.

    Args:
        session_node (Session): Session node containing data to visualize.
        plot_manual (PlotManual): Configuration specifying which data modalities
            to include (timeline, fluorescence, lick, locomotion, etc.).

    Example:
        >>> from kitchen.plotter.plotting_manual import PlotManual
        >>> plot_config = PlotManual(timeline=True, fluorescence=True, lick=True)
        >>> session_overview(my_session_node, plot_config)
    """
    session_name = get_node_name(session_node)
    session_dataset = DataSet(name=session_name, nodes=[session_node])
    try:
        default_style(
            mosaic_style=[[session_name,],],
            content_dict={
                session_name: (
                    partial(flat_view, plot_manual=plot_manual),
                    session_dataset)
                },
            figsize=(5, UNIT_Y_INCHES),
            save_path=routing.default_fig_path(session_dataset, "SessionOverview_{}.png"),
        )
    except Exception as e:
        warnings.warn(f"Cannot plot session overview for {get_node_name(session_node)}: {e}")

def fov_overview(
        fov_node: Fov,
        dataset: DataSet,
        plot_manual: PlotManual,
) -> None:
    """
    Generate flat view overview plots for all sessions within a FOV.

    Creates a multi-panel figure with one row per session.

    Args:
        fov_node (Fov): FOV node defining the spatial scope for session selection.
        dataset (DataSet): Complete dataset used to find sessions within the FOV subtree.
        plot_manual (PlotManual): Configuration specifying which data modalities to include.

    Example:
        >>> from kitchen.plotter.plotting_manual import PlotManual
        >>> plot_config = PlotManual(timeline=True, fluorescence=True)
        >>> fov_overview(my_fov_node, complete_dataset, plot_config)
    """
    session_nodes = dataset.subtree(fov_node).select("session")
    n_session = len(session_nodes)
    try:
        default_style(
            mosaic_style=[[get_node_name(session_node),] for session_node in session_nodes],
            content_dict={
                get_node_name(session_node): (
                    partial(flat_view, plot_manual=plot_manual),
                    DataSet(name=session_node.session_id, nodes=[session_node]))
                for session_node in session_nodes
                },
            figsize=(5, UNIT_Y_INCHES * n_session),
            save_path=routing.default_fig_path(session_nodes, "FovOverview_{}.png"),
        )
    except Exception as e:
        warnings.warn(f"Cannot plot fov overview for {get_node_name(fov_node)}: {e}")


def single_node_trial_avg_default(
        node: Node,
        dataset: DataSet,
        plot_manual: PlotManual,
        trial_rules: dict = PREDEFINED_TRIAL_RULES,
) -> None:
    """
    Generate trial-averaged plots for all trial types within a node's subtree.

    Creates multi-panel figures showing trial-averaged data for each predefined
    trial type. Generates separate figures for each alignment style.

    Args:
        node (Node): Hierarchical node defining the scope for trial selection.
        dataset (DataSet): Complete dataset used to extract the subtree and find trials.
        plot_manual (PlotManual): Configuration specifying which data modalities to include.

    Example:
        >>> from kitchen.plotter.plotting_manual import PlotManual
        >>> plot_config = PlotManual(timeline=True, fluorescence=True, lick=True)
        >>> node_trial_avg_default(my_session_node, complete_dataset, plot_config)
    """
    subtree = dataset.subtree(node)

    # filter out trial types that cannot be plotted
    trial_types = select_from_value(subtree.rule_based_selection(trial_rules),
                                    _self = lambda x: CHECK_PLOT_MANUAL(x, plot_manual))

    n_trial_types = len(trial_types)
    for alignment_name, alignment_events in ALIGNMENT_STYLE.items():
        try:
            assert n_trial_types > 0, f"Cannot find any trial type for {get_node_name(node)}"
            default_style(
                mosaic_style=[[f"{get_node_name(node)}\n{trial_type}"
                            for trial_type in trial_types.keys()],],
                content_dict={
                    f"{get_node_name(node)}\n{trial_type}": (
                        partial(stack_view, plot_manual=plot_manual, sync_events=alignment_events),
                        trial_dataset)
                    for trial_type, trial_dataset in trial_types.items()
                    },
                figsize=(n_trial_types * UNIT_X_INCHES, UNIT_Y_INCHES),
                save_path=routing.default_fig_path(subtree, f"TrialAvg_{{}}_{alignment_name}.png"),
            )
        except Exception as e:
            warnings.warn(f"Cannot plot trial average for {get_node_name(node)} with {alignment_name}: {e}")


def fov_summary_trial_avg_default(
        fov_node: Fov,
        dataset: DataSet,
        plot_manual: PlotManual,
        trial_rules: dict = PREDEFINED_FOVTRIAL_RULES,
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
    fovday_nodes = dataset.subtree(fov_node).select("fovday")
    for alignment_name, alignment_events in ALIGNMENT_STYLE.items():
        try:
            total_mosaic, content_dict = [], {}
            for fovday_node in fovday_nodes:
                subtree = dataset.subtree(fovday_node)                
                all_possible_trial_types = subtree.rule_based_selection(trial_rules)

                # filter out trial types that cannot be plotted
                trial_types = select_from_value(all_possible_trial_types,
                                                _self = lambda dataset: CHECK_PLOT_MANUAL(dataset, plot_manual))

                # update total_mosaic and content_dict
                total_mosaic.append([f"{get_node_name(fovday_node)}\n{trial_type}"
                                    for trial_type in trial_types.keys()])
                content_dict.update({
                    f"{get_node_name(fovday_node)}\n{trial_type}": (
                        partial(stack_view, plot_manual=plot_manual, sync_events=alignment_events),
                        trial_dataset)
                    for trial_type, trial_dataset in trial_types.items()
                    })

                # add subtract view is two trial types are available
                if len(trial_types) == 2:
                    trial_type1, trial_type2 = trial_types.keys()
                    total_mosaic[-1].append(f"{get_node_name(fovday_node)}\nSubtraction")
                    content_dict[f"{get_node_name(fovday_node)}\nSubtraction"] = (
                        partial(subtract_view, plot_manual=plot_manual, sync_events=alignment_events, 
                                subtract_manual=SUBTRACT_MANUAL(name1=trial_type1, name2=trial_type2)),
                        list(trial_types.values())
                    )               
                    max_n_col = max(max_n_col, 3)
                else:
                    max_n_col = max(max_n_col, len(trial_types))

            for i in range(len(total_mosaic)):
                total_mosaic[i] += ["."] * (max_n_col - len(total_mosaic[i]))

            default_style(
                mosaic_style=total_mosaic,
                content_dict=content_dict,
                figsize=(max_n_col * UNIT_X_INCHES, UNIT_Y_INCHES * len(total_mosaic)),
                save_path=routing.default_fig_path(dataset.subtree(fov_node), f"FOVSummary_{{}}_{alignment_name}.png"),
            )
        except Exception as e:
            warnings.warn(f"Cannot plot trial average for {get_node_name(fov_node)} with {alignment_name}: {e}")
