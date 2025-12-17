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
import logging
from typing import Optional, Tuple
import traceback

from kitchen.configs import routing
from kitchen.configs.naming import get_node_name
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.plotter.ax_plotter.advance_plot import subtract_view
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL, PlotManual
from kitchen.plotter.plotting_params import FLAT_X_INCHES, FLAT_Y_INCHES, PARALLEL_X_INCHES, PARALLEL_Y_INCHES, STACK_X_INCHES, STACK_Y_INCHES, UNIT_X_INCHES, UNIT_Y_INCHES
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.plotter.decorators.default_decorators import default_style
from kitchen.plotter.ax_plotter.basic_plot import beam_view, flat_view, heatmap_view, stack_view
from kitchen.utils.sequence_kit import get_sorted_merge_indices, select_from_value


logger = logging.getLogger(__name__)


def flat_view_default_macro(
        dataset: DataSet,
        node_level: str,
        plot_manual: PlotManual,
        prefix_keyword: Optional[str] = None,
        unit_shape: Tuple[float, float] = (FLAT_X_INCHES, FLAT_Y_INCHES),

        **kwargs,
) -> None:
    """
    Generate flat view overview plots for all nodes within a dataset.    
    Creates a multi-panel figure with one row per node.
    """
    nodes2plot = dataset.select(node_level)
    save_name = f"{dataset.name}_{node_level}_flatview"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name
    try:
        default_style(
            mosaic_style=[[get_node_name(node),] for node in nodes2plot],
            content_dict={
                get_node_name(node): (
                    partial(flat_view, plot_manual=plot_manual),
                    DataSet(name=str(node.coordinate), nodes=[node]))
                for node in nodes2plot
                },
            figsize=(unit_shape[0], unit_shape[1] * len(nodes2plot)),
            save_path=routing.default_fig_path(dataset, prefix_str + "_{}.png"),
            **kwargs,
        )
    except Exception as e:
        logger.debug(f"Cannot plot flat view default macro for {dataset.name}: {e}")
        logger.debug(traceback.format_exc())


def stack_view_default_macro(
        dataset: DataSet,
        node_level: str,
        plot_manual: PlotManual,
        prefix_keyword: Optional[str] = None,
        unit_shape: Tuple[float, float] = (STACK_X_INCHES, STACK_Y_INCHES),

        _element_trial_level: str = "trial",
        _append_subtract_view: bool = True,  # for two types node only
        _aligment_style: str = "Aligned2Trial",

        **kwargs,
) -> None:
    """
    Generate stack view overview plots for all nodes within a dataset.    
    Creates a multi-panel figure with each row per node and each column per selected type.
    """
    nodes2plot = dataset.select(node_level)
    save_name = f"{dataset.name}_{_element_trial_level}@{node_level}_stackview"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    try:
        n_col = 0
        mosaic_style, content_dict = [], {}
        for node in nodes2plot:            
            type2dataset = split_dataset_by_trial_type(dataset.subtree(node), 
                                                       plot_manual=plot_manual,
                                                       _element_trial_level =_element_trial_level)
            if len(type2dataset) == 0:
                continue
            
            # update mosaic_style and content_dict
            mosaic_style.append([f"{get_node_name(node)}\n{selected_type}" 
                                    for selected_type in type2dataset.keys()])
            content_dict.update({
                f"{get_node_name(node)}\n{selected_type}": (
                    partial(stack_view, plot_manual=plot_manual, sync_events=alignment_events),
                    type2dataset[selected_type])
                for selected_type in type2dataset.keys()
                })
            
            if _append_subtract_view and len(type2dataset) == 2:
                selected_type1, selected_type2 = type2dataset.keys()
                mosaic_style[-1].append(f"{get_node_name(node)}\nSubtraction")
                content_dict[f"{get_node_name(node)}\nSubtraction"] = (
                    partial(subtract_view, plot_manual=plot_manual, sync_events=alignment_events, 
                            subtract_manual=SUBTRACT_MANUAL(name1=selected_type1, name2=selected_type2)),
                    list(type2dataset.values())
                )
            
            n_col = max(n_col, len(mosaic_style[-1]))
        for i in range(len(mosaic_style)):
            mosaic_style[i] += ["."] * (n_col - len(mosaic_style[i]))

        default_style(
            mosaic_style=mosaic_style,
            content_dict=content_dict,
            figsize=(unit_shape[0] * n_col, unit_shape[1] * len(mosaic_style)),
            save_path=routing.default_fig_path(dataset, prefix_str + f"_{{}}_{_aligment_style}.png"),
            **kwargs,
        )
    except Exception as e:
        logger.debug(f"Cannot plot stack view default macro for {dataset.name} with {_aligment_style}: {e}")
        logger.debug(traceback.format_exc())


def subtract_view_default_macro(
        dataset: DataSet,
        node_level: str,
        plot_manual: PlotManual,
        prefix_keyword: Optional[str] = None,
        unit_shape: Tuple[float, float] = (STACK_X_INCHES, STACK_Y_INCHES),

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",
        _target_types: str = "CuedPuff",

        **kwargs,
) -> None:
    """
    Generate stack view overview plots for all nodes within a dataset.    
    Creates a multi-panel figure with each row per node and each column per selected type.
    """
    nodes2plot = dataset.select(node_level)
    save_name = f"{dataset.name}_{_element_trial_level}@{node_level}_subtractview_{_target_types}"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    try:
        n_col = 0
        mosaic_style, content_dict = [], {}
        for node in nodes2plot:
            type2dataset = split_dataset_by_trial_type(dataset.subtree(node), 
                                                       plot_manual=plot_manual,
                                                       _element_trial_level =_element_trial_level)
            if len(type2dataset) == 0:
                continue
            
            # update mosaic_style and content_dict
            mosaic_style.append([f"{get_node_name(node)}\n{selected_type} vs {_target_types}" 
                                    for selected_type in type2dataset.keys() if selected_type != _target_types])
            content_dict.update({
                f"{get_node_name(node)}\n{selected_type} vs {_target_types}": (
                    partial(subtract_view, plot_manual=plot_manual, sync_events=alignment_events,
                            subtract_manual=SUBTRACT_MANUAL(name1=selected_type, name2=_target_types)),
                    [type2dataset[selected_type], type2dataset[_target_types]])
                for selected_type in type2dataset.keys() if selected_type != _target_types
                })         
            
            n_col = max(n_col, len(mosaic_style[-1]))
        for i in range(len(mosaic_style)):
            mosaic_style[i] += ["."] * (n_col - len(mosaic_style[i]))

        default_style(
            mosaic_style=mosaic_style,
            content_dict=content_dict,
            figsize=(unit_shape[0] * n_col, unit_shape[1] * len(mosaic_style)),
            save_path=routing.default_fig_path(dataset, prefix_str + f"_{{}}_{_aligment_style}.png"),
            **kwargs,
        )
    except Exception as e:
        logger.debug(f"Cannot plot subtract view default macro for {dataset.name} with {_aligment_style}: {e}")
        logger.debug(traceback.format_exc())


def heatmap_view_default_macro(
        dataset: DataSet,
        node_level: str,
        plot_manual: PlotManual,
        prefix_keyword: Optional[str] = None,
        unit_shape: Tuple[float, float] = (STACK_X_INCHES, STACK_Y_INCHES),

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",
        _target_modality: str = "Whisker",
        _sort_row: bool = False,
        _add_dummy: bool = False,

        **kwargs,
) -> None:
    """
    Generate heatmap view overview plots for all nodes within a dataset.    
    Creates a multi-panel figure with each row per node and each column per selected type.
    """
    _target_modality = _target_modality.lower()
    nodes2plot = dataset.select(node_level)
    save_name = f"{dataset.name}_{_element_trial_level}@{node_level}_heatmapview_{_target_modality}{'_sorted' if _sort_row else '_unsorted'}"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    try:
        n_col = 0
        mosaic_style, content_dict = [], {}
        for node in nodes2plot:
            type2dataset = split_dataset_by_trial_type(dataset.subtree(node), 
                                                       plot_manual=plot_manual,
                                                       _element_trial_level =_element_trial_level,
                                                       _add_dummy=_add_dummy)
            if len(type2dataset) == 0:
                continue
            
            # update mosaic_style and content_dict
            mosaic_style.append([f"{get_node_name(node)}\n{selected_type} {_target_modality} {'Sorted' if _sort_row else ''}" 
                                    for selected_type in type2dataset.keys()])
            content_dict.update({
                f"{get_node_name(node)}\n{selected_type} {_target_modality} {'Sorted' if _sort_row else ''}" : (
                    partial(heatmap_view, plot_manual=plot_manual, sync_events=alignment_events, modality_name=_target_modality, _sort_rows=_sort_row), 
                    type2dataset[selected_type])
                for selected_type in type2dataset.keys()
                })
                
            n_col = max(n_col, len(mosaic_style[-1]))
        for i in range(len(mosaic_style)):
            mosaic_style[i] += ["."] * (n_col - len(mosaic_style[i]))

        default_style(
            mosaic_style=mosaic_style,
            content_dict=content_dict,
            figsize=(unit_shape[0] * n_col, unit_shape[1] * len(mosaic_style)),
            save_path=routing.default_fig_path(dataset, prefix_str + f"_{{}}_{_aligment_style}.png"),
            sharey=False,
            **kwargs,
        )
    except Exception as e:
        logger.debug(f"Cannot plot heatmap view default macro for {dataset.name} with {_aligment_style}: {e}")
        logger.debug(traceback.format_exc())


def beam_view_default_macro(
        dataset: DataSet,
        node_level: str,
        plot_manual: PlotManual,
        prefix_keyword: Optional[str] = None,
        unit_shape: Tuple[float, float] = (PARALLEL_X_INCHES, PARALLEL_Y_INCHES),

        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",

        **kwargs,
) -> None:
    """
    Generate beam view overview plots for all nodes within a dataset.    
    Creates a multi-panel figure with each row per node and each column per selected type.
    """
    nodes2plot = dataset.select(node_level)
    save_name = f"{dataset.name}_{_element_trial_level}@{node_level}_beamview"
    prefix_str = f"{prefix_keyword}_{save_name}" if prefix_keyword is not None else save_name

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    try:
        n_col = 0
        mosaic_style, content_dict = [], {}
        for node in nodes2plot:
            type2dataset = split_dataset_by_trial_type(dataset.subtree(node), 
                                                       plot_manual=plot_manual,
                                                       _element_trial_level =_element_trial_level)
            if len(type2dataset) == 0:
                continue
            type2beamoffset = get_sorted_merge_indices({
                trial_type: subtype_dataset.nodes for trial_type, subtype_dataset in type2dataset.items()},
                _convert2offset=True,
            )
            
            # update mosaic_style and content_dict
            mosaic_style.append([f"{get_node_name(node)}\n{selected_type}" 
                                    for selected_type in type2dataset.keys()])
            content_dict.update({
                f"{get_node_name(node)}\n{selected_type}": (
                    partial(beam_view, plot_manual=plot_manual, sync_events=alignment_events, 
                            beam_offsets=type2beamoffset[selected_type]),
                    type2dataset[selected_type])
                for selected_type in type2dataset.keys()
                })
            n_col = max(n_col, len(mosaic_style[-1]))

        for i in range(len(mosaic_style)):
            mosaic_style[i] += ["."] * (n_col - len(mosaic_style[i]))

        default_style(
            mosaic_style=mosaic_style,
            content_dict=content_dict,
            figsize=(unit_shape[0] * n_col, unit_shape[1] * len(mosaic_style)),
            save_path=routing.default_fig_path(dataset, prefix_str + f"_{{}}_{_aligment_style}.png"),
            overlap_ratio=0.7,
            **kwargs,
        )
    except Exception as e:
        logger.debug(f"Cannot plot beam view default macro for {dataset.name} with {_aligment_style}: {e}")
        logger.debug(traceback.format_exc())