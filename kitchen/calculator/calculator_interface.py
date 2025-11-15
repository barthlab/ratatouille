"""
High-level interface for metric calculation across hierarchical datasets.
"""
from typing import Callable, Tuple
import numpy as np

from kitchen.calculator.data_interface import get_data
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.structure.neural_data_structure import Events, TimeSeries

def calculate_metric(
        src_dataset: DataSet, 
        dst_dataset: DataSet,
        data_name: str,
        calc_func: Callable[[TimeSeries | Events], float],
        sync_events: Tuple[str],

        projection_src: str = "trial",
        projection_dst: str = "mice",
    ) -> np.ndarray:
    """
    Calculate metrics across hierarchical dataset with projection between hierarchy levels.

    Computes metrics on specified data modality at source nodes, then aggregates
    results to destination nodes in the experimental hierarchy. 

    Args:
        dataset: Hierarchical dataset containing experimental nodes
        data_name: Name of data modality to analyze (e.g., 'fluorescence', 'lick', 'pupil')
        calc_func: Function to apply to each data segment, must return a float
        sync_events: Tuple of event names for temporal synchronization
        projection_src: Source hierarchy level for metric calculation (default: "trial")
        projection_dst: Destination hierarchy level for result aggregation (default: "mice")

    Returns:
        Array of computed metrics, one value per destination node.
        Values are averaged across all source nodes within each destination's subtree.
    """

    # calculate metric at src nodes
    src_nodes = src_dataset.select(projection_src)

    # Fluorescence
    if data_name == "fluorescence":
        computed_data = {node: calc_func(fluorescence.df_f0) 
                         for node, fluorescence in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    # Timeline
    elif data_name == "timeline":
        computed_data = {node: calc_func(timeline) 
                         for node, timeline in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    # Behavior Events
    elif data_name == "position":
        computed_data = {node: calc_func(position) 
                         for node, position in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    elif data_name == "locomotion":
        computed_data = {node: calc_func(locomotion) 
                         for node, locomotion in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    elif data_name == "lick":
        computed_data = {node: calc_func(lick) 
                         for node, lick in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    # Behavior Timeseries
    elif data_name == "pupil":
        computed_data = {node: calc_func(pupil) 
                         for node, pupil in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    elif data_name == "tongue":
        computed_data = {node: calc_func(tongue) 
                         for node, tongue in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    elif data_name == "nose":
        computed_data = {node: calc_func(nose) 
                         for node, nose in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    elif data_name == "whisker":
        computed_data = {node: calc_func(whisker) 
                         for node, whisker in zip(src_nodes, get_data(src_nodes, data_name, sync_events))}
    else:
        raise NotImplementedError(f"Cannot calculate metric for {data_name}")

    # collect metric at dst nodes    
    dst_nodes = dst_dataset.select(projection_dst)
    return np.array([np.nanmean([computed_data[src_node] for src_node in src_dataset.subtree(dst_node).select(projection_src)]) 
                     for dst_node in dst_nodes])
