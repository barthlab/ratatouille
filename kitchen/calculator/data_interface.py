"""
Data extraction and synchronization interface for hierarchical datasets.
"""
from typing import Iterator, Optional, Tuple, overload, Literal

from kitchen.operator.sync_nodes import left_align_nodes, sync_nodes
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.structure.neural_data_structure import Fluorescence, Events, TimeSeries, Timeline


@overload
def get_data(
        dataset: DataSet, 
        data_name: Literal["fluorescence"], 
        sync_events: Optional[Tuple[str]] = None
    ) -> Iterator[Fluorescence]:
    ...
    
@overload
def get_data(
        dataset: DataSet, 
        data_name: Literal["timeline"], 
        sync_events: Optional[Tuple[str]] = None
    ) -> Iterator[Timeline]:
    ...

@overload
def get_data(
        dataset: DataSet, 
        data_name: Literal["position", "locomotion", "lick"], 
        sync_events: Optional[Tuple[str]] = None
    ) -> Iterator[Events]:
    ...

@overload
def get_data(
        dataset: DataSet, 
        data_name: Literal["pupil", "tongue", "whisker"], 
        sync_events: Optional[Tuple[str]] = None
    ) -> Iterator[TimeSeries]:
    ...

# @overload
# def get_data(
#         dataset: DataSet, 
#         data_name: str,
#         sync_events: Optional[Tuple[str]] = None
#     ) -> Iterator[Events] | Iterator[TimeSeries] | Iterator[Timeline]:
#     ...

def get_data(
        dataset: DataSet,
        data_name: str,
        sync_events: Optional[Tuple[str]] = None
    ) -> Iterator[Fluorescence] | Iterator[Events] | Iterator[TimeSeries] | Iterator[Timeline]:
    """
    Extract specific data modality from hierarchical dataset with optional synchronization.

    Retrieves the specified data type from all nodes in the dataset, with optional
    synchronization to specific events. 

    Args:
        dataset: Hierarchical dataset containing experimental nodes
        data_name: Name of data modality to extract (e.g., 'fluorescence', 'lick', 'pupil')
        sync_events: Optional tuple of event names for temporal synchronization.
                    If None, uses left-alignment of nodes.

    Returns:
        Iterator yielding data objects of the appropriate type for each node.
        Return type depends on data_name:
        - 'fluorescence': Iterator[Fluorescence]
        - 'timeline': Iterator[Timeline]
        - 'position', 'locomotion', 'lick': Iterator[Events]
        - 'pupil', 'tongue', 'whisker': Iterator[TimeSeries]

    Raises:
        AssertionError: If number of nodes changes during synchronization
    """
    dataset_synced = left_align_nodes(dataset) if sync_events is None else \
        sync_nodes(dataset, sync_events, plot_manual=PlotManual(**{data_name: True}))
    assert len(dataset_synced) == len(dataset), f"Number of nodes mismatch after syncing, got {len(dataset_synced)} from {len(dataset)}"
    for node in dataset_synced:
        yield getattr(node.data, data_name)
