from typing import Tuple

from kitchen.settings.plotting import PLOTTING_OVERLAP_HARSH_MODE
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import find_only_one


def sync_nodes(dataset: DataSet, sync_events: Tuple[str]):
    assert len(dataset) >= 2, f"Dataset should have at least 2 nodes for sync check, got {len(dataset)}"

    if PLOTTING_OVERLAP_HARSH_MODE:
        """All nodes should have same availability status of neural data"""
        all_status = [node.data.status() for node in dataset]
        assert all(status == all_status[0] for status in all_status), "Neural data availability mismatch, see status_report.xlsx"

    """All nodes should contain unique align event in timeline"""
    for node in dataset:
        assert node.data.timeline is not None, f"Cannot find timeline in {node}"
        find_only_one(node.data.timeline.v, _self=lambda x: x in sync_events)
    
    # """All nodes should have same group of cells"""
    # all_cells = [tuple(node.data.fluorescence.cell_idx) for node in dataset if node.data.fluorescence is not None]
    # assert all(cells == all_cells[0] for cells in all_cells), f"Number of cells mismatch, got {all_cells}"
    
    """Align all nodes to the given sync events"""
    synced_nodes = []
    for node in dataset:
        assert node.data.timeline is not None, f"Cannot find timeline in {node}"
        node = node.aligned_to(node.data.timeline.filter(sync_events).t[0])
        synced_nodes.append(node)
    return DataSet(name=dataset.name + "_synced", nodes=synced_nodes)


def left_align_nodes(dataset: DataSet):
    """Align all nodes to the leftmost time point"""
    aligned_nodes = []
    for node in dataset:
        assert node.data.timeline is not None, f"Cannot find timeline in {node}"
        assert len(node.data.timeline) > 0, f"Cannot find any event in timeline of {node}"
        aligned_nodes.append(node.aligned_to(node.data.timeline.t[0]))
    return DataSet(name=dataset.name + "_left_aligned", nodes=aligned_nodes)

