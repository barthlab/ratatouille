from typing import Tuple
import numpy as np
import logging

from kitchen.plotter.plotting_manual import PlotManual
from kitchen.settings.plotting import PLOTTING_OVERLAP_HARSH_MODE
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import find_only_one, select_from_key


logger = logging.getLogger(__name__)


def sync_check(dataset: DataSet, sync_events: Tuple[str], plot_manual: PlotManual=PlotManual()):    
    # assert len(dataset) >= 2, f"Dataset should have at least 2 nodes for sync check, got {len(dataset)}"

    if PLOTTING_OVERLAP_HARSH_MODE:
        """All nodes should have same availability status of neural data"""
        all_status = [select_from_key(node.data.status(), **plot_manual._asdict()) for node in dataset]
        assert all(status == all_status[0] for status in all_status), "Neural data availability mismatch, see status_report.xlsx"

    """All nodes should contain unique align event in timeline"""
    for node in dataset:
        assert node.data.timeline is not None, f"Cannot find timeline in {node}"
        find_only_one(node.data.timeline.v, _self=lambda x: x in sync_events)
    
    """All nodes should have same number of cells"""
    if plot_manual.fluorescence:
        all_cells = [node.data.fluorescence.num_cell for node in dataset if node.data.fluorescence is not None]
        assert len(set(all_cells)) == 1, f"Number of cells mismatch, got {all_cells}"
        if all_cells[0] > 1:
            """If nodes have multiple cells, all nodes should have same cell idx"""
            all_cell_idx = [node.data.fluorescence.cell_idx for node in dataset if node.data.fluorescence is not None]
            assert all(np.array_equal(cell_idx, all_cell_idx[0]) for cell_idx in all_cell_idx), f"Cell idx mismatch, got {all_cell_idx}"
            all_node_object_uid = [node.object_uid for node in dataset if node.data.fluorescence is not None]
            assert len(set(all_node_object_uid)) == 1, f"Object uid mismatch, got {all_node_object_uid}"
        

def sync_nodes(dataset: DataSet, sync_events: Tuple[str], plot_manual: PlotManual=PlotManual()):
    sync_check(dataset, sync_events, plot_manual)

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
        try:
            assert node.data.timeline is not None, f"Cannot find timeline in {node}"
            assert len(node.data.timeline) > 0, f"Cannot find any event in timeline of {node}"
            aligned_nodes.append(node.aligned_to(node.data.timeline.t[0]))
        except Exception as e:
            logger.debug(f"Cannot align {node} to leftmost time point: {e}")
    return DataSet(name=dataset.name + "_left_aligned", nodes=aligned_nodes)

