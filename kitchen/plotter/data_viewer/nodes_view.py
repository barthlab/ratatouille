from typing import Tuple
import warnings

from kitchen.operator.grouping import grouping_timeseries
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.data_viewer.group_view import group_plot_whisker, group_plot_pupil, group_plot_locomotion, group_plot_lick, group_plot_timeline, group_plot_single_cell_fluorescence
from kitchen.plotter.decorators import TraceStacker, default_style_single_ax
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, LICK_RATIO, LOCOMOTION_RATIO, POSITION_RATIO, PUPIL_RATIO, TIMELINE_RATIO, WHISKER_RATIO
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.structure.neural_data_structure import Fluorescence
from kitchen.utils.sequence_kit import find_only_one
from kitchen.plotter.data_viewer.unit_view import unit_assertion, unit_plot_locomotion, unit_plot_lick, unit_plot_position, unit_plot_pupil, unit_plot_single_cell_fluorescence, unit_plot_timeline, unit_plot_whisker


@default_style_single_ax(xlabel="Time (s)")
def node_flat_view(
    ax, dataset: DataSet,
    locomotion_flag: bool = True,
    lick_flag: bool = True,
    pupil_flag: bool = True,
    tongue_flag: bool = True,
    whisker_flag: bool = True,
    fluorescence_flag: bool = True,
):
    """Plot a flat view of a node."""
    offset_tracker = TraceStacker()
    node = find_only_one(dataset.nodes)

    """main logic"""
    # 1. plot_timeline
    offset_tracker(unit_plot_timeline)(ax=ax, timeline=node.data.timeline, ratio=TIMELINE_RATIO)
    
    # 2. plot fluorescence
    if fluorescence_flag:
        unit_assertion(node.data.fluorescence, Fluorescence)
        for cell_id in range(node.data.fluorescence.num_cell):
            offset_tracker(unit_plot_single_cell_fluorescence)(
                ax=ax, fluorescence=node.data.fluorescence, cell_id=cell_id, ratio=FLUORESCENCE_RATIO)

    # 3. plot behavior
    if lick_flag:
        offset_tracker(unit_plot_lick)(ax=ax, lick=node.data.lick, ratio=LICK_RATIO)

    # 4. plot locomotion
    if locomotion_flag:
        offset_tracker(unit_plot_locomotion)(ax=ax, locomotion=node.data.locomotion, ratio=LOCOMOTION_RATIO)
        offset_tracker(unit_plot_position)(ax=ax, position=node.data.position, ratio=POSITION_RATIO)

    # 5. plot whisker
    if whisker_flag:
        offset_tracker(unit_plot_whisker)(ax=ax, whisker=node.data.whisker, ratio=WHISKER_RATIO)

    # 6. plot pupil
    if pupil_flag:
        offset_tracker(unit_plot_pupil)(ax=ax, pupil=node.data.pupil, ratio=PUPIL_RATIO)

    return 5, min(offset_tracker.offset / 8, 11)


@default_style_single_ax(xlabel="Time (s)")
def node_avg_view(
    ax, dataset_presync: DataSet,
    sync_events: Tuple[str],
    locomotion_flag: bool = True,
    lick_flag: bool = True,
    pupil_flag: bool = True,
    tongue_flag: bool = True,
    whisker_flag: bool = True,
    fluorescence_flag: bool = True,
):
    """Plot an average view of a node."""
    offset_tracker = TraceStacker()
    try:
        dataset = sync_nodes(dataset_presync, sync_events)
    except Exception as e:
        warnings.warn(f"Cannot sync nodes: {e}")
        return None
    
    """main logic"""
    # 1. plot_timeline
    offset_tracker(group_plot_timeline)(ax=ax, timelines=[node.data.timeline for node in dataset], ratio=TIMELINE_RATIO)
    
    # 2. plot fluorescence
    if fluorescence_flag:
        group_fluorescence = grouping_timeseries([node.data.fluorescence.df_f0 for node in dataset if node.data.fluorescence is not None])
        first_node_fluorescence = dataset.nodes[0].data.fluorescence
        assert first_node_fluorescence is not None, "Cannot find fluorescence in first node"
        for cell_id in range(first_node_fluorescence.num_cell):
            offset_tracker(group_plot_single_cell_fluorescence)(
                ax=ax, group_fluorescence=group_fluorescence, cell_id=cell_id, ratio=FLUORESCENCE_RATIO)

    # 3. plot behavior
    if lick_flag:
        offset_tracker(group_plot_lick)(ax=ax, licks=[node.data.lick for node in dataset], ratio=LICK_RATIO)

    # 4. plot locomotion
    if locomotion_flag:
        offset_tracker(group_plot_locomotion)(ax=ax, locomotions=[node.data.locomotion for node in dataset], ratio=LOCOMOTION_RATIO)

    # 5. plot whisker
    if whisker_flag:
        offset_tracker(group_plot_whisker)(ax=ax, whiskers=[node.data.whisker for node in dataset], ratio=WHISKER_RATIO)

    # 6. plot pupil
    if pupil_flag:
        offset_tracker(group_plot_pupil)(ax=ax, pupils=[node.data.pupil for node in dataset], ratio=PUPIL_RATIO)

    return 2, min(offset_tracker.offset / 5, 11)