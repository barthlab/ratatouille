from typing import List, Tuple
import warnings

from kitchen.operator.grouping import grouping_timeseries
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.data_viewer.group_view import group_plot_whisker, group_plot_pupil, group_plot_locomotion, group_plot_lick, group_plot_timeline, group_plot_single_cell_fluorescence
from kitchen.plotter.decorators import BBOX, TraceStacker, default_style_multi_ax
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, LICK_RATIO, LOCOMOTION_RATIO, PUPIL_RATIO, TIMELINE_RATIO, WHISKER_RATIO

@default_style_multi_ax(xlabel="Time (s)")
def datasets_avg_view(
    axs, datasets_presync: List[DataSet],
    sync_events: List[Tuple[str]],
    titles: List[str],
    locomotion_flag: bool = True,
    lick_flag: bool = True,
    pupil_flag: bool = True,
    tongue_flag: bool = True,
    whisker_flag: bool = True,
    fluorescence_flag: bool = True,
):
    """Plot an average view of a node."""
    assert len(axs) == len(datasets_presync) == len(sync_events) == len(titles), \
        f"Number of axes must match number of datasets and sync events, got {len(axs)} vs {len(datasets_presync)} vs {len(sync_events)} vs {len(titles)}"
    
    offset_tracker = TraceStacker()    
    datasets = []
    for dtst_prsc, sc_vt in zip(datasets_presync, sync_events):
        try:
            datasets.append(sync_nodes(dtst_prsc, sc_vt))
        except Exception as e:
            datasets.append(DataSet(nodes=[]))
            warnings.warn(f"Cannot sync nodes: {e}")
    if not datasets:
        return None
    
    for ax, title in zip(axs, titles):
        ax.set_title(title)

    """main logic"""
    # 1. plot_timeline
    @offset_tracker
    def plot_timeline(y_offset):
        bboxs = []
        for ax, dataset in zip(axs, datasets):
            if dataset.nodes:
                bbox = group_plot_timeline(
                    timelines=[node.timeline for node in dataset], ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)
                bboxs.append(bbox)
        big_bbox = BBOX(min([bbox.ymin for bbox in bboxs]), max([bbox.ymax for bbox in bboxs]))
        return big_bbox
    plot_timeline()
    
    # 2. plot fluorescence
    if fluorescence_flag:
        raise NotImplementedError

    # 3. plot behavior
    if lick_flag:
        @offset_tracker
        def plot_lick(y_offset):
            bboxs = []
            for ax, dataset in zip(axs, datasets):
                if dataset.nodes:
                    bbox = group_plot_lick(
                        licks=[node.lick for node in dataset], ax=ax, y_offset=y_offset, ratio=LICK_RATIO)
                    bboxs.append(bbox)
            big_bbox = BBOX(min([bbox.ymin for bbox in bboxs]), max([bbox.ymax for bbox in bboxs]))
            return big_bbox
        plot_lick()

    # 4. plot locomotion
    if locomotion_flag:
        @offset_tracker
        def plot_locomotion(y_offset):
            bboxs = []
            for ax, dataset in zip(axs, datasets):
                if dataset.nodes:
                    bbox = group_plot_locomotion(
                        locomotions=[node.locomotion for node in dataset], ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)
                    bboxs.append(bbox)
            big_bbox = BBOX(min([bbox.ymin for bbox in bboxs]), max([bbox.ymax for bbox in bboxs]))
            return big_bbox
        plot_locomotion()

    # 5. plot whisker
    if whisker_flag:
        @offset_tracker
        def plot_whisker(y_offset):
            bboxs = []
            for ax, dataset in zip(axs, datasets):
                if dataset.nodes:
                    bbox = group_plot_whisker(
                        whiskers=[node.whisker for node in dataset], ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)
                    bboxs.append(bbox)
            big_bbox = BBOX(min([bbox.ymin for bbox in bboxs]), max([bbox.ymax for bbox in bboxs]))
            return big_bbox
        plot_whisker()  

    # 6. plot pupil
    if pupil_flag:
        @offset_tracker
        def plot_pupil(y_offset):
            bboxs = []
            for ax, dataset in zip(axs, datasets):
                if dataset.nodes:
                    bbox = group_plot_pupil(
                        pupils=[node.pupil for node in dataset], ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)
                    bboxs.append(bbox)
            big_bbox = BBOX(min([bbox.ymin for bbox in bboxs]), max([bbox.ymax for bbox in bboxs]))
            return big_bbox
        plot_pupil()  

    return 2*len(axs), min(offset_tracker.offset / 5, 11)