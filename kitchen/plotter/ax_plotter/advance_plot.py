
from typing import Generator, Tuple
import matplotlib.pyplot as plt

from kitchen.operator.sync_nodes import sync_check, sync_nodes
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, TIMELINE_RATIO, LICK_RATIO, LOCOMOTION_RATIO, PUPIL_RATIO, WHISKER_RATIO
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_timeline
from kitchen.plotter.unit_plotter.unit_trace_advance import SUBTRACT_MANUAL, unit_subtract_lick, unit_subtract_locomotion, unit_subtract_pupil, unit_subtract_single_cell_fluorescence, unit_subtract_whisker
from kitchen.plotter.utils.tick_labels import add_line_legend
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import select_truthy_items

def subtract_view(
        ax: plt.Axes,
        datasets: list[DataSet],
        sync_events: Tuple[str],

        subtract_manual: SUBTRACT_MANUAL = SUBTRACT_MANUAL(),
        plot_manual: PlotManual = PlotManual(),
) -> Generator[float, float, None]:
    assert len(datasets) == 2, "Subtract view only works for 2 datasets"
    
    dataset1, dataset2 = datasets
    try:
        sync_check(dataset1 + dataset2, sync_events, plot_manual)
        dataset1_synced = sync_nodes(dataset1, sync_events, plot_manual)
        dataset2_synced = sync_nodes(dataset2, sync_events, plot_manual)
    except Exception as e:
        raise ValueError(f"Cannot sync nodes: {e}")
    
    """main logic"""
    y_offset = 0

    # 1. plot_timeline    
    if plot_manual.timeline: 
        y_offset = yield unit_plot_timeline( 
            timeline=select_truthy_items(
                [node.data.timeline for node in dataset1_synced] + [node.data.timeline for node in dataset2_synced]), 
            ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)   

    # 2. plot fluorescence
    if plot_manual.fluorescence:
        valid_fluorescence1 = select_truthy_items([node.data.fluorescence for node in dataset1_synced])    
        valid_fluorescence2 = select_truthy_items([node.data.fluorescence for node in dataset2_synced])      
        cell_id_flag = valid_fluorescence1[0].num_cell > 1
        for cell_id in range(valid_fluorescence1[0].num_cell):
            y_offset = yield unit_subtract_single_cell_fluorescence(
                fluorescence1=[fluorescence.extract_cell(cell_id) for fluorescence in valid_fluorescence1],
                fluorescence2=[fluorescence.extract_cell(cell_id) for fluorescence in valid_fluorescence2],
                subtract_manual=subtract_manual,
                ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO, cell_id_flag=cell_id_flag)
    
    # 3. plot behavior
    if plot_manual.lick:
        y_offset = yield unit_subtract_lick(
            lick1=select_truthy_items([node.data.lick for node in dataset1_synced]),
            lick2=select_truthy_items([node.data.lick for node in dataset2_synced]),
            subtract_manual=subtract_manual,
            ax=ax, y_offset=y_offset, ratio=LICK_RATIO)

    # 4. plot locomotion
    if plot_manual.locomotion:
        y_offset = yield unit_subtract_locomotion(
            locomotion1=select_truthy_items([node.data.locomotion for node in dataset1_synced]),
            locomotion2=select_truthy_items([node.data.locomotion for node in dataset2_synced]),
            subtract_manual=subtract_manual,
            ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)

    # 5. plot whisker
    if plot_manual.whisker:
        y_offset = yield unit_subtract_whisker(
            whisker1=select_truthy_items([node.data.whisker for node in dataset1_synced]),
            whisker2=select_truthy_items([node.data.whisker for node in dataset2_synced]),
            subtract_manual=subtract_manual,
            ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)

    # 6. plot pupil
    if plot_manual.pupil:
        y_offset = yield unit_subtract_pupil(
            pupil1=select_truthy_items([node.data.pupil for node in dataset1_synced]),
            pupil2=select_truthy_items([node.data.pupil for node in dataset2_synced]),
            subtract_manual=subtract_manual,
            ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)

    # 7. add legend
    if subtract_manual.name1 is not None and subtract_manual.name2 is not None:
        add_line_legend(ax, {subtract_manual.name1: {"color": subtract_manual.color1, "lw": 0.5}, 
                             subtract_manual.name2: {"color": subtract_manual.color2, "lw": 0.5}})
