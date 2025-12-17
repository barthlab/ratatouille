
from typing import Generator, Optional, Tuple
import matplotlib.pyplot as plt
import logging

import numpy as np

from kitchen.operator.sync_nodes import left_align_nodes, sync_nodes
from kitchen.plotter import style_dicts
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, HEATMAP_OFFSET_RANGE, LICK_RATIO, LOCOMOTION_RATIO, NOSE_RATIO, POSITION_RATIO, POTENTIAL_RATIO, PUPIL_RATIO, SACCADE_RATIO, TIMELINE_RATIO, WHISKER_RATIO
from kitchen.plotter.unit_plotter.unit_heatmap import unit_heatmap_locomotion, unit_heatmap_pupil, unit_heatmap_saccade, unit_heatmap_whisker
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_lick, unit_plot_locomotion, unit_plot_nose, unit_plot_position, unit_plot_potential_conv, unit_plot_pupil, unit_plot_pupil_center, unit_plot_single_cell_fluorescence, unit_plot_timeline, unit_plot_whisker, unit_plot_potential
from kitchen.plotter.utils.tick_labels import add_labeless_yticks, emphasize_yticks
from kitchen.settings.potential import WC_CONVERT_FLAG
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import count_by, find_only_one, select_truthy_items


logger = logging.getLogger(__name__)


def flat_view(
        ax: plt.Axes,
        datasets: DataSet,
        
        plot_manual: PlotManual,
) -> Generator[float, float, None]:
    """Flat view of the only node in the dataset"""
    try:
        dataset_aligned = left_align_nodes(datasets)
    except Exception as e:
        raise ValueError(f"Cannot align nodes: {e}")
    
    try:
        node = find_only_one(dataset_aligned.nodes)
    except Exception as e:
        logger.debug(f"Cannot find unique node in dataset: {e}")
        return
    
    """main logic"""
    y_offset = 0

    # 1. plot_timeline    
    if plot_manual.timeline:
        y_offset = yield unit_plot_timeline(timeline=node.data.timeline, ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)
    
    # 2. plot fluorescence / potential
    if plot_manual.fluorescence:
        if node.data.fluorescence is None:
            y_offset = yield 0
        else:
            for cell_id in range(node.data.fluorescence.num_cell):
                y_offset = yield unit_plot_single_cell_fluorescence(
                    fluorescence=node.data.fluorescence.extract_cell(cell_id), ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO)
                
    elif plot_manual.potential:
        if node.data.potential is None:
            y_offset = yield 0
        else:
            y_offset = yield unit_plot_potential(potential=node.data.potential, ax=ax, y_offset=y_offset, ratio=POTENTIAL_RATIO,
                                                 aspect=plot_manual.potential, wc_flag=WC_CONVERT_FLAG(node))
    elif plot_manual.potential_conv:
        if node.data.potential is None:
            y_offset = yield 0
        else:
            y_offset = yield unit_plot_potential_conv(potential=node.data.potential, ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO,)
            
    # 3. plot behavior
    if plot_manual.lick:
        y_offset = yield unit_plot_lick(lick=node.data.lick, ax=ax, y_offset=y_offset, ratio=LICK_RATIO)

    # 4. plot locomotion
    if plot_manual.locomotion:      
        y_offset = yield unit_plot_locomotion(locomotion=node.data.locomotion, ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)
    if plot_manual.position:
        y_offset = yield unit_plot_position(position=node.data.position, ax=ax, y_offset=y_offset, ratio=POSITION_RATIO)

    # 5. plot whisker
    if plot_manual.whisker:      
        y_offset = yield unit_plot_whisker(whisker=node.data.whisker, ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)

    # 5.5 plot nose
    if plot_manual.nose:      
        y_offset = yield unit_plot_nose(nose=node.data.nose, ax=ax, y_offset=y_offset, ratio=NOSE_RATIO)

    # 6. plot pupil
    if plot_manual.pupil:             
        y_offset = yield unit_plot_pupil(pupil=node.data.pupil, ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)
    if plot_manual.saccade:             
        y_offset = yield unit_plot_pupil_center(pupil=node.data.pupil, ax=ax, y_offset=y_offset, ratio=SACCADE_RATIO)



def stack_view(
        ax: plt.Axes,
        datasets: DataSet,
        sync_events: Tuple[str],

        plot_manual: PlotManual = PlotManual(),
) -> Generator[float, float, None]:
    """Stack view of all nodes in the dataset"""
    try:
        dataset_synced = sync_nodes(datasets, sync_events, plot_manual)
    except Exception as e:
        raise ValueError(f"Cannot sync nodes: {e}")
    
    """main logic"""
    y_offset = 0

    # 1. plot_timeline    
    if plot_manual.timeline:    
        y_offset = yield unit_plot_timeline(    
            timeline=select_truthy_items([node.data.timeline for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)    

    # 2. plot fluorescence / potential
    if plot_manual.fluorescence:    
        valid_fluorescence = select_truthy_items([node.data.fluorescence for node in dataset_synced])    
        cell_id_flag = valid_fluorescence[0].num_cell > 1
        for cell_id in range(valid_fluorescence[0].num_cell):
            y_offset = yield unit_plot_single_cell_fluorescence(
                fluorescence=[fluorescence.extract_cell(cell_id) for fluorescence in valid_fluorescence], 
                ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO, cell_id_flag=cell_id_flag)
    elif plot_manual.potential:    
        valid_potential = select_truthy_items([node.data.potential for node in dataset_synced])    
        y_offset = yield unit_plot_potential(
            potential=valid_potential, ax=ax, y_offset=y_offset, ratio=POTENTIAL_RATIO, 
            aspect=plot_manual.potential, wc_flag=WC_CONVERT_FLAG(dataset_synced.nodes[0]))

    # 3. plot behavior  
    if plot_manual.lick:    
        y_offset = yield unit_plot_lick(
            lick=select_truthy_items([node.data.lick for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=LICK_RATIO)
        
    # 4. plot locomotion
    if plot_manual.locomotion:    
        y_offset = yield unit_plot_locomotion(
            locomotion=select_truthy_items([node.data.locomotion for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO, baseline_subtraction=plot_manual.baseline_subtraction)

    # 5. plot whisker
    if plot_manual.whisker:    
        y_offset = yield unit_plot_whisker(
            whisker=select_truthy_items([node.data.whisker for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO, baseline_subtraction=plot_manual.baseline_subtraction)
    
    if plot_manual.nose:    
        y_offset = yield unit_plot_nose(
            nose=select_truthy_items([node.data.nose for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=NOSE_RATIO)

    # 6. plot pupil
    if plot_manual.pupil:    
        y_offset = yield unit_plot_pupil(
            pupil=select_truthy_items([node.data.pupil for node in dataset_synced]),
            ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO, baseline_subtraction=plot_manual.baseline_subtraction)
    if plot_manual.saccade:    
        y_offset = yield unit_plot_pupil_center(
            pupil=select_truthy_items([node.data.pupil for node in dataset_synced]),
            ax=ax, y_offset=y_offset, ratio=SACCADE_RATIO, baseline_subtraction=plot_manual.baseline_subtraction)


def heatmap_view(
        ax: plt.Axes,
        datasets: DataSet,
        sync_events: Tuple[str],

        plot_manual: PlotManual = PlotManual(),
        modality_name: str = "whisker",
        _sort_rows: bool = False,
) -> Generator[float, float, None]:
    """Stack view of all nodes in the dataset"""
    try:
        dataset_synced = sync_nodes(datasets, sync_events, plot_manual)
    except Exception as e:
        raise ValueError(f"Cannot sync nodes: {e}")
    
    sorting_level_refs = [node.coordinate.temporal_uid.get_hier_value(plot_manual.amplitude_sorting[2]) for node in dataset_synced] if _sort_rows else None
    amplitude_sorting = plot_manual.amplitude_sorting[:2] if _sort_rows else None
    """main logic"""
    y_offset = 0
    
    # 1. plot_timeline    
    if plot_manual.timeline:    
        y_offset = yield unit_plot_timeline(    
            timeline=select_truthy_items([node.data.timeline for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)    

    # # 2. plot fluorescence / potential
    # if plot_manual.fluorescence:    
    #     valid_fluorescence = select_truthy_items([node.data.fluorescence for node in dataset_synced])    
    #     cell_id_flag = valid_fluorescence[0].num_cell > 1
    #     for cell_id in range(valid_fluorescence[0].num_cell):
    #         y_offset = yield unit_plot_single_cell_fluorescence(
    #             fluorescence=[fluorescence.extract_cell(cell_id) for fluorescence in valid_fluorescence], 
    #             ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO, cell_id_flag=cell_id_flag)
    # elif plot_manual.potential:    
    #     valid_potential = select_truthy_items([node.data.potential for node in dataset_synced])    
    #     y_offset = yield unit_plot_potential(
    #         potential=valid_potential, ax=ax, y_offset=y_offset, ratio=POTENTIAL_RATIO, 
    #         aspect=plot_manual.potential, wc_flag=WC_CONVERT_FLAG(dataset_synced.nodes[0]))

    # # 3. plot behavior  
    # if plot_manual.lick:    
    #     y_offset = yield unit_plot_lick(
    #         lick=select_truthy_items([node.data.lick for node in dataset_synced]), 
    #         ax=ax, y_offset=y_offset, ratio=LICK_RATIO)
        
    # 4. plot locomotion
    if modality_name == "locomotion":    
        y_offset = yield unit_heatmap_locomotion(
            locomotion=select_truthy_items([node.data.locomotion for node in dataset_synced]), 
            ax=ax, baseline_subtraction=plot_manual.baseline_subtraction,
            sorting_level_refs=sorting_level_refs,
            amplitude_sorting=amplitude_sorting)

    # 5. plot whisker
    if modality_name == "whisker":    
        y_offset = yield unit_heatmap_whisker(
            whisker=select_truthy_items([node.data.whisker for node in dataset_synced]), 
            ax=ax, baseline_subtraction=plot_manual.baseline_subtraction,
            sorting_level_refs=sorting_level_refs,
            amplitude_sorting=amplitude_sorting)
    
    # if plot_manual.nose:    
    #     y_offset = yield unit_plot_nose(
    #         nose=select_truthy_items([node.data.nose for node in dataset_synced]), 
    #         ax=ax, y_offset=y_offset, ratio=NOSE_RATIO)

    # 6. plot pupil
    if modality_name == "pupil":    
        y_offset = yield unit_heatmap_pupil(
            pupil=select_truthy_items([node.data.pupil for node in dataset_synced]),
            ax=ax, baseline_subtraction=plot_manual.baseline_subtraction,
            sorting_level_refs=sorting_level_refs,
            amplitude_sorting=amplitude_sorting)
    if modality_name == "saccade":    
        y_offset = yield unit_heatmap_saccade(
            saccade=select_truthy_items([node.data.pupil for node in dataset_synced]),
            ax=ax, baseline_subtraction=plot_manual.baseline_subtraction,
            sorting_level_refs=sorting_level_refs,
            amplitude_sorting=amplitude_sorting)
    

    # fancy horizontal line    
    row_height = (HEATMAP_OFFSET_RANGE[1] - HEATMAP_OFFSET_RANGE[0]) / len(dataset_synced)
    session_row_count = count_by(dataset_synced.nodes, lambda node: node.coordinate.temporal_uid.session_id, _sort_key=True)
    day_row_count = count_by(dataset_synced.nodes, lambda node: node.coordinate.temporal_uid.day_id, _sort_key=True)
    session_minor_ticks = [HEATMAP_OFFSET_RANGE[0] + row_height * row_num for row_num in np.cumsum(list(session_row_count.values()))]
    day_major_ticks = [HEATMAP_OFFSET_RANGE[0] + row_height * row_num for row_num in np.cumsum(list(day_row_count.values()))]
    if plot_manual.amplitude_sorting is None:
        emphasize_yticks(ax)
        add_labeless_yticks(ax, day_major_ticks, session_minor_ticks)
    elif plot_manual.amplitude_sorting[2] == "day":
        add_labeless_yticks(ax, day_major_ticks, [])
    elif plot_manual.amplitude_sorting[2] == "session":
        add_labeless_yticks(ax, day_major_ticks, session_minor_ticks)
    elif plot_manual.amplitude_sorting[2] == "mice":
        pass
   
  
def beam_view(
        ax: plt.Axes,
        datasets: DataSet,
        sync_events: Tuple[str],

        plot_manual: PlotManual = PlotManual(),
        beam_offsets: Optional[list[int]] = None,
):
    """Beam view of all nodes in the dataset"""
    try:
        dataset_synced = sync_nodes(datasets, sync_events, plot_manual)
    except Exception as e:
        raise ValueError(f"Cannot sync nodes: {e}")
    
    """main logic"""
    y_offset = 0
    beam_offsets = [0 for _ in dataset_synced] if beam_offsets is None else beam_offsets
    assert len(beam_offsets) == len(dataset_synced), f"Number of beam offsets mismatch, got {len(beam_offsets)} from {len(dataset_synced)}"

    # 1. plot_timeline    
    y_offset = yield unit_plot_timeline(    
        timeline=select_truthy_items([node.data.timeline for node in dataset_synced]), 
        ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)
    
    for node_index, (node, beam_offset) in enumerate(zip(dataset_synced, beam_offsets)):
        # skip the plotting if the beam offset is positive
        for _ in range(beam_offset):
            for _ in range(6):
                y_offset = yield 0
        
        show_yticks = (node_index == 0) and (beam_offset == 0)

        # 2. plot fluorescence / potential
        if plot_manual.fluorescence:    
            raise NotImplementedError
        elif plot_manual.potential:    
            y_offset = yield unit_plot_potential(
                potential=node.data.potential, ax=ax, y_offset=y_offset, ratio=POTENTIAL_RATIO, 
                aspect=plot_manual.potential, yticks_flag=show_yticks,
                wc_flag=WC_CONVERT_FLAG(dataset_synced.nodes[0]))
        else:
            y_offset = yield 0
        
        # 3. plot behavior  
        if plot_manual.lick:    
            y_offset = yield unit_plot_lick(
                lick=node.data.lick, ax=ax, y_offset=y_offset, ratio=LICK_RATIO, yticks_flag=show_yticks)
        else:
            y_offset = yield 0
        
        # 4. plot locomotion
        if plot_manual.locomotion:    
            y_offset = yield unit_plot_locomotion(
                locomotion=node.data.locomotion, ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO, yticks_flag=show_yticks)
            y_offset = yield unit_plot_position(
                position=node.data.position, ax=ax, y_offset=y_offset, ratio=POSITION_RATIO, yticks_flag=show_yticks)
        else:
            y_offset = yield 0

        # 5. plot whisker
        if plot_manual.whisker:    
            y_offset = yield unit_plot_whisker(
                whisker=node.data.whisker, ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO, yticks_flag=show_yticks)
        else:
            y_offset = yield 0
        
        # 5.5 plot nose
        if plot_manual.nose:    
            y_offset = yield unit_plot_nose(
                nose=node.data.nose, ax=ax, y_offset=y_offset, ratio=NOSE_RATIO, yticks_flag=show_yticks)
        else:
            y_offset = yield 0

        # 6. plot pupil
        if plot_manual.pupil:    
            y_offset = yield unit_plot_pupil(
                pupil=node.data.pupil, ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO, yticks_flag=show_yticks)
        else:
            y_offset = yield 0
