
from typing import Generator, Tuple
import matplotlib.pyplot as plt

from kitchen.operator.sync_nodes import left_align_nodes, sync_nodes
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, LICK_RATIO, LOCOMOTION_RATIO, POSITION_RATIO, POTENTIAL_RATIO, PUPIL_RATIO, TIMELINE_RATIO, WHISKER_RATIO
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_lick, unit_plot_locomotion, unit_plot_position, unit_plot_pupil, unit_plot_single_cell_fluorescence, unit_plot_timeline, unit_plot_whisker, unit_plot_potential
from kitchen.settings.potential import WC_CONVERT_FLAG
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import find_only_one, select_truthy_items


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
    
    node = find_only_one(dataset_aligned.nodes)
    
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
            
    # 3. plot behavior
    if plot_manual.lick:
        y_offset = yield unit_plot_lick(lick=node.data.lick, ax=ax, y_offset=y_offset, ratio=LICK_RATIO)

    # 4. plot locomotion
    if plot_manual.locomotion:      
        y_offset = yield unit_plot_locomotion(locomotion=node.data.locomotion, ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)
        y_offset = yield unit_plot_position(position=node.data.position, ax=ax, y_offset=y_offset, ratio=POSITION_RATIO)

    # 5. plot whisker
    if plot_manual.whisker:      
        y_offset = yield unit_plot_whisker(whisker=node.data.whisker, ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)

    # 6. plot pupil
    if plot_manual.pupil:             
        y_offset = yield unit_plot_pupil(pupil=node.data.pupil, ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)



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
            ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)

    # 5. plot whisker
    if plot_manual.whisker:    
        y_offset = yield unit_plot_whisker(
            whisker=select_truthy_items([node.data.whisker for node in dataset_synced]), 
            ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)

    # 6. plot pupil
    if plot_manual.pupil:    
        y_offset = yield unit_plot_pupil(
            pupil=select_truthy_items([node.data.pupil for node in dataset_synced]),
            ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)

  
def beam_view(
        ax: plt.Axes,
        datasets: DataSet,
        sync_events: Tuple[str],

        plot_manual: PlotManual = PlotManual(),
):
    """Beam view of all nodes in the dataset"""
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
        raise NotImplementedError
    elif plot_manual.potential:    
        for node_index, node in enumerate(dataset_synced):
            y_offset = yield unit_plot_potential(
                potential=node.data.potential, ax=ax, y_offset=y_offset, ratio=POTENTIAL_RATIO, 
                aspect=plot_manual.potential, yticks_flag=(node_index == 0),
                wc_flag=WC_CONVERT_FLAG(dataset_synced.nodes[0]))
