
from typing import Generator, Tuple
import matplotlib.pyplot as plt

from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.plotting_params import FLUORESCENCE_RATIO, LICK_RATIO, LOCOMOTION_RATIO, POSITION_RATIO, PUPIL_RATIO, TIMELINE_RATIO, WHISKER_RATIO
from kitchen.plotter.unit_plotter.unit_trace import unit_plot_lick, unit_plot_locomotion, unit_plot_position, unit_plot_pupil, unit_plot_single_cell_fluorescence, unit_plot_timeline, unit_plot_whisker
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import find_only_one, select_truthy_items


def flat_view(
        ax: plt.Axes,
        datasets: DataSet,
        
        locomotion_flag: bool = True,
        lick_flag: bool = True,
        pupil_flag: bool = True,
        tongue_flag: bool = True,
        whisker_flag: bool = True,
        fluorescence_flag: bool = True,
) -> Generator[float, float, None]:
    """Flat view of the only node in the dataset"""
    node = find_only_one(datasets.nodes)
    
    """main logic"""
    y_offset = 0

    # 1. plot_timeline    
    assert node.data.timeline is not None, "Cannot find timeline in node"
    y_offset = yield unit_plot_timeline(timeline=node.data.timeline, ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO)
    
    # 2. plot fluorescence
    if fluorescence_flag:
        assert node.data.fluorescence is not None, "Cannot find fluorescence in node"
        for cell_id in range(node.data.fluorescence.num_cell):
            y_offset = yield unit_plot_single_cell_fluorescence(
                fluorescence=node.data.fluorescence, ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO)

    # 3. plot behavior
    if lick_flag:
        assert node.data.lick is not None, "Cannot find lick in node"
        y_offset = yield unit_plot_lick(lick=node.data.lick, ax=ax, y_offset=y_offset, ratio=LICK_RATIO)

    # 4. plot locomotion
    if locomotion_flag:
        assert node.data.locomotion is not None, "Cannot find locomotion in node"        
        y_offset = yield unit_plot_locomotion(locomotion=node.data.locomotion, ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)
        assert node.data.position is not None, "Cannot find position in node"
        y_offset = yield unit_plot_position(position=node.data.position, ax=ax, y_offset=y_offset, ratio=POSITION_RATIO)

    # 5. plot whisker
    if whisker_flag:
        assert node.data.whisker is not None, "Cannot find whisker in node"        
        y_offset = yield unit_plot_whisker(whisker=node.data.whisker, ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)

    # 6. plot pupil
    if pupil_flag:
        assert node.data.pupil is not None, "Cannot find pupil in node"                
        y_offset = yield unit_plot_pupil(pupil=node.data.pupil, ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)



def stack_view(
        ax: plt.Axes,
        dataset_presync: DataSet,
        sync_events: Tuple[str],

        locomotion_flag: bool = True,
        lick_flag: bool = True,
        pupil_flag: bool = True,
        tongue_flag: bool = True,
        whisker_flag: bool = True,
        fluorescence_flag: bool = True,
) -> Generator[float, float, None]:
    """Stack view of all nodes in the dataset"""
    try:
        dataset = sync_nodes(dataset_presync, sync_events)
    except Exception as e:
        raise ValueError(f"Cannot sync nodes: {e}")
    
    """main logic"""
    y_offset = 0

    # 1. plot_timeline    
    y_offset = yield unit_plot_timeline(
        timeline=select_truthy_items([node.data.timeline for node in dataset]), ax=ax, y_offset=y_offset, ratio=TIMELINE_RATIO) 

    # 2. plot fluorescence
    if fluorescence_flag:
        valid_fluorescence = select_truthy_items([node.data.fluorescence for node in dataset])    
        for cell_id in range(valid_fluorescence[0].num_cell):
            y_offset = yield unit_plot_single_cell_fluorescence(
                fluorescence=[fluorescence.extract_cell(cell_id) for fluorescence in valid_fluorescence], 
                ax=ax, y_offset=y_offset, ratio=FLUORESCENCE_RATIO)

    # 3. plot behavior  
    if lick_flag:
        y_offset = yield unit_plot_lick(
            lick=select_truthy_items([node.data.lick for node in dataset]), ax=ax, y_offset=y_offset, ratio=LICK_RATIO)
        
    # 4. plot locomotion
    if locomotion_flag:
        y_offset = yield unit_plot_locomotion(
            locomotion=select_truthy_items([node.data.locomotion for node in dataset]), ax=ax, y_offset=y_offset, ratio=LOCOMOTION_RATIO)

    # 5. plot whisker
    if whisker_flag:
        y_offset = yield unit_plot_whisker(
            whisker=select_truthy_items([node.data.whisker for node in dataset]), ax=ax, y_offset=y_offset, ratio=WHISKER_RATIO)

    # 6. plot pupil
    if pupil_flag:
        y_offset = yield unit_plot_pupil(
            pupil=select_truthy_items([node.data.pupil for node in dataset]), ax=ax, y_offset=y_offset, ratio=PUPIL_RATIO)

  
    
