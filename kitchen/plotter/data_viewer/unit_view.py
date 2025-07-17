import warnings
import matplotlib.pyplot as plt
import numpy as np

from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, LOCOMOTION_COLOR, POSITION_COLOR, LICK_COLOR, PUPIL_COLOR, WHISKER_COLOR
from kitchen.plotter.decorators import BBOX
from kitchen.plotter.plotting_params import LOCOMOTION_BIN_SIZE, TIME_TICK_DURATION
from kitchen.plotter.style_dicts import FLUORESCENCE_TRACE_STYLE, LOCOMOTION_TRACE_STYLE, POSITION_SCATTER_STYLE, LICK_VLINES_STYLE, PUPIL_TRACE_STYLE, TIMELINE_SCATTER_STYLE, WHISKER_TRACE_STYLE
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.structure.neural_data_structure import Events, Fluorescence, TimeSeries, Timeline


def unit_assertion(data, data_type):
    assert data is not None, f"Expected {data_type}, got None"
    assert isinstance(data, data_type), f"Expected {data_type}, got {type(data)}"    

def unit_plot_locomotion(locomotion: Events, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot locomotion"""
    unit_assertion(locomotion, Events)

    # plot locomotion rate
    plotting_locomotion = locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE)
    bbox = BBOX(np.nanmin(plotting_locomotion.v) * ratio, np.nanmax(plotting_locomotion.v) * ratio)        
    zero_y = y_offset - bbox.ymin
    ax.plot(plotting_locomotion.t, plotting_locomotion.v * ratio + zero_y, **LOCOMOTION_TRACE_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(zero_y, "Locomotion", LOCOMOTION_COLOR), 
                        TICK_PAIR(zero_y + 2 * ratio, "2 cm/s", LOCOMOTION_COLOR)])
    return bbox


def unit_plot_position(position: Events, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot position"""
    unit_assertion(position, Events)

    # plot position
    bbox = BBOX(0, ratio)        
    ax.scatter(position.t, position.v * ratio + y_offset, **POSITION_SCATTER_STYLE)        

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "0°", POSITION_COLOR), 
                        TICK_PAIR(y_offset + 0.5 * ratio, "180°", POSITION_COLOR),
                        TICK_PAIR(y_offset + 1 * ratio, "360°", POSITION_COLOR)])
    return bbox


def unit_plot_lick(lick: Events, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot lick"""    
    unit_assertion(lick, Events)    

    # plot lick
    bbox = BBOX(0, 5*ratio)        
    ax.vlines(x=lick.t, ymin=y_offset, ymax=y_offset + 5*ratio, **LICK_VLINES_STYLE)
    
    # add y ticks
    add_new_yticks(ax, TICK_PAIR(y_offset + 2.5 * ratio, "Lick", LICK_COLOR), add_ref_lines=False)
    return bbox


def unit_plot_pupil(pupil: TimeSeries, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot pupil"""    
    unit_assertion(pupil, TimeSeries)    

    # plot pupil
    bbox = BBOX(0, ratio)        
    ax.plot(pupil.t, pupil.v * ratio + y_offset, **PUPIL_TRACE_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Pupil", PUPIL_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Pupil", PUPIL_COLOR)])
    return bbox


def unit_plot_whisker(whisker: TimeSeries, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot whisker"""    
    unit_assertion(whisker, TimeSeries)    

    # plot whisker
    bbox = BBOX(0, ratio)        
    ax.plot(whisker.t, whisker.v * ratio + y_offset, **WHISKER_TRACE_STYLE)        

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Whisker", WHISKER_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Whisker", WHISKER_COLOR)])
    return bbox


def unit_plot_timeline(timeline: Timeline, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot timeline"""
    unit_assertion(timeline, Timeline)    

    # plot timeline
    bbox = BBOX(0, ratio)        
    for event_time, event_type in zip(timeline.t, timeline.v):
        if event_type not in TIMELINE_SCATTER_STYLE:
            continue
        ax.scatter(event_time, y_offset + 0.5 * ratio , **TIMELINE_SCATTER_STYLE[event_type])

    # set x ticks
    try:
        task_start, task_end = timeline.task_time()
        ax.set_xticks(np.arange(task_start, task_end, TIME_TICK_DURATION, dtype=int), 
                      np.arange(0, task_end - task_start, TIME_TICK_DURATION, dtype=int))
    except Exception as e:
        warnings.warn(f"Cannot set start to end x ticks for timeline: {e}")
    return bbox


def unit_plot_single_cell_fluorescence(fluorescence: Fluorescence, ax: plt.Axes, y_offset: float, cell_id: int, ratio: float = 1.0) -> BBOX:
    """plot a single cell"""
    unit_assertion(fluorescence, Fluorescence)    

    # plot fluorescence
    cell_trace = fluorescence.detrend_f.v[cell_id]
    bbox = BBOX(np.nanmin(cell_trace) * ratio, np.nanmax(cell_trace) * ratio)
    zero_y = y_offset - bbox.ymin
    ax.plot(fluorescence.detrend_f.t, cell_trace * ratio + zero_y, **FLUORESCENCE_TRACE_STYLE)
    
    # add y ticks
    add_new_yticks(ax, TICK_PAIR(zero_y, f"Cell {fluorescence.cell_idx[cell_id]}", FLUORESCENCE_COLOR))          
    if cell_id == 0:
        add_new_yticks(ax, TICK_PAIR(zero_y + 1 * ratio, f"1 {DF_F0_SIGN}", FLUORESCENCE_COLOR))  
    return bbox
    
