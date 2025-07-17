from typing import List
import matplotlib.pyplot as plt
import numpy as np

from kitchen.operator.grouping import GROUP_TUPLE, grouping_events_rate, grouping_timeseries
from kitchen.plotter.utils.group_oreo import oreo_plot
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, GRAND_COLOR_SCHEME, LOCOMOTION_COLOR, LICK_COLOR, PUPIL_COLOR, WHISKER_COLOR
from kitchen.plotter.decorators import BBOX
from kitchen.plotter.plotting_params import LICK_BIN_SIZE, LOCOMOTION_BIN_SIZE
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE, FLUORESCENCE_TRACE_STYLE, LICK_TRACE_STYLE, LOCOMOTION_TRACE_STYLE, PUPIL_TRACE_STYLE, TIMELINE_SCATTER_STYLE, VLINE_STYLE, VSPAN_STYLE, WHISKER_TRACE_STYLE
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.structure.neural_data_structure import Events, TimeSeries, Timeline


def group_assertion(data_list, data_type):
    for data in data_list:
        assert data is not None, f"Expected {data_type}, got None"
        assert isinstance(data, data_type), f"Expected {data_type}, got {type(data)}"    

def group_plot_locomotion(locomotions: List[Events], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot locomotion"""
    group_assertion(locomotions, Events)

    # plot locomotion rate
    group_locomotion = grouping_events_rate(locomotions, LOCOMOTION_BIN_SIZE)
    # bbox = BBOX(np.nanmin(group_locomotion.mean) * ratio, np.nanmax(group_locomotion.mean) * ratio)   
    bbox = BBOX(0, np.nanmax(group_locomotion.mean) * ratio)        
    zero_y = y_offset - bbox.ymin
    oreo_plot(ax, group_locomotion, zero_y, ratio, LOCOMOTION_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(zero_y, "Locomotion", LOCOMOTION_COLOR), 
                        TICK_PAIR(zero_y + 2 * ratio, "2 cm/s", LOCOMOTION_COLOR)])
    return bbox


def group_plot_lick(licks: List[Events], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot lick"""    
    group_assertion(licks, Events)    

    # plot lick
    group_lick = grouping_events_rate(licks, LICK_BIN_SIZE)
    bbox = BBOX(0, np.nanmax(group_lick.mean) * ratio)        
    oreo_plot(ax, group_lick, y_offset, ratio, LICK_TRACE_STYLE, FILL_BETWEEN_STYLE)
    
    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Lick", LICK_COLOR),
                        TICK_PAIR(y_offset + 5 * ratio, "5 Hz", LICK_COLOR)])
    return bbox


def group_plot_pupil(pupils: list[TimeSeries], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot pupil"""    
    group_assertion(pupils, TimeSeries)    

    # plot pupil
    group_pupil = grouping_timeseries(pupils)
    bbox = BBOX(0, ratio)
    oreo_plot(ax, group_pupil, y_offset, ratio, PUPIL_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Pupil", PUPIL_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Pupil", PUPIL_COLOR)])
    return bbox


def group_plot_whisker(whiskers: list[TimeSeries], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot whisker"""    
    group_assertion(whiskers, TimeSeries)    

    # plot whisker
    group_whisker = grouping_timeseries(whiskers)
    bbox = BBOX(0, ratio)
    oreo_plot(ax, group_whisker, y_offset, ratio, WHISKER_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Whisker", WHISKER_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Whisker", WHISKER_COLOR)])
    return bbox


def group_plot_timeline(timelines: list[Timeline], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> BBOX:
    """plot timeline"""
    group_assertion(timelines, Timeline)    

    # plot timeline
    bbox = BBOX(0, ratio)   
    for timeline in timelines:     
        for event_time, event_type in zip(timeline.t, timeline.v):
            if event_type not in TIMELINE_SCATTER_STYLE:
                continue
            ax.scatter(event_time, y_offset + 0.5 * ratio , **TIMELINE_SCATTER_STYLE[event_type])

            # plot vspan or vline
            if "On" not in event_type:
                continue
            end_event = event_type.replace("On", "Off")
            if end_event not in timeline.v:
                continue
            end_time = timeline.filter(end_event).t[0]
            if end_time - event_time >= 0.1:
                ax.axvspan(event_time, end_time, color=GRAND_COLOR_SCHEME[event_type], **VSPAN_STYLE)
            else:
                ax.axvline(event_time, color=GRAND_COLOR_SCHEME[event_type], **VLINE_STYLE)
    return bbox


def group_plot_single_cell_fluorescence(group_fluorescence: GROUP_TUPLE, ax: plt.Axes, y_offset: float, cell_id: int, ratio: float = 1.0) -> BBOX:
    """plot a single cell"""
    # plot fluorescence    
    group_cell_fluorescence = GROUP_TUPLE(t=group_fluorescence.t, mean=group_fluorescence.mean[cell_id], variance=group_fluorescence.variance[cell_id])
    # bbox = BBOX(np.nanmin(group_cell_fluorescence.mean) * ratio, np.nanmax(group_cell_fluorescence.mean) * ratio)
    bbox = BBOX(0, np.nanmax(group_cell_fluorescence.mean) * ratio)
    zero_y = y_offset - bbox.ymin
    oreo_plot(ax, group_cell_fluorescence, zero_y, ratio, FLUORESCENCE_TRACE_STYLE, FILL_BETWEEN_STYLE)
    
    # add y ticks
    add_new_yticks(ax, TICK_PAIR(zero_y, f"Cell {cell_id}", FLUORESCENCE_COLOR))          
    if cell_id == 0:
        add_new_yticks(ax, TICK_PAIR(zero_y + 1 * ratio, f"1 {DF_F0_SIGN}", FLUORESCENCE_COLOR))  
    return bbox
    
