from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import numpy as np

from kitchen.operator.grouping import GROUP_TUPLE, grouping_events_rate, grouping_timeseries
from kitchen.plotter.utils.alpha_calculator import calibrate_alpha, ind_alpha
from kitchen.plotter.utils.oreo_plot import oreo_plot
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, GRAND_COLOR_SCHEME, LOCOMOTION_COLOR, POSITION_COLOR, LICK_COLOR, PUPIL_COLOR, WHISKER_COLOR
from kitchen.plotter.plotting_params import LICK_BIN_SIZE, LOCOMOTION_BIN_SIZE, TIME_TICK_DURATION
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE, FLUORESCENCE_TRACE_STYLE, LICK_TRACE_STYLE, LOCOMOTION_TRACE_STYLE, POSITION_SCATTER_STYLE, LICK_VLINES_STYLE, PUPIL_TRACE_STYLE, TIMELINE_SCATTER_STYLE, VLINE_STYLE, VSPAN_STYLE, WHISKER_TRACE_STYLE
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.settings.plotting import PLOTTING_OVERLAP_HARSH_MODE
from kitchen.structure.neural_data_structure import Events, Fluorescence, TimeSeries, Timeline


def sanity_check(data) -> bool:
    """Check if data is not None"""
    if isinstance(data, list):
        return all(d is not None for d in data) if PLOTTING_OVERLAP_HARSH_MODE else any(d is not None for d in data)
    return data is not None


def unit_plot_locomotion(locomotion: None | Events | list[Events], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot locomotion"""
    if not sanity_check(locomotion):
        return 0
    assert locomotion is not None, "Sanity check failed"

    if isinstance(locomotion, Events):
        # plot single locomotion rate
        plotting_locomotion = locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE)
        y_height = np.nanmax(plotting_locomotion.v) * ratio    
        ax.plot(plotting_locomotion.t, plotting_locomotion.v * ratio + y_offset, **LOCOMOTION_TRACE_STYLE)
    else:
        # plot multiple locomotion rates
        group_locomotion = grouping_events_rate(locomotion, bin_size=LOCOMOTION_BIN_SIZE)
        y_height = np.nanmax(group_locomotion.mean) * ratio
        oreo_plot(ax, group_locomotion, y_offset, ratio, LOCOMOTION_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Locomotion", LOCOMOTION_COLOR), 
                        TICK_PAIR(y_offset + 2 * ratio, "2 cm/s", LOCOMOTION_COLOR)])
    return max(y_height, 2*ratio)


def unit_plot_position(position: None | Events, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot position"""
    if not sanity_check(position):
        return 0
    assert position is not None, "Sanity check failed"

    # plot position      
    ax.scatter(position.t, position.v * ratio + y_offset, **POSITION_SCATTER_STYLE)        

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "0°", POSITION_COLOR), 
                        TICK_PAIR(y_offset + 0.5 * ratio, "180°", POSITION_COLOR),
                        TICK_PAIR(y_offset + 1 * ratio, "360°", POSITION_COLOR)])
    return 1*ratio


def unit_plot_lick(lick: None | Events | list[Events], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot lick"""    
    if not sanity_check(lick):
        return 0
    assert lick is not None, "Sanity check failed"

    if isinstance(lick, Events):
        # plot single lick
        y_height = ratio
        ax.vlines(x=lick.t, ymin=y_offset, ymax=y_offset + ratio, **LICK_VLINES_STYLE)        
        add_new_yticks(ax, TICK_PAIR(y_offset + 0.5 * ratio, "Lick", LICK_COLOR), add_ref_lines=False)
    else:
        # plot multiple licks
        group_lick = grouping_events_rate(lick, bin_size=LICK_BIN_SIZE)
        y_height = max(np.nanmax(group_lick.mean) * ratio, 5*ratio)
        oreo_plot(ax, group_lick, y_offset, ratio, LICK_TRACE_STYLE, FILL_BETWEEN_STYLE)
        add_new_yticks(ax, [TICK_PAIR(y_offset, "Lick", LICK_COLOR),
                            TICK_PAIR(y_offset + 5 * ratio, "5 Hz", LICK_COLOR)])
    return y_height


def unit_plot_pupil(pupil: None | TimeSeries | list[TimeSeries], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot pupil"""    
    if not sanity_check(pupil):
        return 0
    assert pupil is not None, "Sanity check failed"
    
    if isinstance(pupil, TimeSeries):
        # plot single pupil
        ax.plot(pupil.t, pupil.v * ratio + y_offset, **PUPIL_TRACE_STYLE)
    else:
        # plot multiple pupils
        group_pupil = grouping_timeseries(pupil)
        oreo_plot(ax, group_pupil, y_offset, ratio, PUPIL_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Pupil", PUPIL_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Pupil", PUPIL_COLOR)])
    return ratio


def unit_plot_whisker(whisker: None | TimeSeries | list[TimeSeries], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot whisker"""    
    if not sanity_check(whisker):    
        return 0    
    assert whisker is not None, "Sanity check failed"

    if isinstance(whisker, TimeSeries):
        # plot single whisker
        ax.plot(whisker.t, whisker.v * ratio + y_offset, **WHISKER_TRACE_STYLE)        
    else:
        # plot multiple whiskers
        group_whisker = grouping_timeseries(whisker)
        oreo_plot(ax, group_whisker, y_offset, ratio, WHISKER_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Whisker", WHISKER_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Whisker", WHISKER_COLOR)])
    return ratio


def unit_plot_timeline(timeline: None | Timeline | list[Timeline], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot timeline"""
    if not sanity_check(timeline):
        return 0
    assert timeline is not None, "Sanity check failed"
    
    if isinstance(timeline, Timeline):
        # plot single timeline
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
        return ratio
    
    # plot multiple timelines
    # plot all the markers
    all_event = defaultdict(list)
    for one_timeline in timeline:
        for event_time, event_type in zip(one_timeline.t, one_timeline.v):
            if event_type in TIMELINE_SCATTER_STYLE:
                all_event[event_type].append(event_time)
    
    for event_type, event_times in all_event.items():
        num_events = len(event_times)
        ax.scatter(event_times, [y_offset + 0.5 * ratio] * num_events,
                **(TIMELINE_SCATTER_STYLE[event_type] | 
                    {"alpha": ind_alpha(TIMELINE_SCATTER_STYLE[event_type]["alpha"], num_events)}))
    
    # plot vspan or vline
    for one_timeline in timeline:
        for event_time, event_type in zip(one_timeline.t, one_timeline.v):
            if "On" not in event_type or event_type not in GRAND_COLOR_SCHEME:
                continue
            end_event = event_type.replace("On", "Off")
            if end_event not in one_timeline.v:
                continue
            end_time = one_timeline.filter(end_event).t[0]
            if end_time - event_time >= 0.1:
                ax.axvspan(event_time, end_time, color=GRAND_COLOR_SCHEME[event_type],
                            **calibrate_alpha(VSPAN_STYLE, len(all_event[event_type])))
            else:
                ax.axvline(event_time, color=GRAND_COLOR_SCHEME[event_type], 
                            **calibrate_alpha(VLINE_STYLE, len(all_event[event_type])))
    return ratio


def unit_plot_single_cell_fluorescence(fluorescence: None | Fluorescence | list[Fluorescence], ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot a single cell"""
    if not sanity_check(fluorescence):
        return 0
    assert fluorescence is not None, "Sanity check failed"

    if isinstance(fluorescence, Fluorescence):
        # plot single cell fluorescence
        assert fluorescence.num_cell == 1, f"Expected 1 cell, but got {fluorescence.num_cell}"
        cell_trace = fluorescence.detrend_f.v[0]
        ax.plot(fluorescence.detrend_f.t, cell_trace * ratio + y_offset, **FLUORESCENCE_TRACE_STYLE)      

        # add y ticks  
        add_new_yticks(ax, TICK_PAIR(y_offset, f"Cell {fluorescence.cell_idx[0]}", FLUORESCENCE_COLOR))      
        add_new_yticks(ax, TICK_PAIR(y_offset + 1 * ratio,
                                     f"1 {DF_F0_SIGN}" if np.all(fluorescence.cell_order == 0) else "", FLUORESCENCE_COLOR))
        return np.nanmax(cell_trace) * ratio

    # plot multiple cell fluorescence
    group_fluorescence = grouping_timeseries([fluorescence.df_f0 for fluorescence in fluorescence])
    squeezed_fluorescence = GROUP_TUPLE(t=group_fluorescence.t, 
                                        mean=np.squeeze(group_fluorescence.mean, axis=0), 
                                        variance=np.nanstd(group_fluorescence.variance, axis=0))
    oreo_plot(ax, squeezed_fluorescence, y_offset, ratio, FLUORESCENCE_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    example_fluorescence = fluorescence[0]
    add_new_yticks(ax, TICK_PAIR(y_offset, f"Cell {example_fluorescence.cell_idx[0]}", FLUORESCENCE_COLOR))      
    add_new_yticks(ax, TICK_PAIR(y_offset + 1 * ratio, f"1 {DF_F0_SIGN}" if np.all(example_fluorescence.cell_order == 0) else "", FLUORESCENCE_COLOR))    
    return np.nanmax(squeezed_fluorescence.mean) * ratio
        
