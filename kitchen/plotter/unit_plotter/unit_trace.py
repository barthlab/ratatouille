from collections import defaultdict
import random
import logging
from typing import Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from kitchen.operator.grouping import grouping_events_rate, grouping_timeseries
from kitchen.plotter.unit_plotter.unit_yticks import yticks_combo
from kitchen.plotter.utils.alpha_calculator import calibrate_alpha, ind_alpha
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.fluorescence import DF_F0_SIGN, Z_SCORE_SIGN
from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, GRAND_COLOR_SCHEME, LICK_COLOR, POTENTIAL_COLOR
from kitchen.plotter.plotting_params import LICK_BIN_SIZE, LOCOMOTION_BIN_SIZE, RAW_FLUORESCENCE_RATIO, TIME_TICK_DURATION, WC_POTENTIAL_RATIO
from kitchen.plotter.style_dicts import DEEMPHASIZED_POTENTIAL_ADD_STYLE, EMPHASIZED_POTENTIAL_ADD_STYLE, FILL_BETWEEN_STYLE, FLUORESCENCE_TRACE_STYLE, INDIVIDUAL_FLUORESCENCE_TRACE_STYLE, LICK_TRACE_STYLE, LOCOMOTION_TRACE_STYLE, MAX_OVERLAP_ALPHA_NUM_DUE_TO_MATPLOTLLIB_BUG, POSITION_SCATTER_STYLE, LICK_VLINES_STYLE, POTENTIAL_TRACE_STYLE, PUPIL_TRACE_STYLE, SPIKE_POTENTIAL_TRACE_STYLE, TIMELINE_SCATTER_STYLE, VLINE_STYLE, VSPAN_STYLE, WHISKER_TRACE_STYLE
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.settings.plotting import PLOTTING_OVERLAP_HARSH_MODE
from kitchen.settings.potential import SPIKE_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.structure.neural_data_structure import Events, Fluorescence, Potential, TimeSeries, Timeline
from kitchen.utils.sequence_kit import group_by


logger = logging.getLogger(__name__)

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
        if len(locomotion) > 0:
            plotting_locomotion = locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE)
            y_height = np.nanmax(plotting_locomotion.v) * ratio    
            ax.plot(plotting_locomotion.t, plotting_locomotion.v * ratio + y_offset, **LOCOMOTION_TRACE_STYLE)
        else:
            y_height = 0        
    else:
        # plot multiple locomotion rates
        group_locomotion = grouping_events_rate(locomotion, bin_size=LOCOMOTION_BIN_SIZE)
        y_height = np.nanmax(group_locomotion.mean) * ratio
        oreo_plot(ax, group_locomotion, y_offset, ratio, LOCOMOTION_TRACE_STYLE, FILL_BETWEEN_STYLE)

    # add y ticks
    yticks_combo("locomotion", ax, y_offset, ratio)
    return max(y_height, 2*ratio)


def unit_plot_position(position: None | Events, ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot position"""
    if not sanity_check(position):
        return 0
    assert position is not None, "Sanity check failed"

    # plot position      
    ax.scatter(position.t, position.v * ratio + y_offset, **POSITION_SCATTER_STYLE)        

    # add y ticks
    yticks_combo("position", ax, y_offset, ratio)
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
        if len(group_lick) == 0:
            return 0
        y_height = max(np.nanmax(group_lick.mean) * ratio, 5*ratio)
        oreo_plot(ax, group_lick, y_offset, ratio, LICK_TRACE_STYLE, FILL_BETWEEN_STYLE)
        yticks_combo("lick", ax, y_offset, ratio)
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
    yticks_combo("pupil", ax, y_offset, ratio)
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
    yticks_combo("whisker", ax, y_offset, ratio)
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
            ax.axvline(x=event_time, color=GRAND_COLOR_SCHEME[event_type], **VLINE_STYLE)

        # set x ticks
        try:
            task_start, task_end = timeline.task_time()
            ax.set_xticks(np.arange(task_start, task_end, TIME_TICK_DURATION, dtype=int), 
                        np.arange(0, task_end - task_start, TIME_TICK_DURATION, dtype=int))
        except Exception as e:
            logger.debug(f"Cannot set start to end x ticks for timeline: {e}")
        return ratio
    
    # plot multiple timelines
    # decrease timeline number due to matplotlib's bug
    if len(timeline) > MAX_OVERLAP_ALPHA_NUM_DUE_TO_MATPLOTLLIB_BUG:
        timeline = random.sample(timeline, k=MAX_OVERLAP_ALPHA_NUM_DUE_TO_MATPLOTLLIB_BUG)

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
            if event_type not in GRAND_COLOR_SCHEME:
                continue
            if ("On" not in event_type) or (event_type.replace("On", "Off") not in one_timeline.v):
                ax.axvline(event_time, color=GRAND_COLOR_SCHEME[event_type], 
                            **calibrate_alpha(VLINE_STYLE, len(all_event[event_type])))
                continue

            end_event = event_type.replace("On", "Off")
            end_time = one_timeline.filter(end_event).t[0]
            if end_time - event_time >= 0.09:
                ax.axvspan(event_time, end_time, color=GRAND_COLOR_SCHEME[event_type],
                            **calibrate_alpha(VSPAN_STYLE, len(all_event[event_type])))
            else:
                ax.axvline(event_time, color=GRAND_COLOR_SCHEME[event_type], 
                            **calibrate_alpha(VLINE_STYLE, len(all_event[event_type])))
    return ratio


def unit_plot_single_cell_fluorescence(fluorescence: None | Fluorescence | list[Fluorescence], 
                                       ax: plt.Axes, y_offset: float, ratio: float = 1.0,
                                       cell_id_flag: bool = True, individual_trace_flag: bool = False) -> float:
    """plot a single cell fluorescence"""
    if not sanity_check(fluorescence):
        return 0
    assert fluorescence is not None, "Sanity check failed"

    if isinstance(fluorescence, Fluorescence):
        ratio *= RAW_FLUORESCENCE_RATIO
        
        # plot single cell fluorescence
        assert fluorescence.num_cell == 1, f"Expected 1 cell, but got {fluorescence.num_cell}"
        cell_trace = fluorescence.z_score.v[0]
        ax.plot(fluorescence.z_score.t, cell_trace * ratio + y_offset, **FLUORESCENCE_TRACE_STYLE)      

        # add y ticks  
        add_new_yticks(ax, TICK_PAIR(
            y_offset, f"Cell {fluorescence.cell_idx[0]}" if cell_id_flag else "Cell", FLUORESCENCE_COLOR))      
        add_new_yticks(ax, TICK_PAIR(
            y_offset + 1 * ratio,
            f"1 {Z_SCORE_SIGN}" if (np.all(fluorescence.cell_order == 0) or (not cell_id_flag)) else "", FLUORESCENCE_COLOR))
        return max(np.nanmax(cell_trace) * ratio, 1*ratio)

    # plot multiple cell fluorescence
    group_fluorescence = grouping_timeseries([fluorescence.df_f0 for fluorescence in fluorescence]).squeeze(0)
    if len(group_fluorescence) == 0:
        return 0
    oreo_plot(ax, group_fluorescence, y_offset, ratio, FLUORESCENCE_TRACE_STYLE, FILL_BETWEEN_STYLE)
    if individual_trace_flag:
        for individual_fluorescence in group_fluorescence.raw:
            ax.plot(group_fluorescence.t, individual_fluorescence * ratio + y_offset, 
                    **calibrate_alpha(INDIVIDUAL_FLUORESCENCE_TRACE_STYLE, group_fluorescence.data_num))

    # add y ticks
    example_fluorescence = fluorescence[0]
    add_new_yticks(ax, TICK_PAIR(
        y_offset, 
        f"Cell {example_fluorescence.cell_idx[0]}" if cell_id_flag else "Cell", FLUORESCENCE_COLOR))      
    add_new_yticks(ax, TICK_PAIR(
        y_offset + 1 * ratio, 
        f"1 {DF_F0_SIGN}" if (np.all(example_fluorescence.cell_order == 0) or (not cell_id_flag)) else "", FLUORESCENCE_COLOR))    
    return max(np.nanmax(group_fluorescence.mean) * ratio, 1*ratio)
        


def unit_plot_potential(potential: None | Potential | list[Potential], 
                        ax: plt.Axes, y_offset: float, ratio: float = 1.0,
                        spike_mark: bool = True, aspect: Optional[Any] = None, 
                        yticks_flag: bool = True, wc_flag: bool = False,
                        emphasize_rule: str = "median", spike_num_warning_threshold: int = 1000) -> float:
    """plot a single cell potential"""
    if not sanity_check(potential):
        return 0
    assert potential is not None, "Sanity check failed"

    # adjust ratio for whole cell potential and add y ticks
    example_potential = potential[0] if isinstance(potential, list) else potential
    if wc_flag:  # adjust for whole cell potential
        ratio *= WC_POTENTIAL_RATIO
        ytick_template = "potential_wc"
    else:
        ytick_template = "potential_jux"
    
    if yticks_flag:
        yticks_combo(ytick_template, ax, y_offset, ratio)
    else:
        add_new_yticks(ax, TICK_PAIR(y_offset, "", POTENTIAL_COLOR)) 
    
    # warning for large spike plotting
    if spike_mark and len(example_potential.spikes) > spike_num_warning_threshold:
        logger.debug(f"Large number of spikes ({len(example_potential.spikes)}) to plot. This may take a long time.")

    if isinstance(potential, Potential):
        potential_timeseries = potential.aspect(aspect)
        
        # plot single potential
        ax.plot(potential_timeseries.t, potential_timeseries.v * ratio + y_offset, **POTENTIAL_TRACE_STYLE)
        
        # plot all the spikes
        if spike_mark:            
            spike_type_dict = group_by(list(zip(potential.spikes.v, potential.spikes.t)), lambda x: x[0])
            for spike_type, all_spike_fragments in spike_type_dict.items():
                all_spike_timeseries = potential_timeseries.batch_segment(
                    [spike_time for _, spike_time in all_spike_fragments], SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, _auto_align=False)
                
                nan_separator = np.array([np.nan])
                # Interleave NaN separators between the actual data arrays
                # The result will be [t1, NaN, t2, NaN, t3, ...]
                t_concatenated = np.concatenate([arr for ts in all_spike_timeseries for arr in (ts.t, nan_separator)])
                v_concatenated = np.concatenate([arr for ts in all_spike_timeseries for arr in (ts.v, nan_separator)])
                
                ax.plot(t_concatenated, v_concatenated * ratio + y_offset, 
                        **POTENTIAL_TRACE_STYLE | SPIKE_POTENTIAL_TRACE_STYLE[spike_type])
                
        # calculate y height
        y_height = np.nanmax(potential.aspect(aspect).v) * ratio
    else:
        # plot multiple potential
        # determine the emphasize index
        if emphasize_rule == "median":
            emphasize_index = int(len(potential) / 2)
        elif emphasize_rule == "first":
            emphasize_index = 0
        elif emphasize_rule == "last":
            emphasize_index = -1
        elif emphasize_rule == "random":
            emphasize_index = random.randint(0, len(potential) - 1)
        else:
            raise ValueError(f"Unknown emphasize rule: {emphasize_rule}")
        # plot all the potential
        for potential_index, one_potential in enumerate(potential):
            if potential_index == emphasize_index:
                additional_style = EMPHASIZED_POTENTIAL_ADD_STYLE.copy()
            else:
                additional_style = calibrate_alpha(DEEMPHASIZED_POTENTIAL_ADD_STYLE.copy(), len(potential))
            one_potential_timeseries = one_potential.aspect(aspect)

            ax.plot(one_potential_timeseries.t, one_potential_timeseries.v * ratio + y_offset, 
                    **(POTENTIAL_TRACE_STYLE | additional_style))
            
            # plot all the spikes
            if spike_mark:
                spike_type_dict = group_by(list(zip(one_potential.spikes.v, one_potential.spikes.t)), lambda x: x[0])
                for spike_type, all_spike_fragments in spike_type_dict.items():
                    all_spike_timeseries = one_potential_timeseries.batch_segment(
                        [spike_time for _, spike_time in all_spike_fragments], SPIKE_RANGE_RELATIVE_TO_ALIGNMENT, _auto_align=False)
                    nan_separator = np.array([np.nan])
                    t_concatenated = np.concatenate([arr for ts in all_spike_timeseries for arr in (ts.t, nan_separator)])
                    v_concatenated = np.concatenate([arr for ts in all_spike_timeseries for arr in (ts.v, nan_separator)])

                    ax.plot(t_concatenated, v_concatenated * ratio + y_offset, 
                            **POTENTIAL_TRACE_STYLE | SPIKE_POTENTIAL_TRACE_STYLE[spike_type] | additional_style)      
                    
        # calculate y height
        y_height = np.nanmax(np.concatenate([one_potential.aspect(aspect).v for one_potential in potential])) * ratio
    return max(y_height, 1*ratio)




def unit_plot_potential_conv(potential: None | Potential | list[Potential], 
                             ax: plt.Axes, y_offset: float, ratio: float = 1.0,
                             spike_mark: bool = True, individual_trace_flag: bool = False) -> float:
    """plot a single cell convolved potential"""
    if not sanity_check(potential):
        return 0
    assert potential is not None, "Sanity check failed"

    if isinstance(potential, Potential):        
        ax.plot(potential.aspect("conv").t, potential.aspect("conv").v * ratio + y_offset, **FLUORESCENCE_TRACE_STYLE)      

         # plot all the spikes
        if spike_mark:            
            spike_type_dict = group_by(list(zip(potential.spikes.v, potential.spikes.t)), lambda x: x[0])
            for spike_type, all_spike_fragments in spike_type_dict.items():
                ax.eventplot([spike_time for _, spike_time in all_spike_fragments], 
                            lineoffsets=y_offset-0.5, linelengths=0.5,
                            **POTENTIAL_TRACE_STYLE | SPIKE_POTENTIAL_TRACE_STYLE[spike_type])
                
        # add y ticks  
        add_new_yticks(ax, TICK_PAIR(
            y_offset, "conv", FLUORESCENCE_COLOR))      
        add_new_yticks(ax, TICK_PAIR(
            y_offset + 1 * ratio, f"1 {DF_F0_SIGN}", FLUORESCENCE_COLOR))
        return max(np.nanmax(potential.aspect("conv").v) * ratio, 1*ratio)

    # plot multiple cell fluorescence
    group_fluorescence = grouping_timeseries([one_potential.aspect("conv") for one_potential in potential])
    if len(group_fluorescence) == 0:
        return 0
    oreo_plot(ax, group_fluorescence, y_offset, ratio, FLUORESCENCE_TRACE_STYLE, FILL_BETWEEN_STYLE)
    if individual_trace_flag:
        for individual_fluorescence, one_potential in zip(group_fluorescence.raw, potential):
            ax.plot(group_fluorescence.t, individual_fluorescence * ratio + y_offset, 
                    **calibrate_alpha(INDIVIDUAL_FLUORESCENCE_TRACE_STYLE, group_fluorescence.data_num))

            if spike_mark:            
                spike_type_dict = group_by(list(zip(one_potential.spikes.v, one_potential.spikes.t)), lambda x: x[0])
                for spike_type, all_spike_fragments in spike_type_dict.items():
                    ax.eventplot([spike_time for _, spike_time in all_spike_fragments], 
                                lineoffsets=y_offset-0.5, linelengths=0.5,
                                **calibrate_alpha(POTENTIAL_TRACE_STYLE | SPIKE_POTENTIAL_TRACE_STYLE[spike_type], group_fluorescence.data_num))
    # add y ticks
    add_new_yticks(ax, TICK_PAIR(
        y_offset, "conv", FLUORESCENCE_COLOR))      
    add_new_yticks(ax, TICK_PAIR(
        y_offset + 1 * ratio, f"1 {DF_F0_SIGN}", FLUORESCENCE_COLOR)) 
    return max(np.nanmax(group_fluorescence.mean) * ratio, 1*ratio)