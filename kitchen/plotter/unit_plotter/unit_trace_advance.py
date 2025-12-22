from collections import namedtuple
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from kitchen.operator.grouping import grouping_events_rate, grouping_timeseries
from kitchen.plotter import color_scheme
from kitchen.plotter.unit_plotter.unit_trace import sanity_check
from kitchen.plotter.unit_plotter.unit_yticks import yticks_combo
from kitchen.plotter.utils.fill_plot import oreo_plot, sushi_plot
from kitchen.settings.fluorescence import DF_F0_SIGN, Z_SCORE_SIGN
from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, LOCOMOTION_COLOR, LICK_COLOR, PUPIL_COLOR, WHISKER_COLOR
from kitchen.plotter.plotting_params import LICK_BIN_SIZE, LOCOMOTION_BIN_SIZE, RAW_FLUORESCENCE_RATIO
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE, FLUORESCENCE_TRACE_STYLE, LICK_TRACE_STYLE, LOCOMOTION_TRACE_STYLE, PUPIL_CENTER_X_TRACE_STYLE, PUPIL_CENTER_Y_TRACE_STYLE, PUPIL_SACCADE_TRACE_STYLE, PUPIL_TRACE_STYLE, SUBTRACT_STYLE, WHISKER_TRACE_STYLE
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.structure.neural_data_structure import Events, Fluorescence, Pupil, TimeSeries



SUBTRACT_MANUAL = namedtuple(
    "SUBTRACT_MANUAL", 
    ["color1", "name1", "color2", "name2"], 
    # defaults=["#298c8c", None, "#f1a226", None]
    defaults=["C0", None, "C1", None]
)


def unit_subtract_locomotion(
        locomotion1: None | list[Events],
        locomotion2: None | list[Events],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0,
        yticks_flag: bool = True,
        baseline_subtraction: Optional[tuple[float, float]] = None) -> float:
    """plot locomotion"""
    if not sanity_check(locomotion1) or not sanity_check(locomotion2):
        return 0
    assert locomotion1 is not None and locomotion2 is not None, "Sanity check failed"

    # plot multiple locomotion rates
    group_locomotion1 = grouping_events_rate(locomotion1, bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=baseline_subtraction)
    group_locomotion2 = grouping_events_rate(locomotion2, bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=baseline_subtraction)

    oreo_plot(ax, group_locomotion1, y_offset, ratio, LOCOMOTION_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
    oreo_plot(ax, group_locomotion2, y_offset, ratio, LOCOMOTION_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

    # plot subtraction
    sushi_plot(ax, group_locomotion1, group_locomotion2, y_offset, ratio, SUBTRACT_STYLE)

    # add y ticks
    if yticks_flag:
        yticks_combo("locomotion" if baseline_subtraction is None else "delta_locomotion", ax, y_offset, ratio)
    else:
        add_new_yticks(ax, TICK_PAIR(y_offset, "", color_scheme.LOCOMOTION_COLOR))
    return max(np.nanmax(group_locomotion1.mean) * ratio, np.nanmax(group_locomotion2.mean) * ratio, 2*ratio)


def unit_subtract_lick(
        lick1: None | list[Events],
        lick2: None | list[Events],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot lick"""    
    if not sanity_check(lick1) or not sanity_check(lick2):
        return 0
    assert lick1 is not None and lick2 is not None, "Sanity check failed"

    # plot multiple licks
    group_lick1 = grouping_events_rate(lick1, bin_size=LICK_BIN_SIZE)
    group_lick2 = grouping_events_rate(lick2, bin_size=LICK_BIN_SIZE)

    oreo_plot(ax, group_lick1, y_offset, ratio, LICK_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
    oreo_plot(ax, group_lick2, y_offset, ratio, LICK_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

    # plot subtraction
    sushi_plot(ax, group_lick1, group_lick2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Lick", LICK_COLOR),
                        TICK_PAIR(y_offset + 5 * ratio, "5 Hz", LICK_COLOR)])
    return max(np.nanmax(group_lick1.mean) * ratio, np.nanmax(group_lick2.mean) * ratio, 5*ratio)


def unit_subtract_pupil(
        pupil1: None | Pupil | list[Pupil],
        pupil2: None | Pupil | list[Pupil],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0,
        yticks_flag: bool = True,
        baseline_subtraction: Optional[tuple[float, float]] = None) -> float:
    """plot pupil"""    
    if not sanity_check(pupil1) or not sanity_check(pupil2):
        return 0
    assert pupil1 is not None and pupil2 is not None, "Sanity check failed"

    if isinstance(pupil1, Pupil):
        # plot single pupil
        assert isinstance(pupil2, TimeSeries), "Sanity check failed"
        plotting_pupil1 = pupil1.area_ts.copy()
        plotting_pupil2 = pupil2.area_ts.copy()
        if baseline_subtraction is not None:
            plotting_pupil1.v -= np.mean(plotting_pupil1.segment(*baseline_subtraction).v)
            plotting_pupil2.v -= np.mean(plotting_pupil2.segment(*baseline_subtraction).v)
        ax.plot(plotting_pupil1.t, plotting_pupil1.v * ratio + y_offset, **PUPIL_TRACE_STYLE | {"color": subtract_manual.color1})
        ax.plot(plotting_pupil2.t, plotting_pupil2.v * ratio + y_offset, **PUPIL_TRACE_STYLE | {"color": subtract_manual.color2})

        # plot subtraction
        sushi_plot(ax, pupil1.area_ts, pupil2.area_ts, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
        y_height = max(np.nanmax(plotting_pupil1.v), np.nanmax(plotting_pupil2.v)) * ratio
    else:
        # plot multiple pupils
        assert isinstance(pupil2, list), "Sanity check failed"
        group_pupil1 = grouping_timeseries([single_pupil.area_ts for single_pupil in pupil1], 
                                           baseline_subtraction=baseline_subtraction)
        group_pupil2 = grouping_timeseries([single_pupil.area_ts for single_pupil in pupil2],
                                           baseline_subtraction=baseline_subtraction)
        oreo_plot(ax, group_pupil1, y_offset, ratio, PUPIL_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
        oreo_plot(ax, group_pupil2, y_offset, ratio, PUPIL_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

        # plot subtraction
        sushi_plot(ax, group_pupil1, group_pupil2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)     
        y_height = max(np.nanmax(group_pupil1.mean), np.nanmax(group_pupil2.mean)) * ratio

    # add y ticks
    if yticks_flag:
        yticks_combo("pupil" if baseline_subtraction is None else "delta_pupil", ax, y_offset, ratio)
    else:
        add_new_yticks(ax, TICK_PAIR(y_offset, "", color_scheme.PUPIL_COLOR))
    return y_height


def unit_subtract_pupil_center(
        pupil1: None | Pupil | list[Pupil],
        pupil2: None | Pupil | list[Pupil],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0,
        yticks_flag: bool = True,
        baseline_subtraction: Optional[tuple[float, float, bool]] = None) -> float:
    """plot pupil"""    
    if not sanity_check(pupil1) or not sanity_check(pupil2):
        return 0
    assert pupil1 is not None and pupil2 is not None, "Sanity check failed"

    if isinstance(pupil1, Pupil):
        # plot single pupil
        assert isinstance(pupil2, Pupil), "Sanity check failed"
        plotting_pupil_saccade1 = pupil1.saccade_velocity_ts.copy()
        plotting_pupil_saccade2 = pupil2.saccade_velocity_ts.copy()
        ax.plot(plotting_pupil_saccade1.t, plotting_pupil_saccade1.v * ratio + y_offset, **PUPIL_SACCADE_TRACE_STYLE | {"color": subtract_manual.color1})
        ax.plot(plotting_pupil_saccade2.t, plotting_pupil_saccade2.v * ratio + y_offset, **PUPIL_SACCADE_TRACE_STYLE | {"color": subtract_manual.color2})
        y_height = max(np.nanmax(plotting_pupil_saccade1.v), np.nanmax(plotting_pupil_saccade2.v),) * ratio
    else:
        # plot multiple pupils
        assert isinstance(pupil2, list), "Sanity check failed"
        group_pupil_saccade1 = grouping_timeseries([single_pupil.saccade_velocity_ts for single_pupil in pupil1], 
                                                    baseline_subtraction=baseline_subtraction)
        group_pupil_saccade2 = grouping_timeseries([single_pupil.saccade_velocity_ts for single_pupil in pupil2], 
                                                    baseline_subtraction=baseline_subtraction)
        oreo_plot(ax, group_pupil_saccade1, y_offset, ratio, PUPIL_SACCADE_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
        oreo_plot(ax, group_pupil_saccade2, y_offset, ratio, PUPIL_SACCADE_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

        y_height = max(np.nanmax(group_pupil_saccade1.mean), np.nanmax(group_pupil_saccade2.mean)) * ratio
    # add y ticks
    if yticks_flag:
        yticks_combo("saccade", ax, y_offset, ratio)
    else:
        add_new_yticks(ax, TICK_PAIR(y_offset, "", color_scheme.PUPIL_CENTER_COLOR))
    return y_height


def unit_subtract_whisker(
        whisker1: None | TimeSeries | list[TimeSeries],
        whisker2: None | TimeSeries | list[TimeSeries],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0,
        yticks_flag: bool = True,
        baseline_subtraction: Optional[tuple[float, float]] = None) -> float:
    """plot whisker"""   
    if not sanity_check(whisker1) or not sanity_check(whisker2):
        return 0
    assert whisker1 is not None and whisker2 is not None, "Sanity check failed" 

    if isinstance(whisker1, TimeSeries):
        # plot single whisker
        assert isinstance(whisker2, TimeSeries), "Sanity check failed"
        plotting_whisker1 = whisker1.copy()
        plotting_whisker2 = whisker2.copy()
        if baseline_subtraction is not None:
            plotting_whisker1.v -= np.mean(plotting_whisker1.segment(*baseline_subtraction).v)
            plotting_whisker2.v -= np.mean(plotting_whisker2.segment(*baseline_subtraction).v)
        ax.plot(plotting_whisker1.t, plotting_whisker1.v * ratio + y_offset, **WHISKER_TRACE_STYLE | {"color": subtract_manual.color1})
        ax.plot(plotting_whisker2.t, plotting_whisker2.v * ratio + y_offset, **WHISKER_TRACE_STYLE | {"color": subtract_manual.color2})   

        # plot subtraction
        sushi_plot(ax, whisker1, whisker2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
    else:
        # plot multiple whiskers
        assert isinstance(whisker2, list), "Sanity check failed"
        group_whisker1 = grouping_timeseries(whisker1, baseline_subtraction=baseline_subtraction)
        group_whisker2 = grouping_timeseries(whisker2, baseline_subtraction=baseline_subtraction)
        oreo_plot(ax, group_whisker1, y_offset, ratio, WHISKER_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
        oreo_plot(ax, group_whisker2, y_offset, ratio, WHISKER_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

        # plot subtraction
        sushi_plot(ax, group_whisker1, group_whisker2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)

    # add y ticks
    if yticks_flag:
        yticks_combo("whisker" if baseline_subtraction is None else "delta_whisker", ax, y_offset, ratio)
    else:
        add_new_yticks(ax, TICK_PAIR(y_offset, "", color_scheme.WHISKER_COLOR))
    return ratio if baseline_subtraction is None else 0.5 * ratio


def unit_subtract_single_cell_fluorescence(
        fluorescence1: None | Fluorescence | list[Fluorescence],
        fluorescence2: None | Fluorescence | list[Fluorescence],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0,
        cell_id_flag: bool = True) -> float:
    """plot a single cell"""
    if not sanity_check(fluorescence1) or not sanity_check(fluorescence2):
        return 0
    assert fluorescence1 is not None and fluorescence2 is not None, "Sanity check failed"   

    if isinstance(fluorescence1, Fluorescence):
        # plot single cell fluorescence
        assert isinstance(fluorescence2, Fluorescence), "Sanity check failed"
        assert fluorescence1.num_cell == 1 == fluorescence2.num_cell, \
            f"Expected 1 cell, but got {fluorescence1.num_cell} and {fluorescence2.num_cell}"
        assert np.all(fluorescence1.cell_idx == fluorescence2.cell_idx), \
            f"Expected same cell, but got {fluorescence1.cell_idx} and {fluorescence2.cell_idx}"
        
        ratio *= RAW_FLUORESCENCE_RATIO
        cell_trace1 = fluorescence1.z_score.v[0]
        ax.plot(fluorescence1.z_score.t, cell_trace1 * ratio + y_offset,
                **(FLUORESCENCE_TRACE_STYLE | {"color": subtract_manual.color1}))      
        cell_trace2 = fluorescence2.z_score.v[0]
        ax.plot(fluorescence2.z_score.t, cell_trace2 * ratio + y_offset,
                **(FLUORESCENCE_TRACE_STYLE | {"color": subtract_manual.color2}))      

        # add y ticks  
        add_new_yticks(ax, TICK_PAIR(
            y_offset, f"Cell {fluorescence1.cell_idx[0]}" if cell_id_flag else "Cell", FLUORESCENCE_COLOR))      
        add_new_yticks(ax, TICK_PAIR(
            y_offset + 1 * ratio,
            f"1 {Z_SCORE_SIGN}" if (np.all(fluorescence1.cell_order == 0) or (not cell_id_flag)) else "", FLUORESCENCE_COLOR))
        
        # plot subtraction
        sushi_plot(ax, cell_trace1, cell_trace2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
        return max(np.nanmax(cell_trace1.v) * ratio, np.nanmax(cell_trace2.v) * ratio, 1*ratio)

    # plot multiple cell fluorescence
    assert isinstance(fluorescence2, list), "Sanity check failed"
    group_fluorescence1 = grouping_timeseries([fluorescence.df_f0 for fluorescence in fluorescence1]).squeeze(0)
    group_fluorescence2 = grouping_timeseries([fluorescence.df_f0 for fluorescence in fluorescence2]).squeeze(0)
    
    oreo_plot(ax, group_fluorescence1, y_offset, ratio, FLUORESCENCE_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
    oreo_plot(ax, group_fluorescence2, y_offset, ratio, FLUORESCENCE_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)
    
    # add y ticks
    example_fluorescence = fluorescence1[0]
    add_new_yticks(ax, TICK_PAIR(
        y_offset, f"Cell {example_fluorescence.cell_idx[0]}" if cell_id_flag else "Cell", FLUORESCENCE_COLOR))      
    add_new_yticks(ax, TICK_PAIR(
        y_offset + 1 * ratio,
        f"1 {DF_F0_SIGN}" if (np.all(example_fluorescence.cell_order == 0) or (not cell_id_flag)) else "", FLUORESCENCE_COLOR))    

    # plot subtraction
    sushi_plot(ax, group_fluorescence1, group_fluorescence2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
    return max(np.nanmax(group_fluorescence1.mean) * ratio, np.nanmax(group_fluorescence2.mean) * ratio, 1*ratio)
        