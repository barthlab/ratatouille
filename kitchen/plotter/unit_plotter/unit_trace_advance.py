from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

from kitchen.operator.grouping import grouping_events_rate, grouping_timeseries
from kitchen.plotter.unit_plotter.unit_trace import sanity_check
from kitchen.plotter.utils.fill_plot import oreo_plot, sushi_plot
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.plotter.color_scheme import LOCOMOTION_COLOR, LICK_COLOR, PUPIL_COLOR, SUBTRACT_COLOR, WHISKER_COLOR
from kitchen.plotter.plotting_params import LICK_BIN_SIZE, LOCOMOTION_BIN_SIZE
from kitchen.plotter.style_dicts import FILL_BETWEEN_STYLE, FLUORESCENCE_TRACE_STYLE, LICK_TRACE_STYLE, LOCOMOTION_TRACE_STYLE, PUPIL_TRACE_STYLE, SUBTRACT_STYLE, WHISKER_TRACE_STYLE
from kitchen.plotter.utils.tick_labels import TICK_PAIR, add_new_yticks
from kitchen.structure.neural_data_structure import Events, Fluorescence, TimeSeries



SUBTRACT_MANUAL = namedtuple(
    "SUBTRACT_MANUAL", 
    ["color1", "name1", "color2", "name2"], 
    defaults=["#298c8c", None, "#f1a226", None]
)


def unit_subtract_locomotion(
        locomotion1: None | list[Events],
        locomotion2: None | list[Events],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot locomotion"""
    if not sanity_check(locomotion1) or not sanity_check(locomotion2):
        return 0
    assert locomotion1 is not None and locomotion2 is not None, "Sanity check failed"

    # plot multiple locomotion rates
    group_locomotion1 = grouping_events_rate(locomotion1, bin_size=LOCOMOTION_BIN_SIZE)
    group_locomotion2 = grouping_events_rate(locomotion2, bin_size=LOCOMOTION_BIN_SIZE)

    oreo_plot(ax, group_locomotion1, y_offset, ratio, LOCOMOTION_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
    oreo_plot(ax, group_locomotion2, y_offset, ratio, LOCOMOTION_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

    # plot subtraction
    sushi_plot(ax, group_locomotion1, group_locomotion2, y_offset, ratio, SUBTRACT_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Locomotion", LOCOMOTION_COLOR), 
                        TICK_PAIR(y_offset + 2 * ratio, "2 cm/s", LOCOMOTION_COLOR)])
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
        pupil1: None | TimeSeries | list[TimeSeries],
        pupil2: None | TimeSeries | list[TimeSeries],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot pupil"""    
    if not sanity_check(pupil1) or not sanity_check(pupil2):
        return 0
    assert pupil1 is not None and pupil2 is not None, "Sanity check failed"

    if isinstance(pupil1, TimeSeries):
        # plot single pupil
        assert isinstance(pupil2, TimeSeries), "Sanity check failed"
        ax.plot(pupil1.t, pupil1.v * ratio + y_offset, **PUPIL_TRACE_STYLE | {"color": subtract_manual.color1})
        ax.plot(pupil2.t, pupil2.v * ratio + y_offset, **PUPIL_TRACE_STYLE | {"color": subtract_manual.color2})      

        # plot subtraction
        sushi_plot(ax, pupil1, pupil2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
    else:
        # plot multiple pupils
        assert isinstance(pupil2, list), "Sanity check failed"
        group_pupil1 = grouping_timeseries(pupil1)
        group_pupil2 = grouping_timeseries(pupil2)
        oreo_plot(ax, group_pupil1, y_offset, ratio, PUPIL_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
        oreo_plot(ax, group_pupil2, y_offset, ratio, PUPIL_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

        # plot subtraction
        sushi_plot(ax, group_pupil1, group_pupil2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)     

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Pupil", PUPIL_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Pupil", PUPIL_COLOR)])
    return ratio


def unit_subtract_whisker(
        whisker1: None | TimeSeries | list[TimeSeries],
        whisker2: None | TimeSeries | list[TimeSeries],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
    """plot whisker"""   
    if not sanity_check(whisker1) or not sanity_check(whisker2):
        return 0
    assert whisker1 is not None and whisker2 is not None, "Sanity check failed" 

    if isinstance(whisker1, TimeSeries):
        # plot single whisker
        assert isinstance(whisker2, TimeSeries), "Sanity check failed"
        ax.plot(whisker1.t, whisker1.v * ratio + y_offset, **WHISKER_TRACE_STYLE | {"color": subtract_manual.color1})
        ax.plot(whisker2.t, whisker2.v * ratio + y_offset, **WHISKER_TRACE_STYLE | {"color": subtract_manual.color2})      

        # plot subtraction
        sushi_plot(ax, whisker1, whisker2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
    else:
        # plot multiple whiskers
        assert isinstance(whisker2, list), "Sanity check failed"
        group_whisker1 = grouping_timeseries(whisker1)
        group_whisker2 = grouping_timeseries(whisker2)
        oreo_plot(ax, group_whisker1, y_offset, ratio, WHISKER_TRACE_STYLE | {"color": subtract_manual.color1}, FILL_BETWEEN_STYLE)
        oreo_plot(ax, group_whisker2, y_offset, ratio, WHISKER_TRACE_STYLE | {"color": subtract_manual.color2}, FILL_BETWEEN_STYLE)

        # plot subtraction
        sushi_plot(ax, group_whisker1, group_whisker2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)

    # add y ticks
    add_new_yticks(ax, [TICK_PAIR(y_offset, "Min Whisker", WHISKER_COLOR),
                        TICK_PAIR(y_offset + ratio, "Max Whisker", WHISKER_COLOR)])
    return ratio


def unit_subtract_single_cell_fluorescence(
        fluorescence1: None | Fluorescence | list[Fluorescence],
        fluorescence2: None | Fluorescence | list[Fluorescence],
        subtract_manual: SUBTRACT_MANUAL,
        ax: plt.Axes, y_offset: float, ratio: float = 1.0) -> float:
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
        
        cell_trace1 = fluorescence1.detrend_f.v[0]
        ax.plot(fluorescence1.detrend_f.t, cell_trace1 * ratio + y_offset,
                **(FLUORESCENCE_TRACE_STYLE | {"color": subtract_manual.color1}))      
        cell_trace2 = fluorescence2.detrend_f.v[0]
        ax.plot(fluorescence2.detrend_f.t, cell_trace2 * ratio + y_offset,
                **(FLUORESCENCE_TRACE_STYLE | {"color": subtract_manual.color2}))      

        # add y ticks  
        add_new_yticks(ax, TICK_PAIR(y_offset, f"Cell {fluorescence1.cell_idx[0]}", SUBTRACT_COLOR))      
        add_new_yticks(ax, TICK_PAIR(y_offset + 1 * ratio,
                                     f"1 {DF_F0_SIGN}" if np.all(fluorescence1.cell_order == 0) else "", SUBTRACT_COLOR))
        
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
    add_new_yticks(ax, TICK_PAIR(y_offset, f"Cell {example_fluorescence.cell_idx[0]}", SUBTRACT_COLOR))      
    add_new_yticks(ax, TICK_PAIR(y_offset + 1 * ratio, f"1 {DF_F0_SIGN}" if np.all(example_fluorescence.cell_order == 0) else "", SUBTRACT_COLOR))    

    # plot subtraction
    sushi_plot(ax, group_fluorescence1, group_fluorescence2, y_offset, ratio, FILL_BETWEEN_STYLE | SUBTRACT_STYLE)
    return max(np.nanmax(group_fluorescence1.mean) * ratio, np.nanmax(group_fluorescence2.mean) * ratio, 1*ratio)
        