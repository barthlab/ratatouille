from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from kitchen.operator.grouping import AdvancedTimeSeries, calculate_group_tuple, grouping_events_rate, grouping_timeseries
from kitchen.plotter import color_scheme, style_dicts, plotting_params
from kitchen.plotter.plotting_params import HEATMAP_OFFSET_RANGE, LOCOMOTION_BIN_SIZE
from kitchen.plotter.unit_plotter.unit_trace import sanity_check
from kitchen.structure.neural_data_structure import Events, Pupil, TimeSeries


def label_heatmap_y_ticklabels(ax: plt.Axes, row_num: int, extent_range: tuple[float, float]) -> None:
    """label the y axis of the heatmap"""
    row_height = (extent_range[1] - extent_range[0]) / row_num
    ax.set_yticks([extent_range[0], extent_range[1]], ["1", f"{row_num}"], rotation=0)


def default_ax_realign(ax: plt.Axes) -> None:
    """realign the ax to the default heatmap style"""
    ax.set_ylim(0, None)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.spines['top'].set_visible(True)
    ax.invert_yaxis()


def sort_heatmap_array(heatmap_array_adv_ts: AdvancedTimeSeries, sorting_level_refs: Optional[list[str]] = None,
                      amplitude_sorting: Optional[tuple[float, float]] = None) -> np.ndarray:
    """sort the heatmap array based on the amplitude sorting"""
    if amplitude_sorting is None:
        return heatmap_array_adv_ts.raw_array
    amplitude_range = np.searchsorted(heatmap_array_adv_ts.t, amplitude_sorting)
    amplitudes = np.nanmean(heatmap_array_adv_ts.raw_array[:, amplitude_range[0]:amplitude_range[1]], axis=1)
    if sorting_level_refs is not None:
        assert len(sorting_level_refs) == len(amplitudes), "Sorting level refs should have same length as amplitudes"
        sort_idxs = np.lexsort((amplitudes, np.array(sorting_level_refs)))
    else:
        sort_idxs = np.argsort(amplitudes)
    return heatmap_array_adv_ts.raw_array[sort_idxs]


def unit_heatmap_locomotion(locomotion: None | list[Events], ax: plt.Axes,
                         yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                         sorting_level_refs: Optional[list[str]] = None,
                         amplitude_sorting: Optional[tuple[float, float]] = None) -> float:
    """plot locomotion"""
    if not sanity_check(locomotion):
        return 0
    assert locomotion is not None, "Sanity check failed"

    # plot multiple locomotion rates
    group_locomotion = grouping_events_rate(locomotion, bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_locomotion, sorting_level_refs, amplitude_sorting)
    heatmap_extent = (group_locomotion.t[0], group_locomotion.t[-1], HEATMAP_OFFSET_RANGE[1], HEATMAP_OFFSET_RANGE[0])
    
    ax.imshow(heatmap_array, extent=heatmap_extent, 
              cmap=color_scheme.LOCOMOTION_COLORMAP if baseline_subtraction is None else color_scheme.BASELINE_SUBTRACTION_COLORMAP, 
              **plotting_params.LOCOMOTION_VMIN_VMAX[baseline_subtraction is None],
              **style_dicts.HEATMAP_STYLE)
    default_ax_realign(ax)
    
    # add y ticks
    if yticks_flag:
        label_heatmap_y_ticklabels(ax, len(locomotion), HEATMAP_OFFSET_RANGE)
    else:
        ax.set_yticks([])
    return HEATMAP_OFFSET_RANGE[1]


def unit_heatmap_whisker(whisker: None | list[TimeSeries], ax: plt.Axes,
                         yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                         sorting_level_refs: Optional[list[str]] = None,
                         amplitude_sorting: Optional[tuple[float, float]] = None) -> float:
    """plot whisker"""
    if not sanity_check(whisker):
        return 0
    assert whisker is not None, "Sanity check failed"

    # plot multiple whiskers
    group_whisker = grouping_timeseries(whisker, baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_whisker, sorting_level_refs, amplitude_sorting)
    heatmap_extent = (group_whisker.t[0], group_whisker.t[-1], HEATMAP_OFFSET_RANGE[1], HEATMAP_OFFSET_RANGE[0])

    ax.imshow(heatmap_array, extent=heatmap_extent, 
              cmap=color_scheme.WHISKER_COLORMAP if baseline_subtraction is None else color_scheme.BASELINE_SUBTRACTION_COLORMAP,
              **plotting_params.WHISKER_VMIN_VMAX[baseline_subtraction is None], 
              **style_dicts.HEATMAP_STYLE)
    default_ax_realign(ax)
    
    # add y ticks
    if yticks_flag:
        label_heatmap_y_ticklabels(ax, len(whisker), HEATMAP_OFFSET_RANGE)
    else:
        ax.set_yticks([])
    return HEATMAP_OFFSET_RANGE[1]


def unit_heatmap_pupil(pupil: None | list[Pupil], ax: plt.Axes,
                       yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                       sorting_level_refs: Optional[list[str]] = None,
                       amplitude_sorting: Optional[tuple[float, float]] = None) -> float:
    """plot pupil"""
    if not sanity_check(pupil):
        return 0
    assert pupil is not None, "Sanity check failed"

    # plot multiple pupils
    group_pupil = grouping_timeseries([single_pupil.area_ts for single_pupil in pupil], baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_pupil, sorting_level_refs, amplitude_sorting)
    heatmap_extent = (group_pupil.t[0], group_pupil.t[-1], HEATMAP_OFFSET_RANGE[1], HEATMAP_OFFSET_RANGE[0])
    
    ax.imshow(heatmap_array, extent=heatmap_extent, 
              cmap=color_scheme.PUPIL_COLORMAP if baseline_subtraction is None else color_scheme.BASELINE_SUBTRACTION_COLORMAP,
              **plotting_params.PUPIL_VMIN_VMAX[baseline_subtraction is None], 
              **style_dicts.HEATMAP_STYLE)
    default_ax_realign(ax)
    
    # add y ticks
    if yticks_flag:
        label_heatmap_y_ticklabels(ax, len(pupil), HEATMAP_OFFSET_RANGE)
    else:
        ax.set_yticks([])
    return HEATMAP_OFFSET_RANGE[1]


def unit_heatmap_saccade(saccade: None | list[Pupil], ax: plt.Axes,
                         yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                         sorting_level_refs: Optional[list[str]] = None,
                         amplitude_sorting: Optional[tuple[float, float]] = None) -> float:
    """plot saccade"""
    if not sanity_check(saccade):
        return 0
    assert saccade is not None, "Sanity check failed"   
    
    # plot multiple saccades
    group_saccade = grouping_timeseries([single_pupil.saccade_velocity_ts for single_pupil in saccade], 
                                              baseline_subtraction=baseline_subtraction)
    heatmap_array = group_saccade.raw_array
    heatmap_extent = (group_saccade.t[0], group_saccade.t[-1], HEATMAP_OFFSET_RANGE[1], HEATMAP_OFFSET_RANGE[0])
    heatmap_array = sort_heatmap_array(calculate_group_tuple([arr for arr in heatmap_array], group_saccade.t), 
                                        sorting_level_refs, amplitude_sorting)
    
    ax.imshow(heatmap_array, extent=heatmap_extent, 
              cmap=color_scheme.PUPIL_CENTER_COLORMAP if baseline_subtraction is None else color_scheme.BASELINE_SUBTRACTION_COLORMAP,
              **plotting_params.PUPIL_CENTER_VMIN_VMAX[baseline_subtraction is None], 
              **style_dicts.HEATMAP_STYLE)
    default_ax_realign(ax)

    # add y ticks
    if yticks_flag:
        label_heatmap_y_ticklabels(ax, len(saccade), HEATMAP_OFFSET_RANGE)
    else:
        ax.set_yticks([])
    return HEATMAP_OFFSET_RANGE[1]