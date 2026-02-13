from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from kitchen.operator.grouping import AdvancedTimeSeries, calculate_group_tuple, grouping_events_rate, grouping_timeseries
from kitchen.plotter import color_scheme, style_dicts, plotting_params
from kitchen.plotter.plotting_params import HEATMAP_OFFSET_RANGE, LOCOMOTION_BIN_SIZE
from kitchen.plotter.unit_plotter.unit_trace import sanity_check
from kitchen.structure.neural_data_structure import Events, Fluorescence, Pupil, TimeSeries
from kitchen.calculator.sorting_data import get_amplitude_sorted_idxs


def label_heatmap_y_ticklabels(ax: plt.Axes, row_num: int, extent_range: tuple[float, float]) -> None:
    """label the y axis of the heatmap"""
    row_height = (extent_range[1] - extent_range[0]) / row_num
    ax.set_yticks([extent_range[0] + row_height / 2, extent_range[1] - row_height / 2], ["1", f"{row_num}"], rotation=0)
    

def default_ax_realign(ax: plt.Axes) -> None:
    """realign the ax to the default heatmap style"""
    ax.set_ylim(0, None)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.spines['top'].set_visible(True)
    ax.invert_yaxis()


def sort_heatmap_array(heatmap_array_adv_ts: AdvancedTimeSeries, 
                       specified_order: Optional[np.ndarray] = None,
                       sorting_key: str = "none",
                       **kwargs,
):
    """sort the heatmap array based on the specified order or sorting key"""
    if sorting_key == "none":
        return heatmap_array_adv_ts.raw_array[specified_order] if specified_order is not None else heatmap_array_adv_ts.raw_array
    elif sorting_key == "amplitude":
        return heatmap_array_adv_ts.raw_array[get_amplitude_sorted_idxs(heatmap_array_adv_ts, **kwargs)]
    else:
        raise ValueError(f"Unknown sorting key: {sorting_key}")


def unit_heatmap_locomotion(locomotion: None | list[Events], ax: plt.Axes,
                         yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                         **sorting_kwargs
                         ) -> float:
    """plot locomotion"""
    if not sanity_check(locomotion):
        return 0
    assert locomotion is not None, "Sanity check failed"

    # plot multiple locomotion rates
    group_locomotion = grouping_events_rate(locomotion, bin_size=LOCOMOTION_BIN_SIZE, baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_locomotion, **sorting_kwargs)
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
                         **sorting_kwargs
                         ) -> float:
    """plot whisker"""
    if not sanity_check(whisker):
        return 0
    assert whisker is not None, "Sanity check failed"

    # plot multiple whiskers
    group_whisker = grouping_timeseries(whisker, baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_whisker, **sorting_kwargs)
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
                       **sorting_kwargs
                       ) -> float:
    """plot pupil"""
    if not sanity_check(pupil):
        return 0
    assert pupil is not None, "Sanity check failed"

    # plot multiple pupils
    group_pupil = grouping_timeseries([single_pupil.area_ts for single_pupil in pupil], baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_pupil, **sorting_kwargs)
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
                         **sorting_kwargs
                         ) -> float:
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
                                        **sorting_kwargs)
    
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


def unit_heatmap_fluorescence(
                        fluorescence: None | list[Fluorescence], ax: plt.Axes,
                       yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                       **sorting_kwargs
                       ) -> float:
    """plot fluorescence"""
    if not sanity_check(fluorescence):
        return 0
    assert fluorescence is not None, "Sanity check failed"

    # plot multiple fluorescences
    assert all(fluorescence.num_cell == 1 for fluorescence in fluorescence), f"Expected 1 cell, but got {fluorescence}"
    group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) for single_fluorescence in fluorescence], 
                                             baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_fluorescence, **sorting_kwargs)
    heatmap_extent = (group_fluorescence.t[0], group_fluorescence.t[-1], HEATMAP_OFFSET_RANGE[1], HEATMAP_OFFSET_RANGE[0])
    
    ax.imshow(heatmap_array, extent=heatmap_extent, 
              cmap=color_scheme.FLUORESCENCE_COLORMAP if baseline_subtraction is None else color_scheme.BASELINE_SUBTRACTION_COLORMAP,
              **plotting_params.FLUORESCENCE_VMIN_VMAX[baseline_subtraction is None], 
              **style_dicts.HEATMAP_STYLE)
    default_ax_realign(ax)
    
    # add y ticks
    if yticks_flag:
        label_heatmap_y_ticklabels(ax, len(fluorescence), HEATMAP_OFFSET_RANGE)
    else:
        ax.set_yticks([])
    return HEATMAP_OFFSET_RANGE[1]


def unit_heatmap_deconv_fluorescence(
                        fluorescence: None | list[Fluorescence], ax: plt.Axes,
                       yticks_flag: bool = True, baseline_subtraction: Optional[tuple[float, float, bool]] = None,
                       **sorting_kwargs
                       ) -> float:
    """plot fluorescence"""
    if not sanity_check(fluorescence):
        return 0
    assert fluorescence is not None, "Sanity check failed"

    # plot multiple fluorescences
    group_deconv_fluorescence = grouping_timeseries([single_fluorescence.deconv_f.squeeze(0) for single_fluorescence in fluorescence], 
                                             baseline_subtraction=baseline_subtraction)
    heatmap_array = sort_heatmap_array(group_deconv_fluorescence, **sorting_kwargs)
    heatmap_extent = (group_deconv_fluorescence.t[0], group_deconv_fluorescence.t[-1], HEATMAP_OFFSET_RANGE[1], HEATMAP_OFFSET_RANGE[0])
    
    ax.imshow(heatmap_array, extent=heatmap_extent, 
              cmap=color_scheme.DECONV_FLUORESCENCE_COLORMAP if baseline_subtraction is None else color_scheme.BASELINE_SUBTRACTION_COLORMAP,
              **plotting_params.FLUORESCENCE_DECONV_VMIN_VMAX[baseline_subtraction is None], 
              **style_dicts.HEATMAP_STYLE)
    default_ax_realign(ax)
    
    # add y ticks
    if yticks_flag:
        label_heatmap_y_ticklabels(ax, len(fluorescence), HEATMAP_OFFSET_RANGE)
    else:
        ax.set_yticks([])
    return HEATMAP_OFFSET_RANGE[1]