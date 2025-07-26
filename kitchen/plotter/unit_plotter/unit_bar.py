from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from kitchen.plotter.stats_tests.basic_ttest import SIGNIFICANT_ALPHA, stats_mannwhitneyu, stats_ttest_ind, stats_ttest_rel, stats_wilcoxon
from kitchen.plotter.style_dicts import ANNOTATION_LINE_STYLE, ANNOTATION_TEXT_STYLE_SIGNIFICANT, ANNOTATION_TEXT_STYLE_NONSIGNIFICANT, BARPLOT_STYLE, COMPARISON_LINE_STYLE, SWARM_JITTER_STD, SWARM_STYLE
from kitchen.plotter.utils.alpha_calculator import calibrate_alpha



def unit_bar(
        bar_data: Tuple[float, np.ndarray],  # position and data values
        ax: plt.Axes,
        y_offset: float, ratio: float = 1.0,
        bar_styles: dict = {}
):
    bar_position, bar_values = bar_data
    mean = np.nanmean(bar_values)
    sem = np.nanstd(bar_values) / np.sqrt(len(bar_values))
    ax.bar(bar_position, mean * ratio, bottom=y_offset, yerr=sem * ratio, **(BARPLOT_STYLE | bar_styles))
    return mean * ratio + y_offset
    

def unit_swarm(
        bar_data: Tuple[float, np.ndarray],  # position and data values
        ax: plt.Axes,
        y_offset: float, ratio: float = 1.0,
):
    bar_position, bar_values = bar_data
    jitter = np.random.normal(0, SWARM_JITTER_STD, size=len(bar_values))
    ax.scatter(bar_position + jitter, bar_values * ratio + y_offset, **calibrate_alpha(SWARM_STYLE, len(bar_values)))


def unit_comparison_line(
        bar1: Tuple[float, np.ndarray],
        bar2: Tuple[float, np.ndarray],
        ax: plt.Axes,
        y_offset: float, ratio: float = 1.0,
):
    bar1_position, bar1_values = bar1
    bar2_position, bar2_values = bar2
    assert len(bar1_values) == len(bar2_values), "Cannot compare bars with different number of data points"

    for i in range(len(bar1_values)):
        ax.plot([bar1_position, bar2_position], [bar1_values[i] * ratio + y_offset, bar2_values[i] * ratio + y_offset], 
                **calibrate_alpha(COMPARISON_LINE_STYLE, len(bar1_values)))


def unit_comparison_annotation_ttest_ind(
        bar1: Tuple[float, np.ndarray],
        bar2: Tuple[float, np.ndarray],
        ax: plt.Axes,
        y_bbox: Tuple[float, float],
        y_offset: float, ratio: float = 1.0,
):
    y_bottom, y_top = y_bbox
    bar1_position, bar1_values = bar1
    bar2_position, bar2_values = bar2

    p_val, p_text = stats_ttest_ind(bar1_values, bar2_values)

    bar1_values, bar2_values = bar1_values * ratio + y_offset, bar2_values * ratio + y_offset
    ax.plot([bar1_position, bar1_position, bar2_position, bar2_position],
            [y_bottom, y_top, y_top, y_bottom], **ANNOTATION_LINE_STYLE)
    ax.text((bar1_position + bar2_position) / 2, y_top, p_text, 
            **(ANNOTATION_TEXT_STYLE_SIGNIFICANT if p_val < SIGNIFICANT_ALPHA 
               else ANNOTATION_TEXT_STYLE_NONSIGNIFICANT))


def unit_comparison_annotation_ttest_rel(
        bar1: Tuple[float, np.ndarray],
        bar2: Tuple[float, np.ndarray],
        ax: plt.Axes,
        y_bbox: Tuple[float, float],
        y_offset: float, ratio: float = 1.0,
):
    y_bottom, y_top = y_bbox
    bar1_position, bar1_values = bar1
    bar2_position, bar2_values = bar2    

    p_val, p_text = stats_ttest_rel(bar1_values, bar2_values)
    
    bar1_values, bar2_values = bar1_values * ratio + y_offset, bar2_values * ratio + y_offset
    ax.plot([bar1_position, bar1_position, bar2_position, bar2_position],
            [y_bottom, y_top, y_top, y_bottom], **ANNOTATION_LINE_STYLE)
    ax.text((bar1_position + bar2_position) / 2, y_top, p_text, 
            **(ANNOTATION_TEXT_STYLE_SIGNIFICANT if p_val < SIGNIFICANT_ALPHA 
               else ANNOTATION_TEXT_STYLE_NONSIGNIFICANT))

    # Draw lines connecting paired data points
    unit_comparison_line(bar1, bar2, ax, y_offset, ratio)


def unit_comparison_annotation_mannwhitneyu(
        bar1: Tuple[float, np.ndarray],
        bar2: Tuple[float, np.ndarray],
        ax: plt.Axes,
        y_bbox: Tuple[float, float],
        y_offset: float, ratio: float = 1.0,
):
    """
    Performs a Mann-Whitney U test (non-parametric independent t-test) and annotates the plot.
    """
    y_bottom, y_top = y_bbox
    bar1_position, bar1_values = bar1
    bar2_position, bar2_values = bar2

    p_val, p_text = stats_mannwhitneyu(bar1_values, bar2_values)

    bar1_values, bar2_values = bar1_values * ratio + y_offset, bar2_values * ratio + y_offset
    ax.plot([bar1_position, bar1_position, bar2_position, bar2_position],
            [y_bottom, y_top, y_top, y_bottom], **ANNOTATION_LINE_STYLE)
    ax.text((bar1_position + bar2_position) / 2, y_top, p_text,
            **(ANNOTATION_TEXT_STYLE_SIGNIFICANT if p_val < SIGNIFICANT_ALPHA
               else ANNOTATION_TEXT_STYLE_NONSIGNIFICANT))


def unit_comparison_annotation_wilcoxon(
        bar1: Tuple[float, np.ndarray],
        bar2: Tuple[float, np.ndarray],
        ax: plt.Axes,
        y_bbox: Tuple[float, float],
        y_offset: float, ratio: float = 1.0,
):
    """
    Performs a Wilcoxon signed-rank test (non-parametric related t-test) and annotates the plot.
    """
    y_bottom, y_top = y_bbox
    bar1_position, bar1_values = bar1
    bar2_position, bar2_values = bar2

    p_val, p_text = stats_wilcoxon(bar1_values, bar2_values)

    bar1_values, bar2_values = bar1_values * ratio + y_offset, bar2_values * ratio + y_offset
    ax.plot([bar1_position, bar1_position, bar2_position, bar2_position],
            [y_bottom, y_top, y_top, y_bottom], **ANNOTATION_LINE_STYLE)
    ax.text((bar1_position + bar2_position) / 2, y_top, p_text,
            **(ANNOTATION_TEXT_STYLE_SIGNIFICANT if p_val < SIGNIFICANT_ALPHA
               else ANNOTATION_TEXT_STYLE_NONSIGNIFICANT))
    
    # Draw lines connecting paired data points
    unit_comparison_line(bar1, bar2, ax, y_offset, ratio)