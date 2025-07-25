import math

from kitchen.plotter.style_dicts import DEFAULT_ALPHA


def ind_alpha(total_alpha: float, num_overlaps: int):
    """Calculate the individual alpha required to reach the total_alpha"""
    if num_overlaps <= 0:
        return 0
    if total_alpha <= 0.0:
        return 0.0
    if total_alpha >= 1.0:
        return 1.0
    return -math.expm1(math.log1p(-total_alpha) / num_overlaps)


def calibrate_alpha(style: dict, num_overlaps: int):
    """Calibrate the alpha in a style dict"""
    if "alpha" in style:
        return style | {"alpha": ind_alpha(style["alpha"], num_overlaps)}
    else:
        return style | {"alpha": ind_alpha(DEFAULT_ALPHA, num_overlaps)}
