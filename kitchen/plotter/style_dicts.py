from kitchen.settings.timeline import SUPPORTED_TIMELINE_EVENT
from kitchen.plotter.color_scheme import FLUORESCENCE_COLOR, LOCOMOTION_COLOR, POSITION_COLOR, LICK_COLOR, PUPIL_COLOR, SUBTRACT_COLOR, WHISKER_COLOR, PUFF_COLOR, BLANK_COLOR, WATER_COLOR, NOWATER_COLOR, BUZZER_COLOR

DEFAULT_ALPHA = 0.9
MAX_OVERLAP_ALPHA_NUM_DUE_TO_MATPLOTLLIB_BUG = 200 # 255, see matplotlib alpha bug

REFERENCE_LINE_STYLE = {
    "color": "gray",
    "linewidth": 0.5,
    "linestyle": "--",
    "alpha": 0.7,
    "zorder": -5,
}

LOCOMOTION_TRACE_STYLE = {
    "color": LOCOMOTION_COLOR,
    "linewidth": 0.5,
    "alpha": 0.7,
}

POSITION_SCATTER_STYLE = {
    "color": POSITION_COLOR,
    "s": 1,
    "alpha": 0.7,
}

LICK_VLINES_STYLE = {
    "colors": LICK_COLOR,
    "ls": 'solid',
    "alpha": 0.7,
    "lw": 0.1,
}

LICK_TRACE_STYLE = {
    "color": LICK_COLOR,
    "linewidth": 0.5,
    "alpha": 0.7,
}

PUPIL_TRACE_STYLE = {
    "color": PUPIL_COLOR,
    "linewidth": 0.5,
    "alpha": 0.7,
}

WHISKER_TRACE_STYLE = {
    "color": WHISKER_COLOR,
    "linewidth": 0.5,
    "alpha": 0.7,
}

FLUORESCENCE_TRACE_STYLE = {
    "color": FLUORESCENCE_COLOR,
    "linewidth": 0.5,
    "alpha": 0.7,
}

FILL_BETWEEN_STYLE = {
    "lw": 0,
    "alpha": 0.5,
}

SUBTRACT_STYLE = {
    "color": SUBTRACT_COLOR,
    "lw": 0.2,
    "alpha": 0.8,
    "zorder": 10,
}

LEGEND_STYLE = {
    "frameon": False,
    "loc": "lower right",
}

VSPAN_STYLE = {
    "lw": 0,
    "alpha": 0.5,
    "zorder": -10,
}

VLINE_STYLE = {
    "lw": 0.5,
    "alpha": 0.5,
    "ls": "--",
    "zorder": -10,
}

BARPLOT_STYLE = {
    "width": 0.6,    
    "capsize": 4,
    "edgecolor": "black",
    "linewidth": 0.2,
    "error_kw": {'elinewidth': 0.5, "ecolor": "black", "capthick": 0.5, "alpha": 0.7},
    "alpha": 0.8,
    "zorder": -5,
}

SWARM_JITTER_STD = 0.01 * BARPLOT_STYLE["width"]
SWARM_STYLE = {
    "facecolors": "black",
    "edgecolors": "none",
    "s": 2,
    "alpha": 0.9,
    "zorder": 2,
}

COMPARISON_LINE_STYLE = {
    "color": "black",
    "linewidth": 0.5,
    "alpha": 0.7,
    "zorder": 10,
}

ANNOTATION_LINE_STYLE = {
    "color": "black",
    "linewidth": 0.5,
    "alpha": 0.7,
    "zorder": 10,
}

ANNOTATION_TEXT_STYLE_SIGNIFICANT = {
    "ha": 'center',
    "va": 'bottom',
    "color": 'black',
    "alpha": 0.8,
    "zorder": 10,
}

ANNOTATION_TEXT_STYLE_NONSIGNIFICANT = {
    "ha": 'center',
    "va": 'bottom',
    "color": 'gray',
    "alpha": 0.4,
    "zorder": 10,
}

TIMELINE_SCATTER_STYLE = {
    "VerticalPuffOn": {
        "color": PUFF_COLOR,
        "s": 8,
        "marker": "|",
        "alpha": 1,
        "lw": 0.5,
        "zorder": 10,
    },
    "Puff": {
        "color": PUFF_COLOR,
        "s": 8,
        "marker": "|",
        "alpha": 1,
        "lw": 0.5,
        "zorder": 10,
    },
    "HorizontalPuffOn": {
        "color": PUFF_COLOR,
        "s": 5,
        "marker": "-",
        "alpha": 1,
        "lw": 1,
        "zorder": 10,
    },
    "BlankOn": {
        "facecolors": 'none',
        "edgecolors": BLANK_COLOR,
        "s": 4,
        "marker": "o",
        "alpha": 1,
        "lw": 0.5,
        "zorder": 10,
    },
    "Blank": {
        "facecolors": 'none',
        "edgecolors": BLANK_COLOR,
        "s": 4,
        "marker": "o",
        "alpha": 1,
        "lw": 0.5,
        "zorder": 10,
    },
    "WaterOn": {
        "color": WATER_COLOR,
        "s": 4,
        "marker": "^",
        "alpha": 0.7,
        "lw": 0,
        "zorder": 10,
    },
    "NoWaterOn": {
        "color": NOWATER_COLOR,
        "s": 2,
        "marker": "x",
        "alpha": 0.7,
        "lw": 0.8,
        "zorder": 10,
    },
    "BuzzerOn": {
        "color": BUZZER_COLOR,
        "s": 4,
        "marker": "s",
        "alpha": 0.7,
        "lw": 0,
        "zorder": 10,
    },

}

for event_type in TIMELINE_SCATTER_STYLE:
    assert event_type in SUPPORTED_TIMELINE_EVENT, f"Unsupported event {event_type} in TIMELINE_SCATTER_STYLE: {SUPPORTED_TIMELINE_EVENT}"
