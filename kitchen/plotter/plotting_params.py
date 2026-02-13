# Ratio
POSITION_RATIO = 0.6
LOCOMOTION_RATIO = 0.15
LICK_RATIO = 0.1
PUPIL_RATIO = 4
# PUPIL_RATIO = 2
SACCADE_RATIO = 6
WHISKER_RATIO = 1
NOSE_RATIO = 1

RAW_FLUORESCENCE_RATIO = 0.2
FLUORESCENCE_RATIO = 1

POTENTIAL_RATIO = 1
WC_POTENTIAL_RATIO = 1/15

TIMELINE_RATIO = 0.5

RATIO_DICT = {
    "position": POSITION_RATIO,
    "locomotion": LOCOMOTION_RATIO,
    "lick": LICK_RATIO,
    "pupil": PUPIL_RATIO,
    "whisker": WHISKER_RATIO,
    "fluorescence": FLUORESCENCE_RATIO,
    "timeline": TIMELINE_RATIO,
    "nose": NOSE_RATIO,
}

# Bin size
LOCOMOTION_BIN_SIZE = 0.2  # s
LICK_BIN_SIZE = 0.2  # s



# Tick 
TIME_TICK_DURATION = 200  # s


# Annotation 
ANNOTATION_OFFSET_FACTOR = 0.1
ANNOTATION_BBOX_HEIGHT_FACTOR = 0.5


# Figure params
NORMALIZED_PROGRESS_YLIM = 15
DPI = 500

FLAT_X_INCHES = 30      # inches
FLAT_Y_INCHES = 6       # inches

STACK_X_INCHES = 4.     # inches
STACK_Y_INCHES = 4.    # inches

UNIT_X_INCHES = 5       # inches
UNIT_Y_INCHES = 2.      # inches

PARALLEL_X_INCHES = 2.4 # inches
PARALLEL_Y_INCHES = 4.  # inches
ZOOMED_Y_INCHES = 8     # inches

HEATMAP_X_INCHES = 2   # inches
HEATMAP_Y_INCHES = 3   # inches

# Heamap params
HEATMAP_OFFSET_RANGE = (1., 10.)

LOCOMOTION_VMIN_VMAX = {
    False: {"vmin": -2, "vmax": +2},
    True: {"vmin": 0, "vmax": 2.5},
}
WHISKER_VMIN_VMAX = {
    False: {"vmin": -1, "vmax": +1},
    True: {"vmin": 0, "vmax": 1.},
}
PUPIL_VMIN_VMAX = {
    False: {"vmin": -0.2, "vmax": +0.2},
    True: {"vmin": 0, "vmax": 0.5},
}
PUPIL_CENTER_VMIN_VMAX = {
    False: {"vmin": -0.5, "vmax": +0.5},
    True: {"vmin": 0, "vmax": 0.5},
}
FLUORESCENCE_VMIN_VMAX = {
    False: {"vmin": -2, "vmax": +2},
    True: {"vmin": -2, "vmax": 2},
}
FLUORESCENCE_DECONV_VMIN_VMAX = {
    False: {"vmin": -1, "vmax": +1},
    True: {"vmin": 0, "vmax": 5},
}