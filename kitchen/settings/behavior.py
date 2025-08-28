# Behavior data loading settings and constants

# Supported behavior data types
SUPPORTED_BEHAVIOR_TYPES = {
    "LOCOMOTION",
    "LICK",
    "PUPIL",
    "TONGUE",
    "WHISKER",
}

VIDEO_EXTRACTED_BEHAVIOR_TYPES = {
    "PUPIL",
    "TONGUE",
    "WHISKER",
}

OPTICAL_FLOW_EXTRACTED_BEHAVIOR_TYPES = {
    "WHISKER",
}

# Processing parameters
LICK_INACTIVATE_WINDOW = 1 / 20  # s - minimum time between lick events
LOCOMOTION_NUM_TICKS = 600  # number of ticks per round
LOCOMOTION_CIRCUMFERENCE = 29.8  # cm - circumference of the wheel
VIDEO_EXTRACTED_BEHAVIOR_MIN_MAX_PERCENTILE = (2, 98)  # % - percentile for min and max value of video extracted behavior data