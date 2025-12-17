# TRIAL_RANGE_RELATIVE_TO_ALIGNMENT = (-4, 8)  # s 
DEFAULT_TRIAL_RANGE_RELATIVE_TO_ALIGNMENT = (-2, 3)  # s 
DEFAULT_TRIAL_RANGE_FOR_TRIAL_TYPE_ANNOTATION = (-2, 3)  # s
DEFAULT_TRIAL_RANGE_FOR_DUMMY_TRIAL = (-5, 8.5)  # s


# Bout detection
BOUT_FILTER_PARAMETERS = {
    "locomotion": {
        "bout_sticky_window": 0.5,  # s
        "minimal_bout_size": 1,
    },
    "lick": {
        "bout_sticky_window": 0.5,  # s
        "minimal_bout_size": 1,
    },
}



# Whisking threshold for trial type annotation
TRIAL_TYPE_ANNOTATION_WHISKER_THRESHOLD = 0.3
TRIAL_TYPE_ANNOTATION_WHISKER_RANGE = (-2.5, -0.5)  # s
