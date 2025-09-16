# TRIAL_RANGE_RELATIVE_TO_ALIGNMENT = (-4, 8)  # s 
TRIAL_RANGE_RELATIVE_TO_ALIGNMENT = (-1, 2.5)  # s 



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