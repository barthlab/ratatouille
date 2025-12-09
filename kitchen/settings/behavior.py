# Behavior data loading settings and constants

# Supported behavior data types
SUPPORTED_BEHAVIOR_TYPES = {
    "LOCOMOTION",
    "LICK",
    "PUPIL",
    "TONGUE",
    "WHISKER",    
    "NOSE",
}

VIDEO_EXTRACTED_BEHAVIOR_TYPES = {
    "PUPIL",
    "TONGUE",
    "WHISKER",    
    "NOSE",
}

OPTICAL_FLOW_EXTRACTED_BEHAVIOR_TYPES = {
    "WHISKER",
    # "NOSE",
    # "TONGUE",
}


# Processing parameters
LICK_INACTIVATE_WINDOW = 1 / 20  # s - minimum time between lick events
LOCOMOTION_NUM_TICKS = 600  # number of ticks per round
LOCOMOTION_CIRCUMFERENCE = 29.8  # cm - circumference of the wheel
VIDEO_EXTRACTED_BEHAVIOR_MIN_MAX_PERCENTILE = (1, 99)  # % - percentile for min and max value of video extracted behavior data

PUPIL_AREA_NORMALIZATION = 10000  # normalize pupil area to 10000 pixel^2
PUPIL_CENTER_NORMALIZATION = 100  # normalize pupil center to 100 pixel
IS_BLINK_THRESHOLD = 0.5  # threshold for is_blink

VIDOE_CLOSE_DELAY = 0.2  # s - delay for closing the picamera, see https://github.com/raspberrypi/picamera2/issues/1136