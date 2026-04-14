import colorsys
import math
import hashlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


# Behavior color scheme
LOCOMOTION_COLOR = "Blue"
POSITION_COLOR = "Orange"

LICK_COLOR = "Red"

PUPIL_COLOR = "Purple"
PUPIL_CENTER_COLOR = "Brown"
PUPIL_CENTER_SACCADE_COLOR = "Red"
IS_BLINK_COLOR = "Blue"
PUPIL_CENTER_X_COLOR = "Orange"
PUPIL_CENTER_Y_COLOR = "Brown"

WHISKER_COLOR = "Green"
NOSE_COLOR = "Pink"

FLUORESCENCE_COLOR = "Black"
DECONV_FLUORESCENCE_COLOR = "gray"
POTENTIAL_COLOR = "Black"
INDIVIDUAL_FLUORESCENCE_COLOR = "gray"

SUBTRACT_COLOR = "dimgray"

# Spike color scheme
REGULAR_SPIKE_COLOR = "#616161"
EARLY_SPIKE_COLOR = "#ff7043"
SUSTAINED_SPIKE_COLOR = "#7e57c2"
DETECTED_SPIKE_COLOR = "#c2185b"

# Timeline color scheme
STIM_COLOR = "blue"
PUFF_COLOR = "#9e9e9e"
BLUELED_COLOR = "blue"
BLANK_COLOR = "#e0e0e0"
WATER_COLOR = "cornflowerblue"
NOWATER_COLOR = "orangered"
BUZZER_COLOR = "darkorange"
PELTIER_COLOR = "aqua"
DARK_STIM_COLOR = "darkblue"
DARK_PUFF_COLOR = "gray"
DARK_BLUELED_COLOR = "darkblue"
DARK_BLANK_COLOR = "lightgray"
DARK_WATER_COLOR = "blue"
DARK_NOWATER_COLOR = "red"
DARK_BUZZER_COLOR = "chocolate"
DARK_PELTIER_COLOR = "cyan"

# Event color scheme
REWARD_TRIAL_COLOR = "#298c8c"
OMISSION_TRIAL_COLOR = "#f1a226"

# Grand color scheme
GRAND_COLOR_SCHEME = {
    "StimOn": STIM_COLOR,
    "VerticalPuffOn": PUFF_COLOR,
    "Puff": PUFF_COLOR,
    "blueLEDOn": BLUELED_COLOR,
    "HorizontalPuffOn": PUFF_COLOR,
    "BlankOn": BLANK_COLOR,
    "Blank": BLANK_COLOR,
    "WaterOn": WATER_COLOR,
    "NoWaterOn": NOWATER_COLOR,
    "BuzzerOn": BUZZER_COLOR,
    "FakeRelayOn": BLANK_COLOR,
    "PeltierLeftOn": DARK_PELTIER_COLOR,
    "PeltierRightOn": DARK_PELTIER_COLOR,
    "PeltierBothOn": DARK_PELTIER_COLOR,
}
SPIKE_COLOR_SCHEME = {
    "spike": REGULAR_SPIKE_COLOR,
    "regular_spike": REGULAR_SPIKE_COLOR,
    "early_spike": EARLY_SPIKE_COLOR,
    "sustained_spike": SUSTAINED_SPIKE_COLOR,
    "detected_spike": DETECTED_SPIKE_COLOR,
}
TABLEAU_10 = list(mcolors.TABLEAU_COLORS.values())


# Behavior color map
BASELINE_SUBTRACTION_COLORMAP = "RdYlBu_r"

LOCOMOTION_COLORMAP = "Blues"
POSITION_COLORMAP = "Oranges"

LICK_COLORMAP = "Greys"

PUPIL_COLORMAP = "Purples"
PUPIL_CENTER_COLORMAP = "Oranges"

WHISKER_COLORMAP = "Greens"
# NOSE_COLORMAP = "RdYlBu_r"

FLUORESCENCE_COLORMAP = "RdYlBu_r"
DECONV_FLUORESCENCE_COLORMAP = "YlOrBr"
# POTENTIAL_COLORMAP = "RdYlBu_r"
# INDIVIDUAL_FLUORESCENCE_COLORMAP = "RdYlBu_r"

# SUBTRACT_COLORMAP = "RdYlBu_r"





# USE 
from distinctipy import distinctipy
N_COLORS = 11
DISTINCT_COLORS = distinctipy.get_colors(N_COLORS)


def string_to_hex_color(text: str):
    data = text.encode('utf-8')

    hash_object = hashlib.md5(data)
    hex_dig = hash_object.hexdigest()

    return DISTINCT_COLORS[int(hex_dig[:6]) % N_COLORS]


def num_to_hex_color(num: int, max_num: int = N_COLORS):
    if max_num != N_COLORS:
        return distinctipy.get_colors(max_num)[num % max_num]
    return DISTINCT_COLORS[num % N_COLORS]

