import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Behavior color scheme
LOCOMOTION_COLOR = "Blue"
POSITION_COLOR = "Orange"
LICK_COLOR = "Red"
PUPIL_COLOR = "Purple"
WHISKER_COLOR = "Green"
FLUORESCENCE_COLOR = "Black"
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
BLANK_COLOR = "lightgray"
WATER_COLOR = "cornflowerblue"
NOWATER_COLOR = "orangered"
BUZZER_COLOR = "darkorange"
PELTIER_COLOR = "aqua"
DARK_STIM_COLOR = "darkblue"
DARK_PUFF_COLOR = "gray"
DARK_BLANK_COLOR = "gray"
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

