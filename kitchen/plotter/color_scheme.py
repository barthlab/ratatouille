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
DARK_PUFF_COLOR = "darkgray"
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

LICK_COLORMAP = "Reds"

PUPIL_COLORMAP = "Purples"
PUPIL_CENTER_COLORMAP = "Oranges"

WHISKER_COLORMAP = "Greens"
# NOSE_COLORMAP = "RdYlBu_r"

FLUORESCENCE_COLORMAP = "RdYlBu_r"
DECONV_FLUORESCENCE_COLORMAP = "YlOrBr"
# POTENTIAL_COLORMAP = "RdYlBu_r"
# INDIVIDUAL_FLUORESCENCE_COLORMAP = "RdYlBu_r"

# SUBTRACT_COLORMAP = "RdYlBu_r"



def string_to_hex_color(text: str) -> str:
    data = text.encode('utf-8')

    hash_object = hashlib.md5(data)
    hex_dig = hash_object.hexdigest()

    return f"#{hex_dig[:6]}"



def num_to_color(n: int) -> str:
    """
    Returns a consistent color for any positive integer.
    0-9  : Returns standard "Cn" strings (Matplotlib Tableaus).
    10+  : Generates a unique Hex color that fits the style but is distinct.
    """
    if 0 <= n < 10:
        return f"C{n}"

    golden_ratio = 0.618033988749895
    
    h = (n * golden_ratio) % 1.0
    
    s = 0.75 
    v = 0.95 

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))