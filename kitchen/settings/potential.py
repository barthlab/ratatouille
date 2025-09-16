from enum import Enum

SPIKE_MIN_DISTANCE: float = 2 / 1000  # s
SPIKE_MAX_WINDOW_LEN: float = 2 / 1000  # s
SPIKE_MIN_HEIGHT_STD_RATIO: float = 5.0  # times of std
STD_SLIDING_WINDOW = 5  # s

# Bandwidth
SPIKES_BANDWIDTH: float = 300  # Hz
COMPONENTS_BANDWIDTH: tuple[float, ...] = (0.5, 4., 80., 300., 1000.)  # Hz

# Airpuff
JS_AIRPUFF_THRESHOLD: float = 2.5

# Cam timeline
JS_CAM_THRESHOLD: float = 1.5


# For whole cell vs jux adjustment
WC_POTENTIAL_THRESHOLD = -10.  # mV

# Spike range
SPIKE_RANGE_RELATIVE_TO_ALIGNMENT = (-1.5/1000, 1.5/1000)  # s 


# Spike type annotation
SPIKE_ANNOTATION_EARLY_WINDOW = (2/1000, 40/1000)  # s
SPIKE_STICKY_WINDOW = 20/1000  # s


# class PotentialType(str, Enum):
#     WholeCell = "WholeCell"
#     PatchClamp = "PatchClamp"
#     Juxtacellular = "Juxtacellular"

