from enum import Enum

SPIKE_MIN_DISTANCE: float = 2 / 1000  # s
SPIKE_MAX_WINDOW_LEN: float = 2 / 1000  # s
SPIKE_MIN_HEIGHT_STD_RATIO: float = 5.0  # times of std
STD_SLIDING_WINDOW = 5  # s

# Bandwidth
SPIKES_BANDWIDTH: float = 300  # Hz
COMPONENTS_BANDWIDTH: tuple[float, ...] = (0.5, 4, 80, 300, 1000)  # Hz

# Airpuff
JS_AIRPUFF_THRESHOLD: float = 2.5

class PotentialType(str, Enum):
    WholeCell = "WholeCell"
    PatchClamp = "PatchClamp"
    Juxtacellular = "Juxtacellular"

