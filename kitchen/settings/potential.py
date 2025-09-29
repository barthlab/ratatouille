from enum import Enum
import numpy as np

SPIKE_MIN_DISTANCE: float = 2 / 1000  # s
SPIKE_MAX_WINDOW_LEN: float = 2 / 1000  # s
SPIKE_MIN_PROMINENCE_STD_RATIO: float = 4.0  # times of std
SPIKE_MIN_HEIGHT_STD_RATIO: float = 4.0  # times of std
STD_SLIDING_WINDOW = 2  # s

# Bandwidth
SPIKES_BANDWIDTH: float = 300.  # Hz
MAXIMAL_BANDWIDTH: float = 5000.  # Hz
COMPONENTS_BANDWIDTH: tuple[float, ...] = (0.5, 4., 80., 300., 1000.)  # Hz

# Airpuff
JS_AIRPUFF_THRESHOLD: float = 2.5

# Cam timeline
JS_CAM_THRESHOLD: float = 1.5


# For whole cell vs jux adjustment
def WC_CONVERT_FLAG(node):
	return "wc" in node.cohort_id.lower()

# Spike range
SPIKE_RANGE_RELATIVE_TO_ALIGNMENT = (-1./1000, 1./1000)  # s 
CURATION_SPIKE_RANGE_RELATIVE_TO_ALIGNMENT = (-3/1000, 3/1000)  # s 
IMPORTANT_SPIKE_TIMEPOINTS = tuple(i/1000 for i in np.linspace(-1, 1, 11))  # s

# Spike type annotation
SPIKE_ANNOTATION_EARLY_WINDOW = (2/1000, 40/1000)  # s
SPIKE_STICKY_WINDOW = 20/1000  # s


# class PotentialType(str, Enum):
#     WholeCell = "WholeCell"
#     PatchClamp = "PatchClamp"
#     Juxtacellular = "Juxtacellular"

