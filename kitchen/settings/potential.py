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
NARROW_SPIKE_RANGE_FOR_VISUALIZATION =  (-2/1000, 2/1000)  # s
IMPORTANT_SPIKE_TIMEPOINTS = tuple(i/1000 for i in np.linspace(-1, 1, 11))  # s

# Spike type annotation
SPIKE_ANNOTATION_EARLY_WINDOW = (0/1000, 40/1000)  # s
SPIKE_STICKY_WINDOW = 20/1000  # s


# class PotentialType(str, Enum):
#     WholeCell = "WholeCell"
#     PatchClamp = "PatchClamp"
#     Juxtacellular = "Juxtacellular"



# GCaMP6f KERNEL
# EXP_TAU_DECAY = 0.4  # s, decay tau_0.5 GCaMP6f
# EXP_TAU_RISE = 0.08  # s, rise tau_peak GCaMP6f
MODEL_TAU_DECAY = 0.58  # s, calibrate for difference of two exponential model, should be EXP_TAU_DECAY/ln(2)
MODEL_TAU_RISE = 0.02  # s, calibrate for difference of two exponential model, should satisfy EXP_TAU_RISE = (MODEL_TAU_RISE * MODEL_TAU_DECAY)/(MODEL_TAU_DECAY - MODEL_TAU_RISE)ln(MODEL_TAU_DECAY/MODEL_TAU_RISE)
UNIT_DF_F0 = 0.19  # dF/F0, 1AP, GCaMP6f

def GCaMP_kernel(sampling_rate) -> tuple[np.ndarray, np.ndarray]:
    # Define the time vector for the kernel
    kernel_length_s = 5 * MODEL_TAU_DECAY
    t = np.arange(0, kernel_length_s, 1/sampling_rate)

    # Double exponential function for the kernel
    kernel = (np.exp(-t / MODEL_TAU_DECAY) - np.exp(-t / MODEL_TAU_RISE))
    kernel = kernel / np.max(kernel)
        
    # Scale the kernel by the unit dF/F0 for a single action potential
    kernel = kernel * UNIT_DF_F0
    
    return kernel, t