import numpy as np

from kitchen.utils.numpy_kit import numpy_percentile_filter


CROP_BEGGINNG = 0.3  # s - crop the first 300ms of the recording (See MES software manual)
DEFAULT_RECORDING_DURATION = 600  # s - the default recording duration for each session
FAST_MATCHING_MODE = False  # only match the first event

NULL_TTL_OFFSET = 0.5  # s - if no ttl is found, align the fluorescence to the timeline by this offset

# Constants for fluorescence processing
NEUROPIL_SUBTRACTION_COEFFICIENT = 0.7  # coefficient for subtracting neuropil signal
DEFAULT_FRAMES_PER_SESSION = 3061  # default number of frames per session for classic loader
DF_F0_SIGN = r"$\Delta F/F_0$"
Z_SCORE_SIGN = r"$z$-score"


# process fluorescence
DETREND_BASELINE_WINDOW = 60  # s
FLUORESCENCE_MIN_MAX_PERCENTILE = (10, 90)  # % - percentile for min and max value normalization

# baseline for df/f0
TRIAL_DF_F0_WINDOW = (-1, 0)  # s
DF_F0_RANGE = (-3, 10)  # dF/F0 range for clipping




# GCaMP6f Deconvolution
DO_DECONV = False
DECONV_TAU = 0.58  # s
DECONV_PERCENTILE = 20 # %
DECONV_WINDOW_BASELINE = 60.0 # in seconds, window in which to compute max/min filters
def GCaMP_deconvolve(raw_f: np.ndarray, fs: float) -> np.ndarray:
    from suite2p.extraction import dcnv
    window_size = int(DECONV_WINDOW_BASELINE * fs)
    normed_f = raw_f / numpy_percentile_filter(raw_f, window_size, DECONV_PERCENTILE)
    deconv_f = dcnv.oasis(F=normed_f, tau=DECONV_TAU, fs=fs, batch_size=1000) * 10
    return deconv_f