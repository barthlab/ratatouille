import numpy as np


def calculate_spike_width_and_asymmetry(spike_waveforms: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate spike width and asymmetry from spike waveforms.

    spike width = trough to peak time (in ms)
    spike asymmetry = (post_peak_mag - pre_peak_mag) / (pre_peak_mag + post_peak_mag)
    """
    if not isinstance(spike_waveforms, np.ndarray) or spike_waveforms.ndim != 2:
        raise ValueError(f"spike_waveforms should be 2-d array, got {spike_waveforms.shape}")
    if not isinstance(sampling_rate_hz, (int, float)) or sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz should be positive, got {sampling_rate_hz}")

    # peak is in the middle
    n_spikes, n_samples = spike_waveforms.shape
    peak_index = int(n_samples/2)
    
    # convert sample index to ms
    ms_per_sample = 1000.0 / sampling_rate_hz

    # initialize output arrays
    spike_widths_ms = np.zeros(n_spikes)
    spike_asymmetries = np.zeros(n_spikes)

    for i in range(n_spikes):
        waveform = spike_waveforms[i, :]

        # peak is in the middle
        peak_amplitude = waveform[peak_index]

        # trough is the minimum after the peak
        post_peak_waveform = waveform[peak_index:]
        trough_index_relative = np.argmin(post_peak_waveform)
        trough_index = peak_index + trough_index_relative
        trough_amplitude = waveform[trough_index] 

        # calculate width
        width_in_samples = trough_index - peak_index
        spike_widths_ms[i] = width_in_samples * ms_per_sample

        # calculate asymmetry
        post_mag = float(peak_amplitude - trough_amplitude)

        pre_peak_waveform = waveform[:peak_index]
        pre_peak_min_amplitude = np.min(pre_peak_waveform)
        pre_mag = float(peak_amplitude - pre_peak_min_amplitude)
        
        denominator = pre_mag + post_mag
        if denominator > 1e-9: 
             spike_asymmetries[i] = (post_mag - pre_mag) / denominator
        else:
             spike_asymmetries[i] = np.nan

    return spike_widths_ms, spike_asymmetries



def calculate_cv2(spike_times: np.ndarray) -> float:
    """
    Calculates the CV2 for a sequence of spike times.

    CV2 is a measure of local firing regularity, defined as the average of
    2 * |ISI_{i+1} - ISI_i| / (ISI_{i+1} + ISI_i) for all adjacent
    inter-spike intervals (ISIs). It is less sensitive to slow changes in firing rate
    than the standard coefficient of variation (CV).
    """
    if not isinstance(spike_times, np.ndarray) or spike_times.ndim != 1:
        raise TypeError(f"spike_times must be a 1D NumPy array. Got {spike_times.shape}")
    if spike_times.size < 3:
        return np.nan
        
    # Calculate inter-spike intervals (ISIs)
    isis = np.diff(spike_times)
    
    # Calculate the CV2 for each adjacent pair of ISIs    
    diff_isis = np.abs(np.diff(isis))  # Numerator: 2 * |ISI_{i+1} - ISI_i|
    sum_isis = isis[:-1] + isis[1:]  # Denominator: ISI_{i+1} + ISI_i
    
    cv2_pairs = 2 * diff_isis / sum_isis
    return float(np.mean(cv2_pairs))



import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


# CellExplorer fit_ACG start point and bounds:
# [tau_decay, tau_rise, c, d, asymptote, refrac, tau_burst, h]
_CE_P0 = (20.0, 1.0, 30.0, 2.0, 0.5, 5.0, 1.5, 2.0)

_CE_BOUNDS = (
    (1.0,   0.1, 0.0,  0.0, -30.0, 0.0, 0.1,   0.0),
    (500.0, 50.0, 500.0, 15.0, 50.0, 20.0, 5.0, 100.0),
)

_CE_PARAM_NAMES = (
    "tau_decay_ms",
    "tau_rise_ms",
    "c",
    "d",
    "asymptote",
    "refrac_ms",
    "tau_burst_ms",
    "h",
)


def _prepare_spikes(spike_times):
    """Return finite, sorted, 1D spike times."""
    spikes = np.asarray(spike_times, dtype=float)

    if spikes.ndim != 1:
        raise TypeError("spike_times must be a 1D array.")

    return np.sort(spikes[np.isfinite(spikes)])


def _positive_acg_counts(spikes, bin_size, max_lag):
    """Count positive-lag spike pairs using bins centered on multiples of bin_size."""
    n_bins = int(np.round(max_lag / bin_size))
    counts = np.zeros(n_bins + 1, dtype=np.int64)
    upper_lag = max_lag + bin_size / 2

    for i, t in enumerate(spikes[:-1]):
        stop = np.searchsorted(spikes, t + upper_lag, side="right")

        dt = spikes[i + 1 : stop] - t
        k = np.floor(dt / bin_size + 0.5).astype(int)
        k = k[(k >= 1) & (k <= n_bins)]

        if k.size:
            counts += np.bincount(k, minlength=n_bins + 1)

    return counts


def _acg_model(
    x_ms,
    tau_decay,
    tau_rise,
    c,
    d,
    asymptote,
    refrac,
    tau_burst,
    h,
):
    """CellExplorer triple-exponential ACG model; times are in ms."""
    y = (
        c
        * (
            np.exp(-(x_ms - refrac) / tau_decay)
            - d * np.exp(-(x_ms - refrac) / tau_rise)
        )
        + h * np.exp(-(x_ms - refrac) / tau_burst)
        + asymptote
    )

    return np.maximum(y, 0.0)


def calculate_autocorrelogram_rise_time(
    spike_times,
    return_diagnostics=False,
):
    """
    Estimate CellExplorer-style ACG tau_rise in milliseconds.

    spike_times are in seconds.
    """
    spikes = _prepare_spikes(spike_times)

    if spikes.size < 2:
        return (np.nan, {}) if return_diagnostics else np.nan

    # CellExplorer narrow ACG: +/-50 ms, 0.5-ms bins.
    bin_size = 0.0005
    max_lag = 0.050

    counts = _positive_acg_counts(spikes, bin_size, max_lag)

    x_ms = np.arange(1, counts.size) * (bin_size * 1000.0)
    y = counts[1:].astype(float) / (spikes.size * bin_size)

    # CellExplorer zeros the +0.5-ms point before fitting.
    y[0] = 0.0

    try:
        params, _ = curve_fit(
            _acg_model,
            x_ms,
            y,
            p0=_CE_P0,
            bounds=_CE_BOUNDS,
            maxfev=100_000,
        )

        fitted = _acg_model(x_ms, *params)

        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum((y - fitted) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    except (RuntimeError, ValueError, FloatingPointError):
        params = np.full(8, np.nan)
        fitted = np.full_like(y, np.nan)
        r2 = np.nan

    tau_rise_ms = float(params[1])

    if not return_diagnostics:
        return tau_rise_ms

    diagnostics = {
        "n_spikes": spikes.size,
        "x_ms": x_ms,
        "acg_rate_hz": y,
        "fitted_acg": fitted,
        "r2": r2,
        **dict(zip(_CE_PARAM_NAMES, params)),
    }

    return tau_rise_ms, diagnostics


def calculate_autocorrelogram(
    spike_times: np.ndarray,
    bin_size: float = 0.0005,
    max_lag: float = 1,
    normalize: str = "rate",
    smooth_sigma: float = 1.0,
):
    """
    Calculate a symmetric autocorrelogram.

    spike_times, bin_size, and max_lag are in seconds.
    smooth_sigma is in bins.
    """
    spikes = _prepare_spikes(spike_times)

    # Preserved from the original function.
    if spikes.size < 2:
        return np.nan

    if bin_size <= 0 or max_lag <= 0:
        raise ValueError("bin_size and max_lag must be positive.")

    positive = _positive_acg_counts(spikes, bin_size, max_lag)
    n_bins = len(positive) - 1

    lags = np.arange(-n_bins, n_bins + 1) * bin_size

    # Mirror positive pair counts; self-pairs at zero lag are excluded.
    counts = np.r_[positive[:0:-1], 0.0, positive[1:]]

    if normalize == "rate":
        acg = counts / (spikes.size * bin_size)
    elif normalize == "count":
        acg = counts
    else:
        raise ValueError("normalize must be 'rate' or 'count'.")

    if smooth_sigma is not None and smooth_sigma > 0:
        acg = gaussian_filter1d(acg, sigma=smooth_sigma)

    return acg, lags