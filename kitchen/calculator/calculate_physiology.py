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


def calculate_autocorrelogram_rise_time(spike_times: np.ndarray, bin_size: float=0.001, max_lag: float=10):
    if not isinstance(spike_times, np.ndarray) or spike_times.ndim != 1:
        raise TypeError("spike_times must be a 1D NumPy array.")
    
    if len(spike_times) < 2:
        print("Warning: Not enough spikes to compute an autocorrelogram. Returning NaNs.")
        return np.nan

    # Calculate all inter-spike intervals (ISIs) up to max_lag
    lags = []
    for i in range(len(spike_times)):
        for j in range(i + 1, len(spike_times)):
            lag = spike_times[j] - spike_times[i]
            # Since spike_times is sorted, we can break early
            if lag > max_lag:
                break
            lags.append(lag)

    if not lags:
        print(f"Warning: No spike pairs found within the max_lag of {max_lag}s.")
        return np.nan

    # Create the histogram (the autocorrelogram)
    bins = np.arange(bin_size, max_lag + bin_size, bin_size)
    counts, bin_edges = np.histogram(lags, bins=bins)

    # Find the bin with the highest count
    peak_bin_index = np.argmax(counts)
    rise_time = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0
    return rise_time