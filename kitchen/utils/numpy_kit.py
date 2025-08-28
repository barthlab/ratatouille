import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal.windows import gaussian
from numpy.lib.stride_tricks import sliding_window_view

def numpy_percentile_filter(input_array: np.ndarray, s: int, q: float) -> np.ndarray:
    """
    Apply a percentile filter to a 1D or 2D array efficiently using vectorization.

    Parameters
    ----------
    input_array : np.ndarray
        Input 1D or 2D array.
    s : int
        Window size (must be positive).
    q : float
        Percentile to compute (0 <= q <= 100).

    Returns
    -------
    np.ndarray
        Filtered array of same shape as input.
    """
    # --- Input Validation ---
    if not isinstance(input_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if input_array.ndim > 2:
        raise ValueError(f"Input array must be 1- or 2-dimensional, not {input_array.ndim}")
    if not isinstance(s, int) or s <= 0:
        raise ValueError("Window size 's' must be a positive integer.")
    if not (0 <= q <= 100):
        raise ValueError("Percentile 'q' must be between 0 and 100.")

    # --- Handle Empty Input ---
    if input_array.size == 0:
        return input_array.copy()
    if s > input_array.shape[-1]:
        raise ValueError("Window size 's' cannot be larger than the last dimension of the array.")

    # --- Pre-processing for Unified Handling ---
    original_ndim = input_array.ndim
    arr = np.atleast_2d(input_array)

    # --- Padding along columns (axis=1) ---
    pad_left = s // 2
    pad_right = s - 1 - pad_left
    arr_padded = np.pad(arr, pad_width=((0, 0), (pad_left, pad_right)), mode='edge')

    # --- Create Sliding Windows with sliding_window_view ---
    # This returns shape (M, N, s) without manual stride calculations.
    windows = sliding_window_view(arr_padded, window_shape=s, axis=1)

    # --- Calculation ---
    result = np.percentile(windows, q, axis=2)

    # --- Restore original dimensionality ---
    if original_ndim == 1:
        return result.squeeze(axis=0)
    return result


def smart_interp(x_new, xp, fp, method: str = "previous"):
    if method == "previous":
        f_new = interp1d(xp, fp, kind='previous', axis=-1, bounds_error=False)
    elif method == "nearest":
        f_new = interp1d(xp, fp, kind='nearest', axis=-1, bounds_error=False)
    elif method == "linear":
        f_new = interp1d(xp, fp, kind='linear', axis=-1, bounds_error=False)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    return f_new(x_new)


def sliding_std(x: np.ndarray, window_len: int) -> np.ndarray:
    return np.array(pd.Series(x).rolling(window=window_len).std())


def smooth_uniform(arr, window_len):
    """Smoothes with a uniform moving average."""
    window = np.ones(window_len) / window_len
    smoothed_valid = np.convolve(arr, window, mode='same')
    return smoothed_valid

def smooth_gaussian(arr, window_len, std):
    """Smoothes with a Gaussian window. Requires SciPy."""
    window = gaussian(window_len, std=std)
    window /= np.sum(window)
    smoothed_valid = np.convolve(arr, window, mode='same')
    return smoothed_valid
