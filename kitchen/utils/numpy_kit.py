from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy import signal
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


def smart_interp(x_new, xp, fp, method: str = "linear") -> np.ndarray:
    # assert not np.any(np.isnan(fp)), f"NaN found in fp: {fp}"
    assert not np.any(np.isnan(xp)), f"NaN found in xp: {xp}"
    assert not np.any(np.isnan(x_new)), f"NaN found in x_new: {x_new}"
    if method == "previous":
        f_new = interp1d(xp, fp, kind='previous', axis=-1, bounds_error=False, fill_value='extrapolate')
    elif method == "nearest":
        f_new = interp1d(xp, fp, kind='nearest', axis=-1, bounds_error=False, fill_value='extrapolate')
    elif method == "linear":
        f_new = interp1d(xp, fp, kind='linear', axis=-1, bounds_error=False, fill_value='extrapolate')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    f_interped = f_new(x_new)
    # assert not np.any(np.isnan(f_interped)), f"NaN introduced in f_interped: {f_interped} \n x_new: {x_new} \n xp: {xp} \n fp: {fp}"
    return f_interped


def smart_nan_removal(xp, fp, method: str = "nearest"):
    mask = np.isnan(fp)
    if mask.all():
        return np.zeros_like(fp)
    fnew = fp.copy()
    fnew[mask] = smart_interp(xp[mask], xp[~mask], fp[~mask], method=method)
    return fnew


def sliding_std(x: np.ndarray, window_len: int) -> np.ndarray:
    return np.array(pd.Series(x).rolling(window=window_len, center=True).std())

def sliding_mean(x: np.ndarray, window_len: int) -> np.ndarray:
    return np.array(pd.Series(x).rolling(window=window_len, center=True, min_periods=1).mean())


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


def zscore(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    return (arr - np.mean(arr, axis=axis, keepdims=True)) / (np.std(arr, axis=axis, keepdims=True) + 1e-8)


def reorder_indices(A: np.ndarray, B: np.ndarray, _allow_leftover: bool = False) -> np.ndarray:
    """
    TODO:
    Can this function works for condtion that B is a subset of A?
    
    """
    """
    Given two 1-D numpy arrays of strings A and B (same length),
    return an index array 'order' such that A[order] == B.
    Raises ValueError if they cannot be matched one-to-one.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if (not _allow_leftover) and A.shape != B.shape:
        raise ValueError("A and B must have the same shape/length.")

    # Build mapping label -> deque of indices in A
    idx_map = defaultdict(deque)
    for i, v in enumerate(A):
        idx_map[v].append(i)

    # For each label in B, pop one index from the corresponding deque
    order = []
    for v in B:
        if v not in idx_map or not idx_map[v]:
            raise ValueError(f"Value '{v}' in B has no matching unused element in A.")
        order.append(idx_map[v].popleft())

    # Sanity: ensure no leftover indices (shouldn't happen if shapes equal and above checks passed)
    leftover = [k for k, dq in idx_map.items() if dq]
    if leftover and not _allow_leftover:
        raise ValueError(f"Extra unused elements left in A (labels): {leftover}")
    
    return np.array(order, dtype=int)


def remove_project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Remove the projection of b from a."""
    return a - (np.dot(a, b) / np.dot(b, b)) * b



OUTLIER_LABEL = (-1, )

def Labels2CoClusterMatrix(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    
    co_cluster_matrix = (labels[:, None] == labels[None, :]).astype(float)  

    valid_labels = ~np.isin(labels, OUTLIER_LABEL)

    valid_pairs = valid_labels[:, None] & valid_labels[None, :]
    co_cluster_matrix[~valid_pairs] = np.nan

    idx = np.where(valid_labels)[0]
    co_cluster_matrix[idx, idx] = 1.0

    return co_cluster_matrix


def CoClusterMatrix2Labels(co_cluster_matrix: np.ndarray) -> np.ndarray:
    co_cluster_matrix = np.asarray(co_cluster_matrix)

    if co_cluster_matrix.ndim != 2:
        raise ValueError("co_cluster_matrix must be a 2D array.")

    n, m = co_cluster_matrix.shape
    if n != m:
        raise ValueError("co_cluster_matrix must be square.")

    # Valid samples have a 1 on the diagonal.
    # Outliers produced by Labels2CoClusterMatrix have NaN there.
    valid = ~np.isnan(np.diag(co_cluster_matrix))

    labels = np.full(n, OUTLIER_LABEL[0], dtype=int)

    cluster_label = 0

    for i in range(n):
        if not valid[i] or labels[i] != OUTLIER_LABEL[0]:
            continue

        # All valid samples co-clustered with i belong to the same cluster.
        members = valid & (co_cluster_matrix[i] == 1.0)
        labels[members] = cluster_label
        cluster_label += 1

    return labels


def matrices_fillback(
    subsample_matrices: list[np.ndarray],
    subsample_indices: np.ndarray,
    full_size: int,
) -> list[np.ndarray]:
    idx = np.asarray(subsample_indices)
    n = len(idx)

    if any(matrix.shape != (n, n) for matrix in subsample_matrices):
        raise ValueError(f"All matrices must have shape ({n}, {n}).")

    ix = np.ix_(idx, idx)

    def fillback(matrix: np.ndarray) -> np.ndarray:
        result = np.full((full_size, full_size), np.nan)
        result[ix] = matrix
        return result

    return [fillback(matrix) for matrix in subsample_matrices]







def _validate_x(x):
    """Convert input to a clean 1D float array."""
    x = np.asarray(x, dtype=float).ravel()

    if x.size < 2:
        raise ValueError("x must contain at least 2 values.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values.")

    return x


def _mean(x):
    return np.mean(_validate_x(x))

def _variance(x):
    return np.var(_validate_x(x), ddof=1)

def _fano_factor(x):
    """
    Fano factor = Var / |mean|.
    """
    x = _validate_x(x)

    mean = np.mean(np.abs(x))

    if np.isclose(mean, 0):
        return np.nan

    return np.var(x, ddof=1) / abs(mean)


def _coefficient_of_variation(x):
    """
    Coefficient of variation = SD / |mean|.
    """
    x = _validate_x(x)

    mean = np.mean(np.abs(x))

    if np.isclose(mean, 0):
        return np.nan

    return np.std(x, ddof=1) / abs(mean)


def _normalized_participation_ratio(x):
    """
    Participation ratio = (sum(x^2))^2 / sum(x^4)
    Normalized 
    """
    x = _validate_x(x)
    m = len(x)

    if m <= 1:
        return np.nan

    sum_sq = np.sum(x**2)
    sum_fourth = np.sum(x**4)

    if np.isclose(sum_sq, 0):
        return np.nan

    n_eff = sum_sq**2 / sum_fourth
    return (n_eff - 1) / (m - 1)


def _relative_consecutive_difference(x):
    """
        mean(
            2 * |x[t+1] - x[t]|
            / (|x[t+1]| + |x[t]|)
        )
    """
    x = _validate_x(x)

    x1 = x[:-1]
    x2 = x[1:]

    numerator = 2 * np.abs(x2 - x1)
    denominator = np.abs(x2) + np.abs(x1)

    local = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )
    return np.mean(local)


def _lag1_autocorrelation(x):
    """
    Pearson correlation between:

        x[0:N-1] and x[1:N]

    """
    x = _validate_x(x)

    x1 = x[:-1]
    x2 = x[1:]

    if np.isclose(np.std(x1), 0) or np.isclose(np.std(x2), 0):
        return np.nan
    return np.corrcoef(x1, x2)[0, 1]


