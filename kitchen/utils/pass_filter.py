from typing import Optional
import scipy.signal as signal
import numpy as np


def compute_fs(xs: np.ndarray) -> float:
    dt = np.median(np.diff(xs))
    if dt == 0:
        raise ValueError("Time array has zero median difference.")
    return float(1 / dt)


def low_pass(xs: np.ndarray, ys: np.ndarray, cutoff: float, fs: Optional[float] = None) -> np.ndarray:
    if fs is None:
        fs = compute_fs(xs)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)  # type: ignore
    filtered_ys = signal.filtfilt(b, a, ys)
    return filtered_ys


def high_pass(xs: np.ndarray, ys: np.ndarray, cutoff: float, fs: Optional[float] = None) -> np.ndarray:
    if fs is None:
        fs = compute_fs(xs)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)  # type: ignore
    filtered_ys = signal.filtfilt(b, a, ys)
    return filtered_ys