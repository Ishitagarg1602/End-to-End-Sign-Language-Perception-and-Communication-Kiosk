"""
backend/preprocessing/smooth.py
================================
Butterworth low-pass filter for temporal smoothing of landmark sequences.

Applies a 3rd-order Butterworth filter to each feature dimension independently
along the time axis. This removes high-frequency jitter from MediaPipe hand
tracking while preserving the meaningful motion patterns of sign language.
"""

import numpy as np
from scipy.signal import butter, filtfilt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BUTTER_ORDER, BUTTER_CUTOFF, BUTTER_FS


def create_butterworth_filter(order: int = BUTTER_ORDER,
                               cutoff: float = BUTTER_CUTOFF,
                               fs: float = BUTTER_FS) -> tuple:
    """
    Create Butterworth low-pass filter coefficients.

    Args:
        order: Filter order (default: 3).
        cutoff: Cutoff frequency in Hz (default: 6.0).
        fs: Sampling frequency in Hz (default: 30.0).

    Returns:
        Tuple of (b, a) filter coefficient arrays.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Clamp to valid range (0, 1) exclusive for stability
    normal_cutoff = min(max(normal_cutoff, 0.01), 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def smooth_sequence(sequence: np.ndarray,
                     order: int = BUTTER_ORDER,
                     cutoff: float = BUTTER_CUTOFF,
                     fs: float = BUTTER_FS) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to smooth a landmark sequence.

    Filters each of the 126 feature dimensions independently along
    the time axis. Uses zero-phase filtering (filtfilt) to prevent
    phase distortion.

    Requires at least (3 × order + 1) frames for stable filtering.
    Sequences shorter than this minimum are returned unchanged.

    Args:
        sequence: Array of shape (num_frames, 126).
        order: Butterworth filter order.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.

    Returns:
        Smoothed sequence of the same shape.
    """
    num_frames = sequence.shape[0]
    min_frames = 3 * order + 1

    if num_frames < min_frames:
        return sequence.copy()

    b, a = create_butterworth_filter(order, cutoff, fs)
    smoothed = np.zeros_like(sequence)

    for col in range(sequence.shape[1]):
        try:
            smoothed[:, col] = filtfilt(b, a, sequence[:, col])
        except ValueError:
            # Fallback: keep original values if filtering fails
            smoothed[:, col] = sequence[:, col]

    return smoothed
