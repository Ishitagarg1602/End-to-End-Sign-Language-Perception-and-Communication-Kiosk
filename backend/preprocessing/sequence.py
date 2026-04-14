"""
backend/preprocessing/sequence.py
==================================
Sequence standardization: pad, trim, or interpolate to exactly 30 frames.

All sign language sequences must be standardized to the same length (30 frames)
before being fed to the CNN-LSTM model. This module uses SciPy linear
interpolation to resize sequences while preserving temporal dynamics.
"""

import numpy as np
from scipy.interpolate import interp1d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_FRAMES, FEATURES_PER_FRAME


def interpolate_to_target(sequence: np.ndarray,
                           target: int = TARGET_FRAMES) -> np.ndarray:
    """
    Interpolate or pad a landmark sequence to exactly `target` frames.

    Uses linear interpolation via SciPy to smoothly resize the temporal dimension.

    - If sequence length == target: returned as-is
    - If sequence length > target: downsampled via interpolation
    - If sequence length < target: upsampled via interpolation
    - If sequence length == 0: returns zeros
    - If sequence length == 1: repeats the single frame

    Args:
        sequence: Array of shape (num_frames, 126).
        target: Desired number of frames (default: 30).

    Returns:
        Array of shape (target, 126) with float32 dtype.
    """
    num_frames = sequence.shape[0]

    # Edge cases
    if num_frames == 0:
        return np.zeros((target, FEATURES_PER_FRAME), dtype=np.float32)

    if num_frames == 1:
        return np.repeat(sequence, target, axis=0).astype(np.float32)

    if num_frames == target:
        return sequence.astype(np.float32)

    # Linear interpolation
    x_original = np.linspace(0, 1, num_frames)
    x_target = np.linspace(0, 1, target)

    interpolator = interp1d(x_original, sequence, axis=0, kind='linear')
    result = interpolator(x_target)

    return result.astype(np.float32)


def pad_sequence(sequence: np.ndarray,
                  target: int = TARGET_FRAMES,
                  pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or trim a sequence to exactly `target` frames using zero-padding.

    Unlike interpolation, this simply adds zeros at the end (for short sequences)
    or truncates from the end (for long sequences). Useful when interpolation
    is not desired.

    Args:
        sequence: Array of shape (num_frames, features).
        target: Desired number of frames.
        pad_value: Value to use for padding (default: 0.0).

    Returns:
        Array of shape (target, features).
    """
    num_frames = sequence.shape[0]
    features = sequence.shape[1] if len(sequence.shape) > 1 else FEATURES_PER_FRAME

    if num_frames == target:
        return sequence.astype(np.float32)

    if num_frames > target:
        # Trim from the end
        return sequence[:target].astype(np.float32)

    # Pad with pad_value
    padded = np.full((target, features), pad_value, dtype=np.float32)
    padded[:num_frames] = sequence
    return padded


def standardize_sequence(sequence: np.ndarray,
                          target: int = TARGET_FRAMES,
                          method: str = 'interpolate') -> np.ndarray:
    """
    Standardize a sequence to the target length using the specified method.

    Args:
        sequence: Array of shape (num_frames, 126).
        target: Desired number of frames (default: 30).
        method: Either 'interpolate' (default) or 'pad'.

    Returns:
        Array of shape (target, 126).
    """
    if method == 'interpolate':
        return interpolate_to_target(sequence, target)
    elif method == 'pad':
        return pad_sequence(sequence, target)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'interpolate' or 'pad'.")
