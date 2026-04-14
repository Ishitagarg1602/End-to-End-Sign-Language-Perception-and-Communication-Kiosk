"""
backend/preprocessing/normalize.py
====================================
Translation and scale normalization for hand landmarks.

Wrist Centering:
  Subtracts the wrist position (landmark 0) from all 21 keypoints
  per hand, making the representation translation-invariant.

Scale Normalization:
  Divides all coordinates by the Euclidean distance from wrist to
  middle finger MCP (landmark 9), making the representation
  scale-invariant across different hand sizes and camera distances.
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NUM_KEYPOINTS, COORDS_PER_KEYPOINT


# Middle finger MCP is landmark index 9
MCP_LANDMARK_INDEX = 9


def wrist_centering(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply wrist centering: subtract wrist (x,y,z) from all keypoints.

    For each hand (left: indices 0-62, right: indices 63-125):
      - Landmark 0 is the wrist (first 3 values)
      - Subtract wrist (x,y,z) from all 21 keypoints

    This makes landmarks translation-invariant — only relative positions matter.

    Args:
        landmarks: Array of shape (126,) — raw hand landmarks.

    Returns:
        Wrist-centered landmarks of shape (126,).
    """
    centered = landmarks.copy()
    hand_size = NUM_KEYPOINTS * COORDS_PER_KEYPOINT  # 63

    for hand_offset in [0, hand_size]:
        hand_data = centered[hand_offset:hand_offset + hand_size]

        # Skip zero-filled hands (not detected)
        if np.all(hand_data == 0):
            continue

        # Wrist is landmark 0: indices 0, 1, 2
        wrist_x = hand_data[0]
        wrist_y = hand_data[1]
        wrist_z = hand_data[2]

        # Subtract wrist from all 21 keypoints
        for kp in range(NUM_KEYPOINTS):
            idx = kp * COORDS_PER_KEYPOINT
            hand_data[idx] -= wrist_x
            hand_data[idx + 1] -= wrist_y
            hand_data[idx + 2] -= wrist_z

        centered[hand_offset:hand_offset + hand_size] = hand_data

    return centered


def scale_normalization(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks by wrist→middle_finger_MCP distance.

    For each hand, divides all coordinates by the Euclidean distance
    from the wrist (now at origin after centering) to the middle finger
    MCP joint (landmark 9). This normalizes for different hand sizes.

    Should be called AFTER wrist_centering.

    Args:
        landmarks: Wrist-centered array of shape (126,).

    Returns:
        Scale-normalized landmarks of shape (126,).
    """
    normalized = landmarks.copy()
    hand_size = NUM_KEYPOINTS * COORDS_PER_KEYPOINT  # 63

    for hand_offset in [0, hand_size]:
        hand_data = normalized[hand_offset:hand_offset + hand_size]

        # Skip zero-filled hands
        if np.all(hand_data == 0):
            continue

        # MCP landmark 9 coordinates (after wrist centering)
        mcp_idx = MCP_LANDMARK_INDEX * COORDS_PER_KEYPOINT
        mcp_x = hand_data[mcp_idx]
        mcp_y = hand_data[mcp_idx + 1]
        mcp_z = hand_data[mcp_idx + 2]

        scale = np.sqrt(mcp_x**2 + mcp_y**2 + mcp_z**2)

        # Avoid division by zero
        if scale < 1e-6:
            continue

        hand_data /= scale
        normalized[hand_offset:hand_offset + hand_size] = hand_data

    return normalized


def normalize_frame(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply full normalization pipeline to a single frame.

    Convenience function that applies wrist centering then scale normalization.

    Args:
        landmarks: Raw array of shape (126,).

    Returns:
        Fully normalized array of shape (126,).
    """
    normalized = wrist_centering(landmarks)
    normalized = scale_normalization(normalized)
    return normalized


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply normalization to every frame in a sequence.

    Args:
        sequence: Array of shape (N, 126).

    Returns:
        Normalized array of shape (N, 126).
    """
    result = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        result[i] = normalize_frame(sequence[i])
    return result
