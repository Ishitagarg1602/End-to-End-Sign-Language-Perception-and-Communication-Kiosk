
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    NUM_KEYPOINTS, NUM_HANDS, COORDS_PER_KEYPOINT, FEATURES_PER_FRAME,
    MAX_NUM_HANDS, MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE
)


class HandLandmarkExtractor:
    """
    Extracts hand landmarks from video frames using MediaPipe Hands.

    Features per frame: 21 keypoints × 2 hands × 3 coords = 126
    Layout: [left_hand_63_values, right_hand_63_values]

    Attributes:
        hands: MediaPipe Hands instance.
    """

    def __init__(self, static_mode: bool = False):
        """
        Initialize the MediaPipe Hands detector.

        Args:
            static_mode: If True, treats each frame independently (slower but
                         better for non-video sequences). False for real-time.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 126-dimensional feature vector from a single frame.

        Detects up to 2 hands and returns concatenated landmarks.
        If only one hand is detected, the other is zero-filled.
        If no hands are detected, returns None.

        Args:
            frame: BGR image from OpenCV (np.ndarray).

        Returns:
            NumPy array of shape (126,) with float32 values, or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        left_hand = np.zeros(NUM_KEYPOINTS * COORDS_PER_KEYPOINT, dtype=np.float32)
        right_hand = np.zeros(NUM_KEYPOINTS * COORDS_PER_KEYPOINT, dtype=np.float32)

        for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)

            label = handedness.classification[0].label
            if label == 'Left':
                left_hand = coords
            else:
                right_hand = coords

        return np.concatenate([left_hand, right_hand])

    def extract_with_drawing(self, frame: np.ndarray) -> tuple:
        """
        Extract landmarks and draw them on the frame for visualization.

        Args:
            frame: BGR image (will be modified in-place for drawing).

        Returns:
            Tuple of (features: Optional[np.ndarray], annotated_frame: np.ndarray).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, frame

        # Draw landmarks
        for hand_lm in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
            )

        # Extract features (same logic as extract())
        left_hand = np.zeros(NUM_KEYPOINTS * COORDS_PER_KEYPOINT, dtype=np.float32)
        right_hand = np.zeros(NUM_KEYPOINTS * COORDS_PER_KEYPOINT, dtype=np.float32)

        for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)

            label = handedness.classification[0].label
            if label == 'Left':
                left_hand = coords
            else:
                right_hand = coords

        features = np.concatenate([left_hand, right_hand])
        return features, frame

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
