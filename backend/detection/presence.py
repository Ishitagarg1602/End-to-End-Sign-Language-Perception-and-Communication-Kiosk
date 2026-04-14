
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ZONE_X_MIN, ZONE_X_MAX, ZONE_Y_MIN, ZONE_Y_MAX,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE, MEDIAPIPE_MIN_TRACKING_CONFIDENCE
)


class PresenceDetector:
    def __init__(self):
        """Initialize the MediaPipe Pose detector."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lightweight for real-time
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        self.zone_bounds = (ZONE_X_MIN, ZONE_X_MAX, ZONE_Y_MIN, ZONE_Y_MAX)

    def detect(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Check if a person is present in the interaction zone.

        Args:
            frame: BGR image from OpenCV (np.ndarray).

        Returns:
            Tuple of (is_in_zone: bool, num_people_detected: int).
            Note: MediaPipe Pose only detects one person, so num_people is 0 or 1.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return False, 0

        # Check if nose landmark is within the interaction zone
        nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        x_min, x_max, y_min, y_max = self.zone_bounds

        in_zone = (x_min <= nose.x <= x_max and y_min <= nose.y <= y_max)

        return in_zone, 1

    def draw_zone(self, frame: np.ndarray, in_zone: bool) -> np.ndarray:
        """
        Draw the interaction zone rectangle on the frame for visualization.

        Args:
            frame: BGR image from OpenCV.
            in_zone: Whether a person is currently in the zone.

        Returns:
            Frame with zone overlay drawn.
        """
        h, w = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.zone_bounds

        pt1 = (int(w * x_min), int(h * y_min))
        pt2 = (int(w * x_max), int(h * y_max))

        color = (0, 255, 0) if in_zone else (0, 0, 255)  # Green if in zone, red if not
        cv2.rectangle(frame, pt1, pt2, color, 2)

        label = "IN ZONE" if in_zone else "NO PERSON"
        cv2.putText(frame, label, (pt1[0] + 10, pt1[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
