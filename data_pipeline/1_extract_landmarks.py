
import os
import sys
import argparse
import glob
import time
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_tasks
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET_FRAMES = 30          # Standardized sequence length
NUM_KEYPOINTS = 21          # MediaPipe hand keypoints per hand
NUM_HANDS = 2               # Left + Right
FEATURES_PER_FRAME = NUM_KEYPOINTS * NUM_HANDS * 3  # 126
BUTTER_ORDER = 3            # Butterworth filter order
BUTTER_CUTOFF = 6.0         # Cutoff frequency (Hz)
BUTTER_FS = 30.0            # Sampling frequency (assumed 30 FPS)
MIN_DETECTION_RATIO = 0.5   # Flag videos with <50% hand detection

# Path to hand landmarker model (downloaded from Google)
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'hand_landmarker.task'


def create_butterworth_filter(order: int = BUTTER_ORDER,
                               cutoff: float = BUTTER_CUTOFF,
                               fs: float = BUTTER_FS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Butterworth low-pass filter coefficients.

    Args:
        order: Filter order.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.

    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Clamp to valid range (0, 1) exclusive
    normal_cutoff = min(max(normal_cutoff, 0.01), 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def create_hand_landmarker() -> vision.HandLandmarker:
    """
    Create a MediaPipe Hand Landmarker using the Tasks API.

    Returns:
        HandLandmarker instance.
    """
    if not MODEL_PATH.exists():
        print(f"[ERROR] Hand landmarker model not found at {MODEL_PATH}")
        print("Download it with:")
        print("  python -c \"import urllib.request; "
              "urllib.request.urlretrieve("
              "'https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/"
              "hand_landmarker.task', 'models/hand_landmarker.task')\"")
        sys.exit(1)

    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(frame: np.ndarray,
                            detector: vision.HandLandmarker) -> Optional[np.ndarray]:
    """
    Extract 126-dimensional feature vector from a single frame.

    Uses MediaPipe HandLandmarker Tasks API (IMAGE mode) to detect up to 2 hands.
    Each hand contributes 21 keypoints × 3 coordinates (x, y, z) = 63 values.
    Missing hands are zero-filled.

    Args:
        frame: BGR image from OpenCV.
        detector: MediaPipe HandLandmarker instance.

    Returns:
        NumPy array of shape (126,) or None if no hands detected.
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None

    # Initialize both hands as zeros
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
        # Extract 21 keypoints × 3 coords = 63 values
        coords = []
        for lm in hand_landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)

        # Classify as left or right from handedness
        if hand_idx < len(results.handedness):
            label = results.handedness[hand_idx][0].category_name
        else:
            label = 'Right' if hand_idx == 0 else 'Left'

        if label == 'Left':
            left_hand = coords
        else:
            right_hand = coords

    # Concatenate: left (0-62) + right (63-125) = 126
    features = np.concatenate([left_hand, right_hand])
    return features


def wrist_centering(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply wrist centering: subtract wrist position from all keypoints.

    For each hand, landmark 0 is the wrist. We subtract the wrist's (x, y, z)
    from all 21 keypoints of that hand. This makes landmarks translation-invariant.

    Args:
        landmarks: Array of shape (126,).

    Returns:
        Wrist-centered landmarks of shape (126,).
    """
    centered = landmarks.copy()

    for hand_offset in [0, 63]:  # Left hand starts at 0, right at 63
        hand_data = centered[hand_offset:hand_offset + 63]

        # Check if hand is all zeros (not detected)
        if np.all(hand_data == 0):
            continue

        # Wrist is the first landmark (indices 0, 1, 2)
        wrist_x, wrist_y, wrist_z = hand_data[0], hand_data[1], hand_data[2]

        # Subtract wrist from all 21 keypoints
        for i in range(21):
            idx = i * 3
            hand_data[idx] -= wrist_x
            hand_data[idx + 1] -= wrist_y
            hand_data[idx + 2] -= wrist_z

        centered[hand_offset:hand_offset + 63] = hand_data

    return centered


def scale_normalization(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply scale normalization: divide all coords by wrist→middle_finger_base distance.

    This makes the landmarks scale-invariant (different hand sizes produce
    similar values). Middle finger MCP (landmark 9) is used as reference.

    Args:
        landmarks: Wrist-centered array of shape (126,).

    Returns:
        Scale-normalized landmarks of shape (126,).
    """
    normalized = landmarks.copy()

    for hand_offset in [0, 63]:
        hand_data = normalized[hand_offset:hand_offset + 63]

        # Check if hand is all zeros (not detected)
        if np.all(hand_data == 0):
            continue

        # Middle finger MCP is landmark 9 (after wrist centering, wrist is at origin)
        # Landmark 9 coords: indices 27, 28, 29
        mcp_x, mcp_y, mcp_z = hand_data[27], hand_data[28], hand_data[29]
        scale = np.sqrt(mcp_x**2 + mcp_y**2 + mcp_z**2)

        # Avoid division by zero
        if scale < 1e-6:
            continue

        hand_data /= scale
        normalized[hand_offset:hand_offset + 63] = hand_data

    return normalized


def apply_butterworth_filter(sequence: np.ndarray,
                              b: np.ndarray,
                              a: np.ndarray) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to smooth the landmark sequence.

    Each of the 126 feature dimensions is filtered independently along
    the time axis to remove jitter.

    Args:
        sequence: Array of shape (num_frames, 126).
        b, a: Butterworth filter coefficients.

    Returns:
        Smoothed sequence of the same shape.
    """
    num_frames = sequence.shape[0]

    # Need at least 3 * filter_order frames for filtfilt
    min_frames = 3 * BUTTER_ORDER + 1
    if num_frames < min_frames:
        return sequence  # Too short to filter meaningfully

    smoothed = np.zeros_like(sequence)
    for col in range(sequence.shape[1]):
        try:
            smoothed[:, col] = filtfilt(b, a, sequence[:, col])
        except ValueError:
            # If filtering fails, keep original
            smoothed[:, col] = sequence[:, col]

    return smoothed


def interpolate_to_target_frames(sequence: np.ndarray,
                                  target: int = TARGET_FRAMES) -> np.ndarray:
    """
    Interpolate or pad the sequence to exactly `target` frames.

    Uses linear interpolation via SciPy to standardize sequence length.
    If sequence has only 1 frame, it is repeated `target` times.

    Args:
        sequence: Array of shape (num_frames, 126).
        target: Desired number of frames (default 30).

    Returns:
        Array of shape (target, 126).
    """
    num_frames = sequence.shape[0]

    if num_frames == 0:
        return np.zeros((target, FEATURES_PER_FRAME), dtype=np.float32)

    if num_frames == 1:
        return np.repeat(sequence, target, axis=0)

    if num_frames == target:
        return sequence

    # Create interpolation function for each feature dimension
    x_original = np.linspace(0, 1, num_frames)
    x_target = np.linspace(0, 1, target)

    interpolator = interp1d(x_original, sequence, axis=0, kind='linear')
    interpolated = interpolator(x_target)

    return interpolated.astype(np.float32)


def process_video(video_path: str,
                   detector: vision.HandLandmarker,
                   butter_b: np.ndarray,
                   butter_a: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Process a single video file through the full landmark extraction pipeline.

    Pipeline: OpenCV → MediaPipe Hands → wrist centering → scale normalization
              → Butterworth filter → interpolation to 30 frames.

    Args:
        video_path: Path to the .mp4 file.
        detector: MediaPipe HandLandmarker instance.
        butter_b, butter_a: Butterworth filter coefficients.

    Returns:
        Tuple of (landmarks array shape (30, 126), detection_ratio).
        Returns (None, 0.0) if video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return None, 0.0

    raw_landmarks = []
    total_frames = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Extract hand landmarks from this frame (IMAGE mode, no timestamp needed)
        features = extract_hand_landmarks(frame, detector)

        if features is not None:
            detected_frames += 1
            # Apply per-frame preprocessing
            features = wrist_centering(features)
            features = scale_normalization(features)
            raw_landmarks.append(features)
        else:
            # Append zeros for frames with no detection (will be smoothed)
            raw_landmarks.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))

    cap.release()

    if total_frames == 0:
        return None, 0.0

    detection_ratio = detected_frames / total_frames
    sequence = np.array(raw_landmarks, dtype=np.float32)

    # Apply Butterworth smoothing
    sequence = apply_butterworth_filter(sequence, butter_b, butter_a)

    # Interpolate to exactly TARGET_FRAMES
    sequence = interpolate_to_target_frames(sequence, TARGET_FRAMES)

    return sequence, detection_ratio


def main():
    """
    Main entry point: iterate over dataset folder, extract landmarks, and save.

    Expects dataset structure:
        <input_dir>/<word>/*.mp4

    Outputs:
        mvp/landmarks/<word>/<filename>.npy  (shape: 30, 126)
    """
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from ISL sign language videos."
    )
    parser.add_argument(
        '--input', type=str, default='./mvp/dataset',
        help='Path to dataset root folder (default: ./mvp/dataset)'
    )
    parser.add_argument(
        '--output', type=str, default='./mvp/landmarks',
        help='Path to output landmarks folder (default: ./mvp/landmarks)'
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"[ERROR] Dataset directory not found: {input_dir}")
        print("Please place your dataset in mvp/dataset/ with subfolders per word.")
        sys.exit(1)

    # Discover all word folders
    word_folders = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if len(word_folders) == 0:
        print(f"[ERROR] No word folders found in {input_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  ISL Landmark Extraction Pipeline")
    print(f"{'='*60}")
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Word folders     : {len(word_folders)}")
    print(f"  Target frames    : {TARGET_FRAMES}")
    print(f"  Features/frame   : {FEATURES_PER_FRAME}")
    print(f"{'='*60}\n")

    # Create Butterworth filter coefficients
    butter_b, butter_a = create_butterworth_filter()

    # Initialize MediaPipe Hand Landmarker (Tasks API)
    print(f"  Loading MediaPipe Hand Landmarker model...")
    detector = create_hand_landmarker()
    print(f"  Model loaded from: {MODEL_PATH}\n")

    # Statistics
    total_videos = 0
    successful = 0
    failed = 0
    low_detection = []
    start_time = time.time()

    for word_folder in word_folders:
        word = word_folder.name
        videos = sorted(glob.glob(str(word_folder / "*.mp4")))

        if len(videos) == 0:
            print(f"  [WARN] No .mp4 files in {word_folder}")
            continue

        # Create output subfolder (keep original folder name)
        word_output_dir = output_dir / word
        word_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Processing '{word}' ({len(videos)} videos)...")

        for video_path in videos:
            total_videos += 1
            filename = Path(video_path).stem

            # Process video through full pipeline
            landmarks, detection_ratio = process_video(
                video_path, detector, butter_b, butter_a
            )

            if landmarks is None:
                failed += 1
                print(f"    ✗ {filename} — FAILED (cannot read)")
                continue

            # Save as .npy
            output_path = word_output_dir / f"{filename}.npy"
            np.save(str(output_path), landmarks)
            successful += 1

            # Flag low detection
            status = "✓"
            if detection_ratio < MIN_DETECTION_RATIO:
                low_detection.append((filename, word, detection_ratio))
                status = "⚠"

            print(f"    {status} {filename} — {detection_ratio:.0%} detection, "
                  f"shape {landmarks.shape}")

    # Close detector
    detector.close()

    elapsed = time.time() - start_time

    # ─── Extraction Report ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EXTRACTION REPORT")
    print(f"{'='*60}")
    print(f"  Total videos processed : {total_videos}")
    print(f"  Successful             : {successful}")
    print(f"  Failed                 : {failed}")
    print(f"  Success rate           : {successful/max(total_videos,1):.1%}")
    print(f"  Time elapsed           : {elapsed:.1f}s")

    if low_detection:
        print(f"\n  ⚠ Videos with <{MIN_DETECTION_RATIO:.0%} hand detection:")
        for fname, word, ratio in low_detection:
            print(f"    - {word}/{fname}: {ratio:.0%}")

    print(f"\n  Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
