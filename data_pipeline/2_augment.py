import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET_FRAMES = 30
FEATURES_PER_FRAME = 126
LANDMARKS_DIR = Path('./mvp/landmarks')
AUGMENTED_DIR = Path('./mvp/augmented')
SPLITS_DIR = Path('./mvp/splits')
RANDOM_SEED = 42


def mirror_x(sequence: np.ndarray) -> np.ndarray:
    """
    Mirror all X coordinates: x' = 1 - x (simulates left-handed signer).

    In the 126-feature vector, X coordinates are at indices 0, 3, 6, ... 
    for each hand (every 3rd starting from 0 within each hand block).

    Args:
        sequence: Array of shape (30, 126).

    Returns:
        Mirrored sequence of shape (30, 126).
    """
    mirrored = sequence.copy()
    # X coordinate indices for both hands (every 3rd value)
    for hand_offset in [0, 63]:
        for kp in range(21):
            x_idx = hand_offset + kp * 3
            mirrored[:, x_idx] = -mirrored[:, x_idx]  # Negate X (wrist-centered)
    return mirrored


def add_noise(sequence: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add Gaussian noise to all landmarks.

    Args:
        sequence: Array of shape (30, 126).
        sigma: Standard deviation of Gaussian noise.

    Returns:
        Noisy sequence of shape (30, 126).
    """
    noise = np.random.normal(0, sigma, sequence.shape).astype(np.float32)
    return sequence + noise


def stretch_sequence(sequence: np.ndarray, factor: float) -> np.ndarray:
    """
    Time-stretch a sequence by the given factor, then resize to TARGET_FRAMES.

    factor > 1.0 = slow down (expand then trim)
    factor < 1.0 = speed up (compress then pad)

    Args:
        sequence: Array of shape (30, 126).
        factor: Stretch factor.

    Returns:
        Stretched sequence of shape (30, 126).
    """
    num_frames = sequence.shape[0]
    new_length = max(2, int(num_frames * factor))

    x_original = np.linspace(0, 1, num_frames)
    x_new = np.linspace(0, 1, new_length)

    interpolator = interp1d(x_original, sequence, axis=0, kind='linear',
                            fill_value='extrapolate')
    stretched = interpolator(x_new)

    # Resize back to TARGET_FRAMES
    if new_length == TARGET_FRAMES:
        return stretched.astype(np.float32)

    x_stretched = np.linspace(0, 1, new_length)
    x_target = np.linspace(0, 1, TARGET_FRAMES)

    final_interp = interp1d(x_stretched, stretched, axis=0, kind='linear',
                            fill_value='extrapolate')
    result = final_interp(x_target)
    return result.astype(np.float32)


def frame_dropout(sequence: np.ndarray, num_drop: int) -> np.ndarray:
    """
    Randomly drop frames and interpolate to fill gaps.

    Args:
        sequence: Array of shape (30, 126).
        num_drop: Number of frames to drop.

    Returns:
        Sequence with dropped frames interpolated, shape (30, 126).
    """
    num_frames = sequence.shape[0]
    if num_drop >= num_frames - 2:
        return sequence.copy()

    # Choose random frames to drop (never drop first or last)
    drop_indices = np.random.choice(
        range(1, num_frames - 1), size=num_drop, replace=False
    )
    keep_indices = sorted(set(range(num_frames)) - set(drop_indices))

    kept_frames = sequence[keep_indices]

    # Interpolate back to TARGET_FRAMES
    x_kept = np.linspace(0, 1, len(keep_indices))
    x_target = np.linspace(0, 1, TARGET_FRAMES)

    interpolator = interp1d(x_kept, kept_frames, axis=0, kind='linear',
                            fill_value='extrapolate')
    result = interpolator(x_target)
    return result.astype(np.float32)


def rotate_z(sequence: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate hand landmarks around the Z axis by the given angle.

    Only affects X and Y coordinates:
      x' = x·cos(θ) - y·sin(θ)
      y' = x·sin(θ) + y·cos(θ)

    Args:
        sequence: Array of shape (30, 126).
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated sequence of shape (30, 126).
    """
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    rotated = sequence.copy()

    for hand_offset in [0, 63]:
        for kp in range(21):
            x_idx = hand_offset + kp * 3
            y_idx = hand_offset + kp * 3 + 1

            x_vals = rotated[:, x_idx].copy()
            y_vals = rotated[:, y_idx].copy()

            rotated[:, x_idx] = x_vals * cos_t - y_vals * sin_t
            rotated[:, y_idx] = x_vals * sin_t + y_vals * cos_t

    return rotated


def generate_augmentations(sequence: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate all 14 augmentation variants for a single landmark sequence.

    Args:
        sequence: Original landmark array of shape (30, 126).

    Returns:
        List of (variant_name, augmented_array) tuples.
    """
    variants = []

    # 1. Original
    variants.append(('orig', sequence.copy()))

    # 2. Mirror
    variants.append(('mirror', mirror_x(sequence)))

    # 3-5. Noise variants
    variants.append(('noise_s', add_noise(sequence, 0.005)))
    variants.append(('noise_m', add_noise(sequence, 0.01)))
    variants.append(('noise_l', add_noise(sequence, 0.02)))

    # 6-7. Stretch variants
    variants.append(('stretch_slow', stretch_sequence(sequence, 1.3)))
    variants.append(('stretch_fast', stretch_sequence(sequence, 0.7)))

    # 8-9. Frame dropout
    variants.append(('dropout_3', frame_dropout(sequence, 3)))
    variants.append(('dropout_5', frame_dropout(sequence, 5)))

    # 10-11. Rotation
    variants.append(('rot_pos', rotate_z(sequence, 10.0)))
    variants.append(('rot_neg', rotate_z(sequence, -10.0)))

    # 12. Mirror + noise
    mirrored = mirror_x(sequence)
    variants.append(('mirror_noise', add_noise(mirrored, 0.008)))

    # 13. Stretch + noise
    stretched = stretch_sequence(sequence, 1.3)
    variants.append(('stretch_noise', add_noise(stretched, 0.005)))

    # 14. Combined: random combo of mirror, noise, stretch
    combined = sequence.copy()
    if np.random.random() > 0.5:
        combined = mirror_x(combined)
    combined = add_noise(combined, np.random.uniform(0.003, 0.015))
    factor = np.random.uniform(0.8, 1.2)
    combined = stretch_sequence(combined, factor)
    variants.append(('combined', combined))

    return variants


def main():
    """
    Main entry point: augment all landmark .npy files and save results.

    Reads from mvp/landmarks/, writes to mvp/augmented/.
    Also saves labels.npy and classes.json.
    """
    np.random.seed(RANDOM_SEED)

    if not LANDMARKS_DIR.exists():
        print(f"[ERROR] Landmarks directory not found: {LANDMARKS_DIR}")
        print("Please run 1_extract_landmarks.py first.")
        sys.exit(1)

    # Discover word folders
    word_folders = sorted([
        d for d in LANDMARKS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if len(word_folders) == 0:
        print(f"[ERROR] No word folders found in {LANDMARKS_DIR}")
        sys.exit(1)

    # Build class list — convert folder names to snake_case for consistency
    # e.g., "Account Blocked" → "account_blocked", "Good Morning" → "good_morning"
    classes = [d.name.lower().replace(' ', '_') for d in word_folders]

    print(f"{'='*60}")
    print(f"  ISL Data Augmentation Pipeline")
    print(f"{'='*60}")
    print(f"  Input directory  : {LANDMARKS_DIR}")
    print(f"  Output directory : {AUGMENTED_DIR}")
    print(f"  Classes found    : {len(classes)}")
    print(f"  Variants per file: 14")
    print(f"{'='*60}\n")

    all_sequences = []
    all_labels = []
    total_original = 0
    total_augmented = 0
    start_time = time.time()

    for class_idx, word_folder in enumerate(word_folders):
        word = word_folder.name
        npy_files = sorted(word_folder.glob("*.npy"))

        if len(npy_files) == 0:
            print(f"  [WARN] No .npy files in {word_folder}")
            continue

        # Create output subfolder
        word_output_dir = AUGMENTED_DIR / word
        word_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Augmenting '{word}' ({len(npy_files)} files)...")

        for npy_path in npy_files:
            total_original += 1
            filename = npy_path.stem
            sequence = np.load(str(npy_path))

            # Validate shape
            if sequence.shape != (TARGET_FRAMES, FEATURES_PER_FRAME):
                print(f"    [WARN] Unexpected shape {sequence.shape} for {npy_path.name}")
                continue

            # Generate 14 variants
            variants = generate_augmentations(sequence)

            for variant_name, augmented_seq in variants:
                total_augmented += 1

                # Save individual augmented file
                out_filename = f"{filename}_{variant_name}.npy"
                np.save(str(word_output_dir / out_filename), augmented_seq)

                # Collect for bulk arrays
                all_sequences.append(augmented_seq)
                all_labels.append(class_idx)

    elapsed = time.time() - start_time

    # Save bulk arrays
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_sequences = np.array(all_sequences, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    np.save(str(AUGMENTED_DIR / 'all_sequences.npy'), all_sequences)
    np.save(str(AUGMENTED_DIR / 'labels.npy'), all_labels)

    # Save classes.json
    classes_path = SPLITS_DIR / 'classes.json'
    with open(str(classes_path), 'w') as f:
        json.dump(classes, f, indent=2)

    # ─── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AUGMENTATION REPORT")
    print(f"{'='*60}")
    print(f"  Original files      : {total_original}")
    print(f"  Augmented samples   : {total_augmented}")
    print(f"  Augmentation ratio  : {total_augmented/max(total_original,1):.1f}x")
    print(f"  Total array shape   : {all_sequences.shape}")
    print(f"  Labels shape        : {all_labels.shape}")
    print(f"  Classes saved to    : {classes_path}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
