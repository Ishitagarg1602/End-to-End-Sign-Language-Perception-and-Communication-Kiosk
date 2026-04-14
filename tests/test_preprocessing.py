import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

import numpy as np
import pytest


class TestWristCentering:
    """Tests for wrist centering normalization."""

    def test_wrist_becomes_zero(self):
        """After centering, wrist (landmark 0) should be at origin."""
        from preprocessing.normalize import wrist_centering

        # Create landmarks with known wrist position
        landmarks = np.random.rand(126).astype(np.float32)
        landmarks[0] = 0.5   # left wrist x
        landmarks[1] = 0.3   # left wrist y
        landmarks[2] = 0.1   # left wrist z

        centered = wrist_centering(landmarks)

        # Left wrist should now be (0, 0, 0)
        assert abs(centered[0]) < 1e-6
        assert abs(centered[1]) < 1e-6
        assert abs(centered[2]) < 1e-6

    def test_preserves_relative_positions(self):
        """Centering should preserve relative distances between landmarks."""
        from preprocessing.normalize import wrist_centering

        landmarks = np.zeros(126, dtype=np.float32)
        # Set left hand: wrist at (1, 2, 0), landmark 1 at (1.5, 2.5, 0)
        landmarks[0], landmarks[1], landmarks[2] = 1.0, 2.0, 0.0
        landmarks[3], landmarks[4], landmarks[5] = 1.5, 2.5, 0.0

        centered = wrist_centering(landmarks)

        # Landmark 1 should be at (0.5, 0.5, 0.0) relative to wrist
        assert abs(centered[3] - 0.5) < 1e-6
        assert abs(centered[4] - 0.5) < 1e-6
        assert abs(centered[5] - 0.0) < 1e-6

    def test_zero_hand_unchanged(self):
        """Zero-filled hand (not detected) should remain zeros."""
        from preprocessing.normalize import wrist_centering

        landmarks = np.zeros(126, dtype=np.float32)
        # Left hand all zeros, right hand has data
        landmarks[63:] = np.random.rand(63).astype(np.float32)

        centered = wrist_centering(landmarks)
        assert np.all(centered[:63] == 0)

    def test_output_shape(self):
        """Output should have same shape as input."""
        from preprocessing.normalize import wrist_centering

        landmarks = np.random.rand(126).astype(np.float32)
        centered = wrist_centering(landmarks)
        assert centered.shape == (126,)


class TestScaleNormalization:
    """Tests for scale normalization."""

    def test_output_shape(self):
        """Output shape should match input."""
        from preprocessing.normalize import scale_normalization

        landmarks = np.random.rand(126).astype(np.float32)
        normalized = scale_normalization(landmarks)
        assert normalized.shape == (126,)

    def test_zero_hand_unchanged(self):
        """Zero-filled hand should remain zeros."""
        from preprocessing.normalize import scale_normalization

        landmarks = np.zeros(126, dtype=np.float32)
        normalized = scale_normalization(landmarks)
        assert np.all(normalized == 0)


class TestSequenceInterpolation:
    """Tests for sequence standardization."""

    def test_interpolate_longer_to_target(self):
        """Longer sequences should be downsampled."""
        from preprocessing.sequence import interpolate_to_target

        sequence = np.random.rand(50, 126).astype(np.float32)
        result = interpolate_to_target(sequence, target=30)
        assert result.shape == (30, 126)

    def test_interpolate_shorter_to_target(self):
        """Shorter sequences should be upsampled."""
        from preprocessing.sequence import interpolate_to_target

        sequence = np.random.rand(15, 126).astype(np.float32)
        result = interpolate_to_target(sequence, target=30)
        assert result.shape == (30, 126)

    def test_exact_length_unchanged(self):
        """Sequences at target length should pass through."""
        from preprocessing.sequence import interpolate_to_target

        sequence = np.random.rand(30, 126).astype(np.float32)
        result = interpolate_to_target(sequence, target=30)
        assert result.shape == (30, 126)
        assert np.allclose(result, sequence)

    def test_empty_sequence(self):
        """Empty sequence should return zeros."""
        from preprocessing.sequence import interpolate_to_target

        sequence = np.zeros((0, 126), dtype=np.float32)
        result = interpolate_to_target(sequence, target=30)
        assert result.shape == (30, 126)
        assert np.all(result == 0)

    def test_single_frame(self):
        """Single frame should be repeated."""
        from preprocessing.sequence import interpolate_to_target

        sequence = np.ones((1, 126), dtype=np.float32)
        result = interpolate_to_target(sequence, target=30)
        assert result.shape == (30, 126)
        assert np.all(result == 1)


class TestSmoothing:
    """Tests for Butterworth smoothing."""

    def test_output_shape_preserved(self):
        """Smoothing should preserve array shape."""
        from preprocessing.smooth import smooth_sequence

        sequence = np.random.rand(30, 126).astype(np.float32)
        smoothed = smooth_sequence(sequence)
        assert smoothed.shape == (30, 126)

    def test_short_sequence_unchanged(self):
        """Very short sequences should be returned unchanged."""
        from preprocessing.smooth import smooth_sequence

        sequence = np.random.rand(5, 126).astype(np.float32)
        smoothed = smooth_sequence(sequence)
        assert smoothed.shape == (5, 126)

    def test_constant_input_unchanged(self):
        """Constant input should not change after smoothing."""
        from preprocessing.smooth import smooth_sequence

        sequence = np.ones((30, 126), dtype=np.float32) * 0.5
        smoothed = smooth_sequence(sequence)
        assert np.allclose(smoothed, 0.5, atol=1e-5)


class TestNormalizeFrame:
    """Tests for the combined normalize_frame function."""

    def test_full_pipeline(self):
        """normalize_frame should apply both centering and scaling."""
        from preprocessing.normalize import normalize_frame

        landmarks = np.random.rand(126).astype(np.float32)
        result = normalize_frame(landmarks)
        assert result.shape == (126,)

    def test_normalize_sequence(self):
        """normalize_sequence should process all frames."""
        from preprocessing.normalize import normalize_sequence

        sequence = np.random.rand(30, 126).astype(np.float32)
        result = normalize_sequence(sequence)
        assert result.shape == (30, 126)
