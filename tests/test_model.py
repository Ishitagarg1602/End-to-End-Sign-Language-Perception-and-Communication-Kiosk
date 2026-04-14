
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

import numpy as np
import pytest


class TestCNNLSTMArchitecture:
    """Tests for the CNN-LSTM model architecture."""

    def test_model_builds(self):
        """Model should build without errors."""
        from model.architecture import build_model
        model = build_model(num_classes=60)
        assert model is not None

    def test_input_shape(self):
        """Model should accept input shape (None, 30, 126)."""
        from model.architecture import build_model
        model = build_model(num_classes=60)
        assert model.input_shape == (None, 30, 126)

    def test_output_shape(self):
        """Model should output shape (None, 60)."""
        from model.architecture import build_model
        model = build_model(num_classes=60)
        assert model.output_shape == (None, 60)

    def test_custom_classes(self):
        """Model should work with different class counts."""
        from model.architecture import build_model
        model = build_model(num_classes=10)
        assert model.output_shape == (None, 10)

    def test_forward_pass(self):
        """Model should produce valid predictions on random input."""
        from model.architecture import build_model
        model = build_model(num_classes=60)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        dummy_input = np.random.randn(2, 30, 126).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        assert output.shape == (2, 60)
        # Softmax outputs should sum to ~1
        assert np.allclose(np.sum(output, axis=1), 1.0, atol=1e-5)

    def test_parameter_count_reasonable(self):
        """Model should have a reasonable number of parameters."""
        from model.architecture import build_model
        model = build_model(num_classes=60)
        param_count = model.count_params()
        # Should be between 100K and 10M parameters
        assert 100_000 < param_count < 10_000_000

    def test_model_summary(self):
        """get_model_summary should return a non-empty string."""
        from model.architecture import build_model, get_model_summary
        model = build_model(num_classes=60)
        summary = get_model_summary(model)
        assert len(summary) > 0
        assert 'ISL_CNN_LSTM_Attention' in summary


class TestPredictor:
    """Tests for the unified Predictor class."""

    def test_predictor_no_model(self):
        """Predictor should handle missing model gracefully."""
        from model.predict import Predictor
        predictor = Predictor(model_path='/nonexistent/path/model.pkl')
        dummy = np.random.randn(30, 126).astype(np.float32)
        result = predictor.predict(dummy)
        assert result['word'] == 'unknown'
        assert result['confidence'] == 0.0
        assert result['is_low_confidence'] is True

    def test_prediction_result_format(self):
        """Prediction result should have all required keys."""
        from model.predict import Predictor
        predictor = Predictor()
        dummy = np.random.randn(30, 126).astype(np.float32)
        result = predictor.predict(dummy)

        assert 'word' in result
        assert 'confidence' in result
        assert 'class_index' in result
        assert 'top3' in result
        assert 'is_low_confidence' in result
