
import sys
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLASSES_PATH, LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
)

ML_DIR = Path(__file__).resolve().parent.parent.parent / 'ml'
MVP_DIR = Path(__file__).resolve().parent.parent.parent / 'mvp'
CNN_LSTM_PATH = ML_DIR / 'saved_models' / 'best_model.h5'
KNN_PATH = MVP_DIR / 'model.pkl'


class Predictor:
    

    def __init__(self, model_path: Optional[str] = None):
        
        self.model = None
        self.model_type = None
        self.classes = []

        # Load class names
        if CLASSES_PATH.exists():
            with open(str(CLASSES_PATH), 'r') as f:
                self.classes = json.load(f)

        # Load model
        if model_path:
            self._load_model(Path(model_path))
        else:
            self._auto_load()

    def _auto_load(self):
        """Auto-select and load the best available model."""
        if CNN_LSTM_PATH.exists():
            self._load_model(CNN_LSTM_PATH)
        elif KNN_PATH.exists():
            self._load_model(KNN_PATH)
        else:
            print("[WARNING] No model found. Predictions will return defaults.")

    def _load_model(self, path: Path):
        """
        Load a model from the given path.

        Args:
            path: Path to the model file (.h5 or .pkl).
        """
        suffix = path.suffix.lower()

        if suffix == '.h5':
            import tensorflow as tf
            
            # Universal Keras 3 .h5 deserialization bug patch
            def clean_layer(LayerClass, bad_kwargs):
                class CustomLayer(LayerClass):
                    def __init__(self, **kwargs):
                        for k in bad_kwargs:
                            kwargs.pop(k, None)
                        super().__init__(**kwargs)
                return CustomLayer

            custom_objects = {
                'BatchNormalization': clean_layer(tf.keras.layers.BatchNormalization, ['renorm', 'renorm_clipping', 'renorm_momentum']),
                'MultiHeadAttention': clean_layer(tf.keras.layers.MultiHeadAttention, ['use_gate']),
                'Dense': clean_layer(tf.keras.layers.Dense, ['quantization_config']),
                'Conv1D': clean_layer(tf.keras.layers.Conv1D, ['quantization_config']),
                'LSTM': clean_layer(tf.keras.layers.LSTM, ['quantization_config']),
                'Bidirectional': clean_layer(tf.keras.layers.Bidirectional, ['quantization_config']),
                'Dropout': clean_layer(tf.keras.layers.Dropout, ['quantization_config']),
                'LayerNormalization': clean_layer(tf.keras.layers.LayerNormalization, ['quantization_config']),
                'MaxPooling1D': clean_layer(tf.keras.layers.MaxPooling1D, ['quantization_config'])
            }

            self.model = tf.keras.models.load_model(str(path), custom_objects=custom_objects)
            self.model_type = 'cnn_lstm'
            print(f"[Predictor] Loaded CNN-LSTM model from {path}")

        elif suffix == '.pkl':
            with open(str(path), 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'knn'
            print(f"[Predictor] Loaded KNN model from {path}")

        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def predict(self, sequence: np.ndarray) -> Dict:
        """
        Run prediction on a preprocessed landmark sequence.

        Args:
            sequence: Array of shape (30, 126) — preprocessed landmarks.

        Returns:
            Dictionary with:
              - word: predicted class name (str)
              - confidence: prediction confidence (float)
              - class_index: predicted class index (int)
              - top3: list of top-3 predictions [{word, confidence}]
              - is_low_confidence: bool (confidence < threshold)
        """
        if self.model is None:
            return {
                'word': 'unknown',
                'confidence': 0.0,
                'class_index': -1,
                'top3': [],
                'is_low_confidence': True
            }

        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)  # (1, 30, 126)

        if self.model_type == 'cnn_lstm':
            return self._predict_cnn_lstm(sequence)
        elif self.model_type == 'knn':
            return self._predict_knn(sequence)

    def _predict_cnn_lstm(self, sequence: np.ndarray) -> Dict:
        """Prediction using CNN-LSTM model."""
        probas = self.model.predict(sequence, verbose=0)[0]
        return self._format_result(probas)

    def _predict_knn(self, sequence: np.ndarray) -> Dict:
        """Prediction using KNN model (needs flattened input)."""
        flat = sequence.reshape(1, -1)  # (1, 3780)
        probas = self.model.predict_proba(flat)[0]
        return self._format_result(probas)

    def _format_result(self, probas: np.ndarray) -> Dict:
        """Format probability array into structured prediction result."""
        pred_idx = int(np.argmax(probas))
        confidence = float(probas[pred_idx])
        word = self.classes[pred_idx] if pred_idx < len(self.classes) else 'unknown'

        # Top 3 predictions
        top3_indices = np.argsort(probas)[-3:][::-1]
        top3 = [
            {
                'word': self.classes[int(i)] if int(i) < len(self.classes) else '?',
                'confidence': float(probas[int(i)])
            }
            for i in top3_indices
        ]

        return {
            'word': word,
            'confidence': round(confidence, 4),
            'class_index': pred_idx,
            'top3': top3,
            'is_low_confidence': confidence < LOW_CONFIDENCE_THRESHOLD
        }


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    predictor = Predictor()
    if predictor.model:
        # Test with random input
        dummy = np.random.randn(30, 126).astype(np.float32)
        result = predictor.predict(dummy)
        print(f"\nTest prediction: {result}")
    else:
        print("No model available for testing.")
