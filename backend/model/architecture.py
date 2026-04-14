
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, Multiply, Permute,
    RepeatVector, Flatten, Lambda, Activation
)
import tensorflow.keras.backend as K

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TARGET_FRAMES, FEATURES_PER_FRAME, NUM_CLASSES,
    DROPOUT_CNN, DROPOUT_DENSE
)


def attention_layer(inputs):
    """
    Time-step attention mechanism.

    Learns a weight for each time step, then computes a weighted sum
    of the LSTM outputs. This lets the model focus on frames where
    the most distinctive hand shapes occur.

    Args:
        inputs: Tensor of shape (batch, time_steps, features).

    Returns:
        Context vector of shape (batch, features).
    """
    # Learn importance score per time step
    # (batch, time_steps, features) → (batch, time_steps, 1)
    score = Dense(1, use_bias=False)(inputs)

    # Softmax over time dimension → attention weights
    # (batch, time_steps, 1)
    weights = Activation('softmax')(score)

    # Weighted sum: element-wise multiply then sum over time
    # (batch, time_steps, features) * (batch, time_steps, 1) → sum → (batch, features)
    context = Multiply()([inputs, weights])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    return context


def build_model(num_classes: int = NUM_CLASSES,
                input_frames: int = TARGET_FRAMES,
                input_features: int = FEATURES_PER_FRAME) -> Model:
    """
    Build the CNN-LSTM model with Attention for ISL recognition.

    Architecture follows the exact specification:
      Conv1D → BatchNorm → MaxPool → Conv1D → BatchNorm → Dropout
      → BiLSTM → BiLSTM → Attention → Dense → Dropout → Dense(softmax)

    Args:
        num_classes: Number of output classes (default: 60).
        input_frames: Number of time steps (default: 30).
        input_features: Features per frame (default: 126).

    Returns:
        Compiled-ready Keras Model (not compiled — caller should compile).
    """
    # Input layer
    inputs = Input(shape=(input_frames, input_features), name='input_landmarks')

    # ─── CNN Block 1 ─────────────────────────────────────────────────────
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
               name='conv1d_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    # Shape: (batch, 15, 64)

    # ─── CNN Block 2 ─────────────────────────────────────────────────────
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same',
               name='conv1d_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(DROPOUT_CNN, name='dropout_cnn')(x)
    # Shape: (batch, 15, 128)

    # ─── Bidirectional LSTM Block 1 ───────────────────────────────────────
    x = Bidirectional(LSTM(128, return_sequences=True, name='lstm_1'),
                      name='bilstm_1')(x)
    # Shape: (batch, 15, 256)

    # ─── Bidirectional LSTM Block 2 ───────────────────────────────────────
    x = Bidirectional(LSTM(64, return_sequences=True, name='lstm_2'),
                      name='bilstm_2')(x)
    # Shape: (batch, 15, 128)

    # ─── Attention Layer ──────────────────────────────────────────────────
    x = attention_layer(x)
    # Shape: (batch, 128)

    # ─── Dense Classifier ────────────────────────────────────────────────
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(DROPOUT_DENSE, name='dropout_dense')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    # Shape: (batch, 60)

    model = Model(inputs=inputs, outputs=outputs, name='ISL_CNN_LSTM_Attention')

    return model


def get_model_summary(model: Model) -> str:
    """
    Get a string representation of the model summary.

    Args:
        model: Keras Model instance.

    Returns:
        Model summary as a string.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building CNN-LSTM model with Attention...")
    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
