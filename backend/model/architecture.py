import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TARGET_FRAMES, FEATURES_PER_FRAME, NUM_CLASSES,
    DROPOUT_CNN, DROPOUT_DENSE
)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """
    Standard Transformer Encoder block with self-attention and feed-forward network.
    """
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Add()([x, inputs])

    # Feed Forward Part
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation="relu")(y)
    y = Dropout(dropout)(y)
    y = Dense(inputs.shape[-1])(y)
    return layers.Add()([y, x])


def build_model(num_classes: int = NUM_CLASSES,
                input_frames: int = TARGET_FRAMES,
                input_features: int = FEATURES_PER_FRAME) -> Model:
    """
    Build the CNN-BiLSTM-Transformer model for highly accurate ISL recognition.
    
    Architecture:
      Conv1D → BatchNorm → MaxPool → Conv1D → BatchNorm → Dropout
      → BiLSTM → BiLSTM
      → Transformer Encoder (MultiHeadAttention + FF)
      → GlobalAveragePooling1D → Dense → Dropout → Dense(softmax)
    """
    # Input layer
    inputs = Input(shape=(input_frames, input_features), name='input_landmarks')

    # ─── CNN Block (Spatial/Local Temporal Extraction) ──────────────────────
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    # Shape: (batch, 15, 64)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv1d_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(DROPOUT_CNN, name='dropout_cnn')(x)
    # Shape: (batch, 15, 128)

    # ─── Bidirectional LSTM Block (Sequential Context) ──────────────────────
    x = Bidirectional(LSTM(128, return_sequences=True, name='lstm_1'), name='bilstm_1')(x)
    # Shape: (batch, 15, 256)

    x = Bidirectional(LSTM(64, return_sequences=True, name='lstm_2'), name='bilstm_2')(x)
    # Shape: (batch, 15, 128)

    # ─── Transformer Encoder Block (Global Context & Attention) ─────────────
    # Allows the model to look at the entire sequence simultaneously and weigh
    # the most important temporal frames regardless of distance.
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    # Shape: (batch, 15, 128)

    # ─── Pooling & Dense Classifier ─────────────────────────────────────────
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    # Shape: (batch, 128)

    x = Dense(128, activation='relu', name='classifier_dense')(x)
    x = Dropout(DROPOUT_DENSE, name='dropout_dense')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    # Shape: (batch, NUM_CLASSES)

    model = Model(inputs=inputs, outputs=outputs, name='ISL_CNN_BiLSTM_Transformer')

    return model


def get_model_summary(model: Model) -> str:
    """
    Get a string representation of the model summary.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building CNN-BiLSTM-Transformer model...")
    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
