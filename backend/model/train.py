
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SPLITS_DIR, CLASSES_PATH, PLOTS_DIR,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, MIN_LR, NUM_CLASSES
)
from model.architecture import build_model


# ─── Paths ───────────────────────────────────────────────────────────────────
ML_DIR = Path(__file__).resolve().parent.parent.parent / 'ml'
SAVED_MODELS_DIR = ML_DIR / 'saved_models'
BEST_MODEL_PATH = SAVED_MODELS_DIR / 'best_model.h5'


def load_training_data() -> dict:
    """
    Load train/val/test splits from .npy files.

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test,
        classes list, and num_classes.
    """
    data = {}
    for name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        path = SPLITS_DIR / f"{name}.npy"
        if not path.exists():
            print(f"[ERROR] {path} not found. Run data pipeline first.")
            sys.exit(1)
        data[name] = np.load(str(path))

    # Load class names
    if CLASSES_PATH.exists():
        with open(str(CLASSES_PATH), 'r') as f:
            data['classes'] = json.load(f)
    else:
        data['classes'] = [str(i) for i in range(NUM_CLASSES)]

    data['num_classes'] = len(data['classes'])
    return data


def compute_weights(y_train: np.ndarray, num_classes: int) -> dict:
    """
    Compute class weights to handle imbalanced data.

    Args:
        y_train: Training labels (integer encoded).
        num_classes: Total number of classes.

    Returns:
        Dictionary mapping class index to weight.
    """
    unique_classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}

    # Fill in missing classes with weight 1.0
    for i in range(num_classes):
        if i not in weight_dict:
            weight_dict[i] = 1.0

    return weight_dict


def plot_training_history(history, save_dir: Path) -> None:
    """
    Plot and save training/validation accuracy and loss curves.

    Args:
        history: Keras training history object.
        save_dir: Directory to save plot images.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_dir / 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to {save_dir / 'training_curves.png'}")


def train():
    """
    Main training function: load data, build model, train, and save.
    """
    print(f"{'='*60}")
    print(f"  CNN-LSTM Training — ISL Sign Language Recognition")
    print(f"{'='*60}\n")

    # ─── Load Data ────────────────────────────────────────────────────────
    print("  Loading data...")
    data = load_training_data()

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    num_classes = data['num_classes']
    classes = data['classes']

    print(f"  Train : {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Val   : {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test  : {X_test.shape}, Labels: {y_test.shape}")
    print(f"  Classes: {num_classes}")

    # ─── One-Hot Encode Labels ────────────────────────────────────────────
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # ─── Compute Class Weights ────────────────────────────────────────────
    class_weights = compute_weights(y_train, num_classes)
    print(f"\n  Class weights computed ({min(class_weights.values()):.2f} - "
          f"{max(class_weights.values()):.2f})")

    # ─── Build Model ──────────────────────────────────────────────────────
    print(f"\n  Building CNN-LSTM model...")
    model = build_model(num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ─── Callbacks ────────────────────────────────────────────────────────
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        )
    ]

    # ─── Train ────────────────────────────────────────────────────────────
    print(f"\n  Starting training...")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Max epochs : {MAX_EPOCHS}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Model path : {BEST_MODEL_PATH}\n")

    start_time = time.time()

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    train_time = time.time() - start_time

    # ─── Evaluate ─────────────────────────────────────────────────────────
    print(f"\n  Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Training time  : {train_time/60:.1f} minutes")
    print(f"  Best val acc   : {max(history.history['val_accuracy']):.4f}")
    print(f"  Test accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Test loss      : {test_loss:.4f}")
    print(f"  Model saved to : {BEST_MODEL_PATH}")

    # ─── Plot ─────────────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, PLOTS_DIR)

    print(f"{'='*60}")


if __name__ == '__main__':
    train()
