
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SPLITS_DIR, CLASSES_PATH, NUM_CLASSES

ML_DIR = Path(__file__).resolve().parent.parent.parent / 'ml'
SAVED_MODELS_DIR = ML_DIR / 'saved_models'
PLOTS_DIR = ML_DIR / 'plots'
BEST_MODEL_PATH = SAVED_MODELS_DIR / 'best_model.h5'


def load_test_data() -> tuple:
    """
    Load test split and class names.

    Returns:
        Tuple of (X_test, y_test, classes).
    """
    X_test = np.load(str(SPLITS_DIR / 'X_test.npy'))
    y_test = np.load(str(SPLITS_DIR / 'y_test.npy'))

    if CLASSES_PATH.exists():
        with open(str(CLASSES_PATH), 'r') as f:
            classes = json.load(f)
    else:
        classes = [str(i) for i in range(NUM_CLASSES)]

    return X_test, y_test, classes


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           classes: list, save_path: Path) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Args:
        y_true: True labels (integer).
        y_pred: Predicted labels (integer).
        classes: List of class names.
        save_path: Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5, linecolor='white')

    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True', fontsize=14, fontweight='bold')
    ax.set_title('ISL Recognition — Confusion Matrix', fontsize=16, fontweight='bold')

    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def find_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                         classes: list, top_n: int = 10) -> list:
    """
    Find the most confused class pairs.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class name list.
        top_n: Number of top pairs to return.

    Returns:
        List of (true_class, pred_class, count) tuples.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                true_cls = classes[i] if i < len(classes) else str(i)
                pred_cls = classes[j] if j < len(classes) else str(j)
                pairs.append((true_cls, pred_cls, int(cm[i][j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def evaluate():
    """
    Main evaluation function: load model, predict, report metrics.
    """
    print(f"{'='*60}")
    print(f"  CNN-LSTM Model Evaluation")
    print(f"{'='*60}\n")

    # ─── Load ─────────────────────────────────────────────────────────────
    if not BEST_MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {BEST_MODEL_PATH}")
        print("Please train the model first (python -m backend.model.train).")
        sys.exit(1)

    print(f"  Loading model from {BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    X_test, y_test, classes = load_test_data()
    print(f"  Test set: {X_test.shape}")

    # ─── Predict ──────────────────────────────────────────────────────────
    print(f"  Running predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # ─── Metrics ──────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # ─── Per-Class Report ─────────────────────────────────────────────────
    print(f"\n  PER-CLASS REPORT:")
    class_names = [classes[i] if i < len(classes) else str(i)
                   for i in range(len(classes))]
    report = classification_report(y_test, y_pred,
                                    target_names=class_names,
                                    zero_division=0)
    print(report)

    # ─── Top 10 Confused Pairs ────────────────────────────────────────────
    print(f"  TOP 10 MOST CONFUSED PAIRS:")
    confused = find_confused_pairs(y_test, y_pred, classes)
    if confused:
        for i, (t, p, c) in enumerate(confused):
            print(f"    {i+1:2d}. {t:20s} → {p:20s}  ({c} times)")
    else:
        print(f"    No confusions! Perfect classification!")

    # ─── Confusion Matrix ─────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, classes,
                          PLOTS_DIR / 'confusion_matrix.png')

    print(f"\n{'='*60}")


if __name__ == '__main__':
    evaluate()
