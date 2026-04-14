
import os
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# ─── Constants ───────────────────────────────────────────────────────────────
AUGMENTED_DIR = Path('./mvp/augmented')
SPLITS_DIR = Path('./mvp/splits')
PLOTS_DIR = Path('./plots')
CLASSES_PATH = SPLITS_DIR / 'classes.json'
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_augmented_data() -> tuple:
    """
    Load the augmented dataset arrays.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray, classes: list).
    """
    sequences_path = AUGMENTED_DIR / 'all_sequences.npy'
    labels_path = AUGMENTED_DIR / 'labels.npy'

    if not sequences_path.exists() or not labels_path.exists():
        print(f"[ERROR] Augmented data not found in {AUGMENTED_DIR}")
        print("Please run 2_augment.py first.")
        sys.exit(1)

    X = np.load(str(sequences_path))
    y = np.load(str(labels_path))

    # Load class names
    if CLASSES_PATH.exists():
        with open(str(CLASSES_PATH), 'r') as f:
            classes = json.load(f)
    else:
        classes = [str(i) for i in range(y.max() + 1)]

    return X, y, classes


def plot_class_distribution(y: np.ndarray, classes: list) -> None:
    """
    Generate and save class distribution bar plot.

    Args:
        y: Label array.
        classes: List of class names.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = Counter(y)
    class_counts = [(classes[i] if i < len(classes) else str(i), counts.get(i, 0))
                    for i in range(len(classes))]

    names = [c[0] for c in class_counts]
    values = [c[1] for c in class_counts]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))

    colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='white',
                  linewidth=0.5)

    ax.set_xlabel('Sign Word', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('ISL Dataset — Class Distribution (After Augmentation)',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)

    # Add value labels on bars
    for bar_item, val in zip(bars, values):
        ax.text(bar_item.get_x() + bar_item.get_width() / 2, bar_item.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=6)

    # Add statistics text
    mean_count = np.mean(values)
    std_count = np.std(values)
    ax.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {mean_count:.0f} ± {std_count:.0f}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = PLOTS_DIR / 'class_distribution.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distribution plot saved to: {plot_path}")


def split_dataset(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Split dataset into train (70%), val (15%), test (15%) with stratification.

    Args:
        X: Feature array of shape (N, 30, 126).
        y: Label array of shape (N,).

    Returns:
        Dictionary with train/val/test arrays.
    """
    # First split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=y
    )

    # Second split: 50% of temp for val, 50% for test (15% each of total)
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=RANDOM_SEED, stratify=y_temp
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def main():
    """
    Main entry point: load data, check health, plot distribution, split and save.
    """
    print(f"{'='*60}")
    print(f"  ISL Dataset Health Check")
    print(f"{'='*60}\n")

    # ─── Load Data ────────────────────────────────────────────────────────
    X, y, classes = load_augmented_data()

    print(f"  Total samples   : {len(X)}")
    print(f"  Feature shape   : {X.shape}")
    print(f"  Classes         : {len(classes)}")
    print(f"  Unique labels   : {len(np.unique(y))}")

    # ─── Class Distribution ───────────────────────────────────────────────
    print(f"\n  CLASS DISTRIBUTION:")
    counts = Counter(y)
    for i, cls_name in enumerate(classes):
        count = counts.get(i, 0)
        bar = '#' * (count // 5)
        print(f"    {cls_name:20s}: {count:4d}  {bar}")

    # Check for imbalance
    count_values = list(counts.values())
    min_count = min(count_values)
    max_count = max(count_values)
    imbalance_ratio = max_count / max(min_count, 1)

    print(f"\n  Min samples/class : {min_count}")
    print(f"  Max samples/class : {max_count}")
    print(f"  Imbalance ratio   : {imbalance_ratio:.2f}")

    if imbalance_ratio > 2.0:
        print(f"  [!] WARNING: Significant class imbalance detected!")
    else:
        print(f"  [OK] Class distribution looks balanced.")

    # ─── Plot ─────────────────────────────────────────────────────────────
    print(f"\n  Generating distribution plot...")
    plot_class_distribution(y, classes)

    # ─── Split ────────────────────────────────────────────────────────────
    print(f"\n  Splitting dataset (train={TRAIN_RATIO:.0%}, "
          f"val={VAL_RATIO:.0%}, test={TEST_RATIO:.0%})...")

    splits = split_dataset(X, y)

    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, arr in splits.items():
        save_path = SPLITS_DIR / f"{name}.npy"
        np.save(str(save_path), arr)
        print(f"    Saved {name}: {arr.shape}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Train : {splits['X_train'].shape[0]} samples ({splits['X_train'].shape[0]/len(X):.0%})")
    print(f"  Val   : {splits['X_val'].shape[0]} samples ({splits['X_val'].shape[0]/len(X):.0%})")
    print(f"  Test  : {splits['X_test'].shape[0]} samples ({splits['X_test'].shape[0]/len(X):.0%})")
    print(f"  Splits saved to: {SPLITS_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
