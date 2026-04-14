
import sys
import json
import time
from pathlib import Path
import pickle
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


SPLITS_DIR = Path('./mvp/splits')
MODEL_PATH = Path('./mvp/model.pkl')
CLASSES_PATH = SPLITS_DIR / 'classes.json'
K_CANDIDATES = [3, 5, 7, 9, 11]


def load_splits() -> dict:
    splits = {}
    for name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        path = SPLITS_DIR / f"{name}.npy"
        if not path.exists():
            print(f"[ERROR] Split file not found: {path}")
            print("Please run data_pipeline/3_health_check.py first.")
            sys.exit(1)
        splits[name] = np.load(str(path))
    return splits


def load_classes() -> list:
    if not CLASSES_PATH.exists():
        print(f"[ERROR] classes.json not found at {CLASSES_PATH}")
        sys.exit(1)

    with open(str(CLASSES_PATH), 'r') as f:
        classes = json.load(f)
    return classes


def find_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                         classes: list, top_n: int = 10) -> list:
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)

    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                class_a = classes[i] if i < len(classes) else str(i)
                class_b = classes[j] if j < len(classes) else str(j)
                confused_pairs.append((class_a, class_b, cm[i][j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    return confused_pairs[:top_n]


def main():
    print(f"{'='*60}")
    print(f"  ISL KNN Classifier — MVP Training")
    print(f"{'='*60}\n")

    print("  Loading splits...")
    splits = load_splits()
    classes = load_classes()

    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']

    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"  Classes: {len(classes)}")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(f"\n  Flattened shape: {X_train_flat.shape}")

    print(f"\n  Searching best k from {K_CANDIDATES}...")
    start_time = time.time()

    best_k = K_CANDIDATES[0]
    best_val_acc = 0.0
    k_results = []

    for k in K_CANDIDATES:
        knn_trial = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        )
        knn_trial.fit(X_train_flat, y_train)
        val_pred = knn_trial.predict(X_val_flat)
        val_acc = accuracy_score(y_val, val_pred)
        k_results.append((k, val_acc))
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            marker = " ← best"
        print(f"    k={k:2d}  val_acc={val_acc:.4f} ({val_acc*100:.1f}%){marker}")

    print(f"\n  ✓ Best k = {best_k} (val accuracy: {best_val_acc:.4f})")

    print(f"\n  Training final KNN (k={best_k})...")

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    knn.fit(X_train_flat, y_train)

    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s")

    print(f"\n  Evaluating...")

    train_pred = knn.predict(X_train_flat)
    val_pred = knn.predict(X_val_flat)
    test_pred = knn.predict(X_test_flat)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Train accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"  Val accuracy   : {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"  Test accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")

    print(f"\n  TOP 10 MOST CONFUSED PAIRS (on test set):")
    confused = find_confused_pairs(y_test, test_pred, classes)

    if confused:
        for i, (a, b, count) in enumerate(confused):
            print(f"    {i+1:2d}. {a:20s} ↔ {b:20s}  ({count} confusions)")
    else:
        print(f"    No confusions found — perfect classification!")

    print(f"\n  PER-CLASS REPORT (test set):")
    class_names = [classes[i] if i < len(classes) else str(i)
                   for i in range(len(classes))]
    report = classification_report(y_test, test_pred,
                                    target_names=class_names,
                                    zero_division=0)
    print(report)

    print(f"  Saving model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(str(MODEL_PATH), 'wb') as f:
        pickle.dump(knn, f)

    model_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  Model saved. Size: {model_size:.2f} MB")

    if not CLASSES_PATH.exists():
        with open(str(CLASSES_PATH), 'w') as f:
            json.dump(classes, f, indent=2)
        print(f"  Classes saved to {CLASSES_PATH}")

    print(f"\n{'='*60}")
    print(f"  MVP KNN model ready! Use mvp/model.pkl for predictions.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

