#!/usr/bin/env python3

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPLITS_DIR = SCRIPT_DIR / 'splits'
CLASSES_PATH = SPLITS_DIR / 'classes.json'
MODEL_PATH = SCRIPT_DIR / 'model_cnn_bilstm.pt'
FEEDBACK_DIR = PROJECT_ROOT / 'feedback'

TARGET_FRAMES = 30
FEATURES_PER_FRAME = 126

BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 50
PATIENCE = 10
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BUTTER_ORDER = 3
BUTTER_CUTOFF = 6.0
BUTTER_FS = 30.0


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        weights = self.attn(lstm_out)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context, weights.squeeze(-1)


class SignLanguageModel(nn.Module):
    def __init__(self, input_dim=126, num_classes=61,
                 cnn_channels=128, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=lstm_hidden,
            num_layers=lstm_layers, bidirectional=True,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
        )
        self.attention = Attention(lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        context, _ = self.attention(lstm_out)
        return self.classifier(context)


def preprocess_sequence(raw_frames):
    arr = np.array(raw_frames, dtype=np.float32)
    n_frames = arr.shape[0]

    if n_frames >= 8:
        try:
            b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF / (BUTTER_FS / 2), btype='low')
            for col in range(arr.shape[1]):
                arr[:, col] = filtfilt(b, a, arr[:, col])
        except Exception:
            pass

    if n_frames == TARGET_FRAMES:
        return arr.astype(np.float32)

    if n_frames == 1:
        return np.repeat(arr, TARGET_FRAMES, axis=0)
    elif n_frames == 2:
        # just repeat the first frame for half, second for half
        return np.repeat(arr, TARGET_FRAMES // 2 + 1, axis=0)[:TARGET_FRAMES]

    x_old = np.linspace(0, 1, n_frames)
    x_new = np.linspace(0, 1, TARGET_FRAMES)
    interpolated = np.zeros((TARGET_FRAMES, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        f = interp1d(x_old, arr[:, col], kind='linear', fill_value='extrapolate')
        interpolated[:, col] = f(x_new)

    return interpolated


def load_feedback_data(classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X_list = []
    y_list = []
    skipped = 0

    if not FEEDBACK_DIR.exists():
        print("  No feedback directory found.")
        return None, None

    for word_dir in sorted(FEEDBACK_DIR.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name
        if word not in class_to_idx:
            print(f"  [SKIP] Unknown class: '{word}' — not in classes.json")
            skipped += 1
            continue

        class_idx = class_to_idx[word]
        npy_files = list(word_dir.glob('*.npy'))

        for npy_file in npy_files:
            try:
                raw = np.load(str(npy_file))
                if raw.ndim != 2 or raw.shape[1] != FEATURES_PER_FRAME:
                    print(f"  [SKIP] Bad shape {raw.shape}: {npy_file.name}")
                    skipped += 1
                    continue
                if raw.shape[0] < 1:
                    print(f"  [SKIP] Empty data: {npy_file.name}")
                    skipped += 1
                    continue

                processed = preprocess_sequence(raw)
                X_list.append(processed)
                y_list.append(class_idx)
            except Exception as e:
                print(f"  [SKIP] Error loading {npy_file}: {e}")
                skipped += 1

    if not X_list:
        print(f"  No valid feedback samples found. Skipped: {skipped}")
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"  Loaded {len(X)} feedback samples across {len(set(y))} classes (skipped {skipped})")
    return X, y


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def main():
    print("=" * 60)
    print("  Retrain Model with Feedback Data")
    print("=" * 60)

    if not CLASSES_PATH.exists():
        print(f"  [ERROR] classes.json not found: {CLASSES_PATH}")
        sys.exit(1)
    with open(str(CLASSES_PATH)) as f:
        classes = json.load(f)
    print(f"  Classes: {len(classes)}")

    print("\n  Loading original training data...")
    X_train_orig = np.load(str(SPLITS_DIR / 'X_train.npy'))
    y_train_orig = np.load(str(SPLITS_DIR / 'y_train.npy')).astype(np.int64)
    X_val = np.load(str(SPLITS_DIR / 'X_val.npy'))
    y_val = np.load(str(SPLITS_DIR / 'y_val.npy')).astype(np.int64)
    X_test = np.load(str(SPLITS_DIR / 'X_test.npy'))
    y_test = np.load(str(SPLITS_DIR / 'y_test.npy')).astype(np.int64)
    print(f"  Original train: {len(X_train_orig)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n  Loading feedback data...")
    X_fb, y_fb = load_feedback_data(classes)

    if X_fb is not None and len(X_fb) > 0:
        n_repeats = 3
        X_fb_repeated = np.repeat(X_fb, n_repeats, axis=0)
        y_fb_repeated = np.repeat(y_fb, n_repeats, axis=0)
        print(f"  Feedback oversampled {n_repeats}x: {len(X_fb)} -> {len(X_fb_repeated)} samples")

        X_train = np.concatenate([X_train_orig, X_fb_repeated], axis=0)
        y_train = np.concatenate([y_train_orig, y_fb_repeated], axis=0)
        print(f"  Merged training set: {len(X_train)} samples")
    else:
        print("  No feedback data found — training on original data only")
        X_train = X_train_orig
        y_train = y_train_orig

    if X_train.ndim == 2:
        X_train = X_train.reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME)
    if X_val.ndim == 2:
        X_val = X_val.reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME)
    if X_test.ndim == 2:
        X_test = X_test.reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  Loading pretrained model from {MODEL_PATH}...")
    checkpoint = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    config = checkpoint['model_config']
    model = SignLanguageModel(
        input_dim=config['input_dim'], num_classes=config['num_classes'],
        cnn_channels=config['cnn_channels'], lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'], dropout=0.3,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    old_acc = checkpoint.get('test_accuracy', 0)
    print(f"  Pretrained accuracy: {old_acc:.2%}")
    print(f"  Device: {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"\n  Training for up to {EPOCHS} epochs (lr={LEARNING_RATE}, patience={PATIENCE})...\n")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.2e}{improved}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best val_acc={best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    model.load_state_dict(best_state)
    model.to(DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Previous accuracy: {old_acc:.2%}")

    if test_acc >= old_acc - 0.01:
        print(f"\n  Saving improved model to {MODEL_PATH}...")
        save_data = {
            'model_state_dict': best_state,
            'model_config': config,
            'classes': classes,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'feedback_samples': len(X_fb) if X_fb is not None else 0,
        }
        torch.save(save_data, str(MODEL_PATH))
        print(f"  Model saved ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"\n  [WARNING] New accuracy ({test_acc:.2%}) is significantly lower than original ({old_acc:.2%})")
        print(f"  Model NOT saved to avoid regression. Check feedback data quality.")

    print("\n" + "=" * 60)
    print(f"  DONE — Test accuracy: {test_acc*100:.1f}% (was {old_acc*100:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()

