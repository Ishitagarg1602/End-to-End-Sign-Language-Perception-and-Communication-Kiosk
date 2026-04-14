#!/usr/bin/env python3

import sys
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d

SPLITS_DIR = Path('./mvp/splits')
CLASSES_PATH = SPLITS_DIR / 'classes.json'
MODEL_PATH = Path('./mvp/model_cnn_bilstm.pt')
TARGET_FRAMES = 30
FEATURES_PER_FRAME = 126

BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPOCHS = 200
PATIENCE = 25
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def online_augment(batch_x: torch.Tensor) -> torch.Tensor:
    B, T, F = batch_x.shape
    augmented = batch_x.clone()

    for i in range(B):
        seq = augmented[i].numpy()

        if np.random.random() < 0.5:
            sigma = np.random.uniform(0.003, 0.015)
            seq = seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)

        if np.random.random() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            num_frames = seq.shape[0]
            new_len = max(2, int(num_frames * factor))
            x_orig = np.linspace(0, 1, num_frames)
            x_new = np.linspace(0, 1, new_len)
            interp = interp1d(x_orig, seq, axis=0, kind='linear', fill_value='extrapolate')
            stretched = interp(x_new)
            x_s = np.linspace(0, 1, new_len)
            x_t = np.linspace(0, 1, TARGET_FRAMES)
            interp2 = interp1d(x_s, stretched, axis=0, kind='linear', fill_value='extrapolate')
            seq = interp2(x_t).astype(np.float32)

        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, 4)
            keep = sorted(set(range(TARGET_FRAMES)) - set(np.random.choice(range(1, TARGET_FRAMES-1), n_drop, replace=False)))
            kept = seq[keep]
            x_k = np.linspace(0, 1, len(keep))
            x_t = np.linspace(0, 1, TARGET_FRAMES)
            interp = interp1d(x_k, kept, axis=0, kind='linear', fill_value='extrapolate')
            seq = interp(x_t).astype(np.float32)

        if np.random.random() < 0.25:
            scale = np.random.uniform(0.85, 1.15)
            seq = seq * scale

        if np.random.random() < 0.2:
            angle = np.random.uniform(-15, 15)
            theta = np.radians(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotated = seq.copy()
            for hand_offset in [0, 63]:
                for kp in range(21):
                    x_idx = hand_offset + kp * 3
                    y_idx = hand_offset + kp * 3 + 1
                    x_vals = rotated[:, x_idx].copy()
                    y_vals = rotated[:, y_idx].copy()
                    rotated[:, x_idx] = x_vals * cos_t - y_vals * sin_t
                    rotated[:, y_idx] = x_vals * sin_t + y_vals * cos_t
            seq = rotated

        augmented[i] = torch.FloatTensor(seq)

    return augmented


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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
                 cnn_channels=128, lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.attention = Attention(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits


class SignDataset(Dataset):

    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X.reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME))
        self.y = torch.LongTensor(y.astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = online_augment(x.unsqueeze(0)).squeeze(0)
        return x, y


def load_splits():
    splits = {}
    for name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        path = SPLITS_DIR / f"{name}.npy"
        if not path.exists():
            print(f"  [ERROR] Split file not found: {path}")
            print("  Please run data_pipeline/3_health_check.py first.")
            sys.exit(1)
        splits[name] = np.load(str(path))
    return splits


def load_classes():
    if not CLASSES_PATH.exists():
        print(f"  [ERROR] classes.json not found at {CLASSES_PATH}")
        sys.exit(1)
    with open(str(CLASSES_PATH), 'r') as f:
        return json.load(f)


def train_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if use_mixup and np.random.random() < 0.5:
            X_batch, y_a, y_b, lam = mixup_data(X_batch, y_batch, MIXUP_ALPHA)
            logits = model(X_batch)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        optimizer.zero_grad()
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
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    print("=" * 60)
    print("  ISL Sign Language — V2 Training (Anti-Overfitting)")
    print("=" * 60)

    print("\n  Loading data splits...")
    splits = load_splits()
    classes = load_classes()
    num_classes = len(classes)
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")
    print(f"  Device: {DEVICE}")

    train_ds = SignDataset(splits['X_train'], splits['y_train'], augment=True)
    val_ds = SignDataset(splits['X_val'], splits['y_val'], augment=False)
    test_ds = SignDataset(splits['X_test'], splits['y_test'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SignLanguageModel(
        input_dim=FEATURES_PER_FRAME,
        num_classes=num_classes,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    print("\n  ★ V2 Improvements:")
    print(f"    Label smoothing: {LABEL_SMOOTHING}")
    print(f"    Mixup alpha: {MIXUP_ALPHA}")
    print(f"    Online augmentation: ON (noise, stretch, dropout, scale, rotation)")
    print(f"    Dropout: 0.5 (was 0.3)")
    print(f"    LR schedule: Cosine Annealing")
    print(f"    Batch size: {BATCH_SIZE} (was 64)")
    print(f"    Weight decay: {WEIGHT_DECAY}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    print(f"\n  Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                             criterion, DEVICE, use_mixup=True)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion_eval, DEVICE)
        scheduler.step(epoch)

        lr = optimizer.param_groups[0]['lr']

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " ★"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={lr:.2e}{improved}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best val_loss={best_val_loss:.4f} | Best val_acc={best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    print("\n  Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion_eval, DEVICE)
    print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    try:
        from sklearn.metrics import classification_report
        report = classification_report(test_labels, test_preds,
                                        target_names=classes,
                                        zero_division=0, output_dict=True)
        print(f"\n  Per-class results (lowest 10):")
        class_accs = [(c, report[c]['f1-score']) for c in classes if c in report]
        class_accs.sort(key=lambda x: x[1])
        for name, f1 in class_accs[:10]:
            print(f"    {name:25s}  F1={f1:.3f}")

        print(f"\n  Per-class results (highest 10):")
        for name, f1 in class_accs[-10:]:
            print(f"    {name:25s}  F1={f1:.3f}")
    except ImportError:
        print("  (sklearn not available for per-class report)")

    print(f"\n  Saving model to {MODEL_PATH}...")
    save_data = {
        'model_state_dict': best_state,
        'model_config': {
            'input_dim': FEATURES_PER_FRAME,
            'num_classes': num_classes,
            'cnn_channels': 128,
            'lstm_hidden': 128,
            'lstm_layers': 2,
            'dropout': 0.5,
        },
        'classes': classes,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'training_version': 'v2',
    }
    torch.save(save_data, str(MODEL_PATH))
    print(f"  ✓ Model saved ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    classes_out = Path('./mvp/classes.json')
    with open(str(classes_out), 'w') as f:
        json.dump(classes, f)
    print(f"  ✓ Classes saved to {classes_out}")

    print("\n" + "=" * 60)
    print(f"  DONE — Test accuracy: {test_acc*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()

