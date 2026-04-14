#!/usr/bin/env python3

import sys
import json
import time
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SPLITS_DIR = Path('./mvp/splits')
CLASSES_PATH = SPLITS_DIR / 'classes.json'
MODEL_PATH = Path('./mvp/model_cnn_bilstm.pt')
TARGET_FRAMES = 30
FEATURES_PER_FRAME = 126

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 210
PATIENCE = 25
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.4
WARMUP_EPOCHS = 5
MIN_LR = 1e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
            nn.Dropout(0.4),
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


def prepare_data(splits):
    X_train = splits['X_train'].reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME)
    X_val = splits['X_val'].reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME)
    X_test = splits['X_test'].reshape(-1, TARGET_FRAMES, FEATURES_PER_FRAME)

    y_train = splits['y_train'].astype(np.int64)
    y_val = splits['y_val'].astype(np.int64)
    y_test = splits['y_test'].astype(np.int64)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


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
    print("  ISL Sign Language — CNN-BiLSTM + Attention Training")
    print("=" * 60)

    print("\n  Loading data splits...")
    splits = load_splits()
    classes = load_classes()
    num_classes = len(classes)
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")
    print(f"  Device: {DEVICE}")

    train_loader, val_loader, test_loader = prepare_data(splits)

    model = SignLanguageModel(
        input_dim=FEATURES_PER_FRAME,
        num_classes=num_classes,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return max(MIN_LR / LEARNING_RATE, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n  Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                             criterion, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

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
        model, test_loader, criterion, DEVICE)
    print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(test_labels, test_preds,
                                    target_names=classes,
                                    zero_division=0, output_dict=True)
    print(f"\n  Per-class results (lowest 10):")
    class_accs = [(c, report[c]['f1-score']) for c in classes if c in report]
    class_accs.sort(key=lambda x: x[1])
    for name, f1 in class_accs[:10]:
        print(f"    {name:25s}  F1={f1:.3f}")

    print(f"\n  Saving model to {MODEL_PATH}...")
    save_data = {
        'model_state_dict': best_state,
        'model_config': {
            'input_dim': FEATURES_PER_FRAME,
            'num_classes': num_classes,
            'cnn_channels': 128,
            'lstm_hidden': 128,
            'lstm_layers': 2,
            'dropout': DROPOUT,
        },
        'classes': classes,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
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

