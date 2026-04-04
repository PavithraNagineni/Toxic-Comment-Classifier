"""
Toxic Comment Classifier - Training Pipeline
Multi-label classification with BCEWithLogitsLoss + MLflow tracking
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
import mlflow
import mlflow.pytorch

from model import TextCNN, ToxicDataset
from preprocessing import TextPreprocessor, LABEL_COLUMNS

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "data_path": "data/train.csv",           # Jigsaw Toxic Comment dataset
    "artifacts_dir": "artifacts",
    "embed_dim": 128,
    "num_filters": 128,
    "filter_sizes": [2, 3, 4, 5],
    "dropout": 0.5,
    "max_vocab": 50_000,
    "max_len": 200,
    "min_freq": 2,
    "epochs": 20,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
    "test_size": 0.1,
    "val_size": 0.1,
    "mlflow_experiment": "toxic_comment_classifier",
}


# ─── Metric Helpers ───────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_prob >= threshold).astype(int)
    per_class_auc = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            per_class_auc.append(roc_auc_score(y_true[:, i], y_pred_prob[:, i]))
    return {
        "mean_roc_auc": float(np.mean(per_class_auc)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, all_probs


# ─── Main ─────────────────────────────────────────────────────────────────────
def train():
    df = pd.read_csv(CONFIG["data_path"])
    print(f"Dataset: {len(df)} rows | Labels: {LABEL_COLUMNS}")

    # Check for class imbalance
    for col in LABEL_COLUMNS:
        pos_rate = df[col].mean() * 100
        print(f"  {col}: {pos_rate:.2f}% positive")

    texts = df["comment_text"].tolist()
    labels = df[LABEL_COLUMNS].values.astype(np.float32)

    # Build NLP preprocessor
    preprocessor = TextPreprocessor(
        max_vocab=CONFIG["max_vocab"],
        max_len=CONFIG["max_len"],
        min_freq=CONFIG["min_freq"],
    )
    preprocessor.fit(texts)
    X = preprocessor.batch_encode(texts)

    os.makedirs(CONFIG["artifacts_dir"], exist_ok=True)
    preprocessor.save(CONFIG["artifacts_dir"])

    # Stratify on toxic label (most informative)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=CONFIG["test_size"] + CONFIG["val_size"],
        stratify=labels[:, 0].astype(int), random_state=42
    )
    val_frac = CONFIG["val_size"] / (CONFIG["test_size"] + CONFIG["val_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_frac,
        stratify=y_temp[:, 0].astype(int), random_state=42
    )
    print(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(ToxicDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(ToxicDataset(X_val, y_val), batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(ToxicDataset(X_test, y_test), batch_size=CONFIG["batch_size"])

    model = TextCNN(
        vocab_size=preprocessor.vocab_size,
        embed_dim=CONFIG["embed_dim"],
        num_filters=CONFIG["num_filters"],
        filter_sizes=CONFIG["filter_sizes"],
        num_classes=len(LABEL_COLUMNS),
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Weighted BCE for class imbalance
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    mlflow.set_experiment(CONFIG["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in CONFIG.items()
                           if k not in ["data_path", "artifacts_dir", "mlflow_experiment"]})

        best_val_auc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(1, CONFIG["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_metrics, _ = evaluate(model, val_loader, criterion)
            scheduler.step()

            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val AUC: {val_metrics['mean_roc_auc']:.4f} | "
                  f"Val F1-micro: {val_metrics['f1_micro']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }, step=epoch)

            if val_metrics["mean_roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["mean_roc_auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        test_metrics, _ = evaluate(model, test_loader, criterion)
        print("\n=== Test Results ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            mlflow.log_metric(f"test_{k}", v)

        # Save model
        model_path = os.path.join(CONFIG["artifacts_dir"], "textcnn_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": preprocessor.vocab_size,
            "embed_dim": CONFIG["embed_dim"],
            "num_filters": CONFIG["num_filters"],
            "filter_sizes": CONFIG["filter_sizes"],
            "num_classes": len(LABEL_COLUMNS),
            "dropout": CONFIG["dropout"],
        }, model_path)
        mlflow.pytorch.log_model(model, "textcnn")
        print(f"\nSaved to {model_path} | Best Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    train()
