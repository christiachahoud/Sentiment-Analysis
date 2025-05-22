"""
Training script for Hybrid BERT + CNN + MLP GoEmotions Classifier
- Loads preprocessed datasets from artifacts/
- Trains model using BCEWithLogitsLoss + label smoothing
- Tracks F1 (micro/macro) and Hamming Loss
- Saves best model based on validation F1-micro
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import f1_score, hamming_loss
from pathlib import Path
from model import HybridBERTClassifier
from tqdm import tqdm

# ========== Utility ==========
def smooth_labels(labels, smoothing=0.05):
    return labels * (1 - smoothing) + 0.5 * smoothing

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training", leave=True):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        smoothed = smooth_labels(labels, smoothing=0.05)
        loss = loss_fn(logits, smoothed)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = torch.tensor(np.vstack(all_preds))
    all_labels = torch.tensor(np.vstack(all_labels))
    return total_loss / len(dataloader), all_preds, all_labels

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Training", leave=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = torch.tensor(np.vstack(all_preds))
    all_labels = torch.tensor(np.vstack(all_labels))
    return total_loss / len(dataloader), all_preds, all_labels

# ========== Main ==========
def main():
    # === Hyperparameters ===
    BATCH_SIZE = 16
    NUM_EPOCHS = 4

    # === Paths ===
    ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
    train_dataset = torch.load(ARTIFACT_DIR / "train_dataset.pt", weights_only=False)
    val_dataset = torch.load(ARTIFACT_DIR / "val_dataset.pt", weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    num_classes = train_dataset.labels.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridBERTClassifier(num_labels=num_classes).to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": model.convs.parameters(), "lr": 1e-3},
        {"params": model.mlp.parameters(), "lr": 1e-3},
        {"params": model.mlp_projection.parameters(), "lr": 1e-3},
        {"params": model.gate_layer.parameters(), "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
        {"params": model.layernorm_cnn.parameters(), "lr": 1e-3},
        {"params": model.layernorm_mlp.parameters(), "lr": 1e-3}
    ], weight_decay=1e-2)

    loss_fn = nn.BCEWithLogitsLoss()
    best_f1 = 0.0

    # === Training Loop ===
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_preds, train_labels = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, loss_fn, device)

        train_f1_micro = f1_score(train_labels, train_preds, average='micro')
        train_f1_macro = f1_score(train_labels, train_preds, average='macro')
        train_hamming = hamming_loss(train_labels, train_preds)

        val_f1_micro = f1_score(val_labels, val_preds, average='micro')
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        val_hamming = hamming_loss(val_labels, val_preds)

        if val_f1_micro > best_f1:
            best_f1 = val_f1_micro
            torch.save(model.state_dict(), "best_hybrid_model.pt")
            print(f"✅ New best model saved (Val F1 Micro: {best_f1:.4f})")

        print(f"Train Loss: {train_loss:.4f} | F1 Micro: {train_f1_micro:.4f} | F1 Macro: {train_f1_macro:.4f} | Hamming Loss: {train_hamming:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | F1 Micro: {val_f1_micro:.4f} | F1 Macro: {val_f1_macro:.4f} | Hamming Loss: {val_hamming:.4f}")

if __name__ == "__main__":
    main()
