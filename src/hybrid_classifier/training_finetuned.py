"""
Option 2b-Hybrid: BERT + CNN + MLP for Multi-Label Emotion Detection (GoEmotions)
This script:
- Uses BERT to get token-level embeddings
- Applies a 1D CNN over token embeddings to extract local patterns
- Applies an MLP on the [CLS] token for global context
- Concatenates both outputs and passes through a final classifier
- Trains using BCEWithLogitsLoss for multi-label classification (28 emotions)
- Tracks F1 (micro/macro) and Hamming Loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

def smooth_labels(labels, smoothing=0.05):
    return labels * (1 - smoothing) + 0.5 * smoothing

# =======================
# Hybrid Model Definition
# =======================
class HybridBERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=28):
        super(HybridBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # CNN branch (multi-kernel)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=k, padding=k // 2)
            for k in [2, 3, 4, 5]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)

        # MLP branch for [CLS] token
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )
        self.mlp_projection = nn.Linear(128, 512)  # match CNN output size

        # Regularization layers
        self.layernorm_cnn = nn.LayerNorm(512)
        self.layernorm_mlp = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)

        # For no dropout:
        # self.dropout = nn.Identity()

        # Gated fusion: input is [512 CNN + 512 MLP] → output 512
        self.gate_layer = nn.Linear(512 + 512, 512)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, seq_len, 768]
        cls_token = sequence_output[:, 0, :]         # [B, 768]

        # CNN branch
        cnn_input = sequence_output.transpose(1, 2)  # [B, 768, seq_len]
        cnn_features = []
        for conv in self.convs:
            conv_out = conv(cnn_input)                         # [B, 128, seq_len]
            pooled = self.pool(conv_out).squeeze(-1)           # [B, 128]
            cnn_features.append(pooled)
        pooled_cnn = torch.cat(cnn_features, dim=1)            # [B, 512]
        pooled_cnn = self.dropout(self.layernorm_cnn(pooled_cnn))

        # MLP branch
        mlp_output = self.mlp(cls_token)                       # [B, 128]
        mlp_output = self.mlp_projection(mlp_output)           # [B, 512]
        mlp_output = self.dropout(self.layernorm_mlp(mlp_output))

        # Gated fusion
        gate_input = torch.cat([pooled_cnn, mlp_output], dim=1)  # [B, 1024]
        gate = torch.sigmoid(self.gate_layer(gate_input))        # [B, 512]
        fused = gate * pooled_cnn + (1 - gate) * mlp_output      # [B, 512]

        # Final classification
        logits = self.classifier(fused)                          # [B, 28]
        return logits
# =======================
# Dataset Wrapper
# =======================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# =======================
# Training Function
# =======================
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        # Apply label smoothing
        smoothed_labels = smooth_labels(labels, smoothing=0.05)
        loss = loss_fn(logits, smoothed_labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    hamming = hamming_loss(all_labels, all_preds)
    return total_loss / len(dataloader), f1_micro, f1_macro, hamming

# =======================
# Evaluation Function
# =======================
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    hamming = hamming_loss(all_labels, all_preds)
    return total_loss / len(dataloader), f1_micro, f1_macro, hamming

# =======================
# Main Script
# =======================
label_merge_map = {
    'admiration': 'approval',
    'desire': 'joy',
    'relief': 'joy',
    'remorse': 'sadness',
    'nervousness': 'fear',
    'disappointment': 'sadness'
}

def main():
    # === Hyperparameters ===
    BATCH_SIZE = 16
    NUM_EPOCHS = 4
    # LEARNING_RATE = 2e-5

    # === Load Data ===
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def merge_label_list(label_list, merge_map):
        return list(set([merge_map.get(label, label) for label in label_list]))

    def load_data(path, merge_map=None):
        df = pd.read_csv(path)
        texts = df['text'].tolist()
        labels = [ast.literal_eval(vec) for vec in df['labels']]

        if merge_map:
            labels = [merge_label_list(label_list, merge_map) for label_list in labels]

        return texts, labels

    train_texts, train_labels = load_data(DATA_DIR / "go_emotions_train_balanced.csv", merge_map=label_merge_map)
    val_texts, val_labels = load_data(DATA_DIR / "go_emotions_val_balanced.csv", merge_map=label_merge_map)

    subset_size = 5000  # or 1000 for very fast debugging
    train_texts = train_texts[:subset_size]
    train_labels = train_labels[:subset_size]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels)
    y_val = mlb.transform(val_labels)

    train_dataset = EmotionDataset(train_texts, y_train, tokenizer)
    val_dataset = EmotionDataset(val_texts, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # === Model Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(set(label for label_list in train_labels for label in label_list))
    model = HybridBERTClassifier(num_labels=num_classes).to(device)

    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
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

    best_f1 = 0
    # === Training Loop ===
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_f1_micro, train_f1_macro, train_hamming = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_f1_micro, val_f1_macro, val_hamming = evaluate(model, val_loader, loss_fn, device)

        if val_f1_micro > best_f1:
            best_f1 = val_f1_micro
            torch.save(model.state_dict(), "best_hybrid_model.pt")
            print("Saved new best model with F1 Micro:", best_f1)
            
        print(f"Train Loss: {train_loss:.4f} | F1 Micro: {train_f1_micro:.4f} | F1 Macro: {train_f1_macro:.4f} | Hamming Loss: {train_hamming:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | F1 Micro: {val_f1_micro:.4f} | F1 Macro: {val_f1_macro:.4f} | Hamming Loss: {val_hamming:.4f}")

if __name__ == '__main__':
    main()