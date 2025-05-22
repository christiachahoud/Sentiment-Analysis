import torch, numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import f1_score, hamming_loss, classification_report
from torch.utils.data import DataLoader
from model import EmotionDataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import ast

# ========== CONFIG ==========
MODEL_NAME = "bert-base-multilingual-cased"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
THRESHOLD = 0.3  # Adjusted threshold
UNFREEZE_LAYERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ========== DATA LOADING ==========
def load_data(path, mlb_path):
    df = pd.read_csv(path)
    df['labels'] = df['labels'].apply(ast.literal_eval)
    texts = df['text'].tolist()

    classes = np.load(mlb_path, allow_pickle=True)
    label_to_idx = {label: i for i, label in enumerate(classes)}

    y = []
    for label_list in df['labels']:
        vec = [0] * len(classes)
        for lbl in label_list:
            if lbl in label_to_idx:
                vec[label_to_idx[lbl]] = 1
        y.append(vec)

    return texts, y

train_texts, train_labels = load_data(DATA_DIR / "go_emotions_train_balanced.csv", ARTIFACT_DIR / "mlb_classes.npy")
val_texts, val_labels = load_data(DATA_DIR / "go_emotions_val_balanced.csv", ARTIFACT_DIR / "mlb_classes.npy")

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== MODEL ==========
class HybridMBERTClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.3, cnn_filters=100, cnn_kernel_size=3, mlp_hidden=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.bert.config.hidden_size

        self.cnn = nn.Conv1d(hidden, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2)
        self.cnn_dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gate_fc = nn.Linear(cnn_filters + mlp_hidden, cnn_filters + mlp_hidden)
        self.classifier = nn.Linear(cnn_filters + mlp_hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        last_hidden = out.last_hidden_state

        # === IMPROVEMENT 1: Mean pooling instead of pooler_output ===
        pooled_output = last_hidden.mean(dim=1)

        # CNN over token embeddings
        x_cnn = self.cnn(last_hidden.transpose(1, 2))
        x_cnn = torch.max(torch.relu(x_cnn), dim=2).values
        x_cnn = self.cnn_dropout(x_cnn)

        # MLP on mean pooled BERT
        x_mlp = self.mlp(pooled_output)

        # Gated fusion
        x = torch.cat([x_cnn, x_mlp], dim=1)
        gate = torch.sigmoid(self.gate_fc(x))
        x = x * gate

        return self.classifier(x)

# ========== LOSS ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ========== TRAINING ==========
def freeze_layers(model, unfreeze_last=4):
    total = len(model.bert.encoder.layer)
    for i, layer in enumerate(model.bert.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = (i >= total - unfreeze_last)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            preds = (torch.sigmoid(logits) > THRESHOLD).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    print(classification_report(all_labels, all_preds, zero_division=0))
    return np.mean(losses), f1_micro, f1_macro, hamming

# ========== MAIN ==========
model = HybridMBERTClassifier(num_labels=len(train_labels[0])).to(DEVICE)
freeze_layers(model, UNFREEZE_LAYERS)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = FocalLoss()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    val_loss, val_f1_micro, val_f1_macro, val_hamming = evaluate(model, val_loader)
    print(f"Train Loss: {epoch_loss / len(train_loader):.4f}")
    print(f"Val   Loss: {val_loss:.4f} | F1 Micro: {val_f1_micro:.4f} | F1 Macro: {val_f1_macro:.4f} | Hamming: {val_hamming:.4f}")
