import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.load_model import load_hybrid_model

# ======== Dataset definition =========
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
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Ensure fixed-length tensor
        if label.shape != (28,):
            print(f"Bad label shape at index {idx}: {label.shape}")
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label  # Already tensor
        }

# ======== Load your validation CSV and parse ========
def load_data(path):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    labels = [ast.literal_eval(vec) for vec in df['labels']]  # Or 'label_vector' if that's your column
    return texts, labels

# ======== Main Inference Function ========
def run_inference_and_export():
    val_texts, val_labels = load_data("data/go_emotions_val_balanced.csv")
    mlb = MultiLabelBinarizer(classes=list(range(28)))
    y_val = mlb.fit_transform(val_labels)               # Now shape is (N, 28)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    val_dataset = EmotionDataset(val_texts, y_val, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_hybrid_model("src/hybrid_classifier/best_hybrid_model.pt", device=device)

    all_labels, all_probs  = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Running Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    y_true = np.vstack(all_labels)
    y_probs = np.vstack(all_probs)

    # Load per-label optimized thresholds
    thresholds = np.load("data/optimal_thresholds.npy")

    # Compute binary predictions
    y_pred_default = (y_probs > 0.5).astype(int)
    y_pred_optimized = (y_probs > thresholds).astype(int)

    # Save outputs
    np.save("y_true.npy", y_true)
    np.save("y_pred_probs.npy", y_probs)
    np.save("y_pred_default.npy", y_pred_default)
    np.save("y_pred_optimized.npy", y_pred_optimized)

    print("✅ Saved: y_true.npy, y_pred_probs.npy, y_pred_default.npy, y_pred_optimized.npy")

if __name__ == "__main__":
    run_inference_and_export()
