import pandas as pd, numpy as np, torch, ast
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
from model import EmotionDataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_NAME = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
label_merge_map = {
    'admiration': 'approval',
    'desire': 'joy',
    'relief': 'joy',
    'remorse': 'sadness',
    'nervousness': 'fear',
    'disappointment': 'sadness'
}

def merge_label_list(label_list, merge_map):
    return list(set([merge_map.get(label, label) for label in label_list]))

def load_data(path, merge_map):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    labels = [merge_label_list(ast.literal_eval(l), merge_map) for l in df['labels']]
    return texts, labels

train_texts, train_labels = load_data(DATA_DIR / "go_emotions_train_balanced.csv", label_merge_map)
val_texts, val_labels = load_data(DATA_DIR / "go_emotions_val_balanced.csv", label_merge_map)

subset_size = 5000
train_texts = train_texts[:subset_size]
train_labels = train_labels[:subset_size]

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
y_val = mlb.transform(val_labels)

np.save(OUTPUT_DIR / "mlb_classes.npy", mlb.classes_)

train_dataset = EmotionDataset(train_texts, y_train, tokenizer)
val_dataset = EmotionDataset(val_texts, y_val, tokenizer)

torch.save(train_dataset, OUTPUT_DIR / "train_dataset.pt")
torch.save(val_dataset, OUTPUT_DIR / "val_dataset.pt")

print("\nPreprocessing complete.")
print(f"Saved artifacts to: {OUTPUT_DIR}")
print(f"- train_dataset.pt ({len(train_dataset)} samples)")
print(f"- val_dataset.pt   ({len(val_dataset)} samples)")
print(f"- mlb_classes.npy  ({len(mlb.classes_)} labels): {list(mlb.classes_)}")
