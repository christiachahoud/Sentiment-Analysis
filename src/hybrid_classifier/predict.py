import torch
from transformers import BertTokenizer
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.load_model import load_hybrid_model

# === Emotion Labels (GoEmotions 28) ===
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# === Inference Function ===
def predict_emotions(text, model, tokenizer, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize input
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Apply threshold
    pred_indices = np.where(probs >= threshold)[0]
    predictions = [(EMOTION_LABELS[i], float(probs[i])) for i in pred_indices]

    return predictions

# === Example usage ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = load_hybrid_model(device=device)

    text_input = "millions of kids are being murdered in gaza"
    preds = predict_emotions(text_input, model, tokenizer)

    print("\nPredicted Emotions:")
    if preds:
        for emotion, prob in preds:
            print(f"  - {emotion}: {prob:.2f}")
    else:
        print("No emotions above threshold.")
