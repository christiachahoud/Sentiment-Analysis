import re
import html
import emoji
from transformers import AutoTokenizer
import unicodedata
import contractions

def clean_text(text):
    text = html.unescape(text)                          # Convert HTML entities (e.g., &amp;)
    text = emoji.demojize(text, delimiters=(" ", " "))  # Convert emojis to text (e.g., 😂 → :face_with_tears_of_joy:)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text) # Remove URLs
    text = re.sub(r"@\w+", '', text)                    # Remove mentions
    text = re.sub(r"#\w+", '', text)                    # Remove hashtags (optional)
    text = re.sub(r"\s+", ' ', text).strip()            # Normalize whitespace
    return text


def normalize_text(text, lowercase=True, remove_accents=True):
    if lowercase:
        text = text.lower()
    if remove_accents:
        text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    text = contractions.fix(text)
    return text

# Set later depending on chosen tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/your/model_or_tokenizer")

def tokenize_text(text, max_length=128):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",  # for PyTorch; use "tf" if using TensorFlow
    )

def preprocess_text_for_inference(raw_text, max_length=128):
    cleaned = clean_text(raw_text)
    normalized = normalize_text(cleaned)
    tokenized = tokenize_text(normalized, max_length=max_length)
    return tokenized

def preprocess_batch(text_list, max_length=128):
    cleaned = [normalize_text(clean_text(t)) for t in text_list]
    return tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )