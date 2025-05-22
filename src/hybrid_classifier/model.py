"""
Model and Dataset Definition for GoEmotions Multi-Label Classifier
- Hybrid BERT + CNN + MLP architecture
- EmotionDataset for tokenization and label loading
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel

# ========== Dataset Wrapper ==========
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

# ========== Hybrid BERT Model ==========
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
        self.mlp_projection = nn.Linear(128, 512)

        self.layernorm_cnn = nn.LayerNorm(512)
        self.layernorm_mlp = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)

        self.gate_layer = nn.Linear(512 + 512, 512)

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
        cnn_features = [self.pool(conv(cnn_input)).squeeze(-1) for conv in self.convs]
        pooled_cnn = torch.cat(cnn_features, dim=1)
        pooled_cnn = self.dropout(self.layernorm_cnn(pooled_cnn))

        # MLP branch
        mlp_output = self.mlp(cls_token)
        mlp_output = self.mlp_projection(mlp_output)
        mlp_output = self.dropout(self.layernorm_mlp(mlp_output))

        # Gated fusion
        gate_input = torch.cat([pooled_cnn, mlp_output], dim=1)
        gate = torch.sigmoid(self.gate_layer(gate_input))
        fused = gate * pooled_cnn + (1 - gate) * mlp_output

        logits = self.classifier(fused)
        return logits
