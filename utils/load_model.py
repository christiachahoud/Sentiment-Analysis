import torch
from transformers import BertModel
import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# === Define the full model class (copied from training) ===
class HybridBERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=28):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=k, padding=k // 2)
            for k in [2, 3, 4, 5]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)

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
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]

        cnn_input = sequence_output.transpose(1, 2)
        cnn_features = [self.pool(conv(cnn_input)).squeeze(-1) for conv in self.convs]
        pooled_cnn = torch.cat(cnn_features, dim=1)
        pooled_cnn = self.dropout(self.layernorm_cnn(pooled_cnn))

        mlp_output = self.mlp(cls_token)
        mlp_output = self.mlp_projection(mlp_output)
        mlp_output = self.dropout(self.layernorm_mlp(mlp_output))

        gate_input = torch.cat([pooled_cnn, mlp_output], dim=1)
        gate = torch.sigmoid(self.gate_layer(gate_input))
        fused = gate * pooled_cnn + (1 - gate) * mlp_output

        return self.classifier(fused)

# === Reusable loader ===
def load_hybrid_model(model_path="src/hybrid_classifier/best_hybrid_model.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridBERTClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
