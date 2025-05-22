import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

# === Label Names from GoEmotions ===
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

# === Load Predictions and Ground Truth ===
y_true = np.load("src/hybrid_classifier/evaluation/y_true.npy")
y_pred_default = np.load("src/hybrid_classifier/evaluation/y_pred_default.npy")
y_pred_optimized = np.load("src/hybrid_classifier/evaluation/y_pred_optimized.npy")

# === Evaluate Metrics ===
def get_metrics(y_pred, label=""):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    df = pd.DataFrame({
        'Label': EMOTION_LABELS,
        f'Precision_{label}': precision,
        f'Recall_{label}': recall,
        f'F1_{label}': f1,
        'Support': support  # same for both
    })
    return df

df_default = get_metrics(y_pred_default, label="default")
df_optimized = get_metrics(y_pred_optimized, label="opt")

# === Merge Results ===
df_metrics = df_default.merge(df_optimized, on=['Label', 'Support'])
df_metrics['F1_improvement'] = df_metrics['F1_opt'] - df_metrics['F1_default']

# # === Save as CSV ===
# df_metrics.to_csv("per_label_metrics_comparison.csv", index=False)
# print("Saved: per_label_metrics_comparison.csv")

# # === Display Results ===
# print("\n📉 Worst 5 labels by F1 (default):")
# print(df_metrics.sort_values("F1_default").head(5))

# print("\n📈 Biggest improvements after threshold optimization:")
# print(df_metrics.sort_values("F1_improvement", ascending=False).head(5))

# === Optional: Confusion Matrix Plot ===
def plot_confusion_for_label(label_name, use="default"):
    idx = EMOTION_LABELS.index(label_name)
    cm = multilabel_confusion_matrix(
        y_true, y_pred_default if use == "default" else y_pred_optimized
    )[idx]
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not '+label_name, label_name],
                yticklabels=['Not '+label_name, label_name])
    plt.title(f"Confusion Matrix for '{label_name}' ({use})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_confusion_for_label("annoyance", use="opt")

# === Grouped Barplot: F1 Default vs Optimized ===
df_plot = df_metrics.sort_values("F1_opt", ascending=False)[["Label", "F1_default", "F1_opt"]]

# Melt the DataFrame for seaborn
df_melted = df_plot.melt(id_vars="Label", value_vars=["F1_default", "F1_opt"],
                         var_name="Thresholding", value_name="F1-score")

plt.figure(figsize=(14, 7))
sns.barplot(data=df_melted, x="F1-score", y="Label", hue="Thresholding", palette=["#4C72B0", "#55A868"])
plt.title("📊 F1-Score per Label: Default vs Optimized Thresholds")
plt.xlabel("F1-score")
plt.ylabel("Emotion Label")
plt.legend(title="Thresholding")
plt.tight_layout()
plt.savefig("f1_grouped_barplot.png", dpi=300)
plt.show()

print("📊 Saved: f1_grouped_barplot.png")
