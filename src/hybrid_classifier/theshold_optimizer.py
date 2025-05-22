import numpy as np
from sklearn.metrics import precision_recall_curve

# Load ground truth and predicted probabilities
y_true = np.load("src/hybrid_classifier/evaluation/y_true.npy")
y_probs = np.load("src/hybrid_classifier/evaluation/y_pred_probs.npy")

optimal_thresholds = []

for i in range(y_true.shape[1]):
    precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    if len(thresholds) == 0:  # Safeguard in case of edge case
        best_thresh = 0.5
    else:
        best_thresh = thresholds[np.argmax(f1)]
    optimal_thresholds.append(best_thresh)
    print(f"Label {i}: best threshold = {best_thresh:.3f}")

# Save thresholds
np.save("optimal_thresholds.npy", np.array(optimal_thresholds))
print("✅ Saved: optimal_thresholds.npy")
