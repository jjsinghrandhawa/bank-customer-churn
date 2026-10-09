
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "models" / "best_model.pkl"
DATA_PATH = Path(r"F:\CV\archive\Customer-Churn-Records.csv")
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the final saved model
import joblib

model = joblib.load(MODEL_PATH)

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Match the target and feature preparation used in training.
# Complain is excluded to prevent target leakage.
X = df.drop(columns=["Exited", "Complain"], errors="ignore")
y = df["Exited"]

# Evaluate the saved model on the full dataset.
# IMPORTANT: this is a diagnostic report, not an unbiased
# test-set estimate, because the model was trained on part
# of this dataset.
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred, zero_division=0),
    "recall": recall_score(y, y_pred, zero_division=0),
    "f1": f1_score(y, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y, y_prob),
    "pr_auc": average_precision_score(y, y_prob),
}

# Save metrics
with open(OUTPUT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMODEL EVALUATION METRICS")
print("=" * 45)

for metric, value in metrics.items():
    print(f"{metric:15s}: {value:.4f}")

# Save classification report
report = classification_report(
    y,
    y_pred,
    target_names=["Stayed (0)", "Churned (1)"],
    zero_division=0,
    output_dict=True,
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

# Plot confusion matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"],
)

disp.plot(values_format="d")
plt.title("Bank Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.show()

print(f"\nEvaluation files saved to: {OUTPUT_DIR}")
