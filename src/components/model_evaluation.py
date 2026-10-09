
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.logger import logger


def generate_evaluation_report(
    results,
    best_models,
    X_test,
    y_test,
):
    """Generate evaluation reports from already-trained models."""

    output_dir = Path("artifacts/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = []

    for model_name, model in best_models.items():
        # Reuse the fitted model; do not retrain.
        y_pred = model.predict(X_test)

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        # Use metrics already calculated on the held-out test set.
        test_metrics = results[model_name]["test_metrics"]

        comparison.append({
            "model": model_name,
            **test_metrics,
            "cv_roc_auc": results[model_name]["cv_roc_auc"],
        })

        # Save class-level classification report.
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Stayed (0)", "Churned (1)"],
            zero_division=0,
            output_dict=True,
        )

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]+", "_", model_name
        ).strip("_").lower()

        pd.DataFrame(report).transpose().to_csv(
            output_dir / f"{safe_name}_classification_report.csv"
        )

        # Save confusion matrix.
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Stayed", "Churned"],
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False,
        )
        ax.set_title(f"{model_name} — Test Set")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{safe_name}_confusion_matrix.png",
            dpi=150,
        )
        plt.close(fig)

    # Save side-by-side comparison.
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values(
        by="roc_auc",
        ascending=False,
    )

    comparison_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    with open(output_dir / "model_comparison.json", "w") as file:
        json.dump(
            comparison_df.to_dict(orient="records"),
            file,
            indent=4,
            default=float,
        )

    logger.info(
        f"Evaluation reports saved to {output_dir.resolve()}"
    )

    print("\nHELD-OUT TEST SET MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.round(4).to_string(index=False))
    print(f"\nReports saved to: {output_dir.resolve()}")

    return comparison_df
