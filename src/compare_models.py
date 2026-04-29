import json

import pandas as pd

from config import FIGURES_DIR, METRICS_DIR
from plots import plot_model_comparison
from utils import create_dirs


MODEL_NAMES = ["cnn", "resnet50", "efficientnet_b0", "mobilenet_v2"]
DISPLAY_NAMES = {
    "cnn": "Custom CNN",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v2": "MobileNetV2",
}


def compare_models():
    create_dirs()
    rows = []
    required_keys = ["accuracy", "precision", "recall", "f1_score", "parameters"]

    for model_name in MODEL_NAMES:
        summary_path = METRICS_DIR / f"{model_name}_summary.json"
        if not summary_path.exists():
            print(f"Skipping {model_name}: summary file not found at {summary_path}")
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                summary = json.load(file)
        except json.JSONDecodeError as error:
            print(f"Skipping {model_name}: invalid JSON in {summary_path}: {error}")
            continue

        missing_keys = [key for key in required_keys if key not in summary]
        if missing_keys:
            print(f"Skipping {model_name}: missing keys in summary: {missing_keys}")
            continue

        rows.append(
            {
                "Model": DISPLAY_NAMES[model_name],
                "Accuracy": summary["accuracy"],
                "Precision": summary["precision"],
                "Recall": summary["recall"],
                "F1-score": summary["f1_score"],
                "Parameters": summary["parameters"],
            }
        )

    if not rows:
        raise FileNotFoundError("No evaluation summaries found in results/metrics.")

    comparison_df = pd.DataFrame(rows)
    comparison_path = METRICS_DIR / "model_comparison.csv"
    figure_path = FIGURES_DIR / "model_comparison.png"

    comparison_df.to_csv(comparison_path, index=False)
    plot_model_comparison(comparison_df, figure_path)

    print(comparison_df)
    print(f"Model comparison saved to {comparison_path}")
    print(f"Model comparison plot saved to {figure_path}")


if __name__ == "__main__":
    compare_models()
