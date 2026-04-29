import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from config import CLASS_NAMES, DEVICE, FIGURES_DIR, METRICS_DIR, NUM_CLASSES, SAVED_MODELS_DIR
from dataset import get_dataloaders
from models import get_model
from plots import plot_confusion_matrix
from utils import count_parameters, create_dirs, save_json, set_seed
from config import RANDOM_SEED


def load_state_dict(model_path):
    try:
        return torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=DEVICE)


def evaluate(model_name):
    set_seed(RANDOM_SEED)
    create_dirs()

    _, val_loader = get_dataloaders()
    model = get_model(model_name, freeze_backbone=False).to(DEVICE)

    model_path = SAVED_MODELS_DIR / f"best_{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    state_dict = load_state_dict(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predicted.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(NUM_CLASSES)))

    report_path = METRICS_DIR / f"{model_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    confusion_matrix_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, model_name, confusion_matrix_path)

    summary = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "parameters": count_parameters(model),
    }
    summary_path = METRICS_DIR / f"{model_name}_summary.json"
    save_json(summary, summary_path)

    print(report)
    print(f"Summary saved to {summary_path}")
    print(f"Confusion matrix saved to {confusion_matrix_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate APTOS diabetic retinopathy classifier.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn", "resnet50", "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(model_name=args.model)
