import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    DEVICE,
    FIGURES_DIR,
    LEARNING_RATE,
    METRICS_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
    SAVED_MODELS_DIR,
)
from dataset import get_class_weights, get_dataloaders
from models import get_model
from plots import plot_training_history
from utils import count_parameters, create_dirs, get_gpu_info, save_history, save_json, set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, dim=1)
        total += batch_size
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    return running_loss / total, correct / total


def validate_one_epoch(model, dataloader, criterion, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, dim=1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    return running_loss / total, correct / total


def train(model_name, freeze_backbone=False):
    set_seed(RANDOM_SEED)
    create_dirs()

    gpu_info = get_gpu_info()
    print(f"Device: {DEVICE}")
    if gpu_info["cuda_available"]:
        print(f"GPU: {gpu_info['device']} ({gpu_info['total_memory_gb']} GB)")

    train_loader, val_loader = get_dataloaders()
    model = get_model(model_name, freeze_backbone=freeze_backbone).to(DEVICE)

    class_weights = get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    history = []
    best_val_accuracy = 0.0
    best_model_path = SAVED_MODELS_DIR / f"best_{model_name}.pth"
    parameters = count_parameters(model)
    trainable_parameters = count_parameters(model, trainable_only=True)

    print(f"Model: {model_name}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Parameters: {parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp
        )
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, use_amp)
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )

        print(
            "Train loss: "
            f"{train_loss:.4f} | Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6g}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    history_path = METRICS_DIR / f"{model_name}_history.csv"
    history_df = save_history(history, history_path)

    figure_path = FIGURES_DIR / f"{model_name}_training_history.png"
    plot_training_history(history_df, model_name, figure_path)

    final_metrics = history[-1]
    training_summary = {
        "model": model_name,
        "freeze_backbone": freeze_backbone,
        "best_val_accuracy": best_val_accuracy,
        "final_train_accuracy": final_metrics["train_accuracy"],
        "final_val_accuracy": final_metrics["val_accuracy"],
        "final_train_loss": final_metrics["train_loss"],
        "final_val_loss": final_metrics["val_loss"],
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "class_weights": class_weights.detach().cpu().tolist(),
    }
    summary_path = METRICS_DIR / f"{model_name}_training_summary.json"
    save_json(training_summary, summary_path)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Training history saved to {history_path}")
    print(f"Training plots saved to {figure_path}")
    print(f"Training summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train APTOS diabetic retinopathy classifier.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn", "resnet50", "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze pretrained backbone and train only the classifier head.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_name=args.model, freeze_backbone=args.freeze_backbone)
