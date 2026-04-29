import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(history_df, model_name, save_path):
    has_learning_rate = "learning_rate" in history_df.columns
    if has_learning_rate:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    axes[0].set_title(f"{model_name} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train accuracy")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation accuracy")
    axes[1].set_title(f"{model_name} accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if has_learning_rate:
        axes[2].plot(history_df["epoch"], history_df["learning_rate"], label="Learning rate")
        axes[2].set_title(f"{model_name} learning rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning rate")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)

    ax.set_title(f"{model_name} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_df, save_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(comparison_df["Model"]))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - 1.5) * width, comparison_df[metric], width, label=metric)

    ax.set_title("Model comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
