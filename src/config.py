from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
IMAGE_DIR = DATA_DIR / "train_images"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
