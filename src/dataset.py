from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
    TRAIN_CSV,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class APTOSDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / f"{row['id_code']}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        label = int(row["diagnosis"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _validate_dataset_paths(train_csv, image_dir):
    if not Path(train_csv).exists():
        raise FileNotFoundError(
            f"train.csv not found: {train_csv}. Place APTOS train.csv in the data directory."
        )

    if not Path(image_dir).exists():
        raise FileNotFoundError(
            f"train_images directory not found: {image_dir}. Place APTOS images in data/train_images."
        )


def get_class_weights(train_csv=TRAIN_CSV):
    if not Path(train_csv).exists():
        raise FileNotFoundError(
            f"train.csv not found: {train_csv}. Place APTOS train.csv in the data directory."
        )

    df = pd.read_csv(train_csv)
    if "diagnosis" not in df.columns:
        raise ValueError("train.csv is missing required column: diagnosis")

    class_counts = df["diagnosis"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    total_samples = len(df)

    weights = []
    for class_count in class_counts:
        if class_count > 0:
            weights.append(total_samples / (NUM_CLASSES * class_count))
        else:
            weights.append(0.0)

    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    train_csv=TRAIN_CSV,
    image_dir=IMAGE_DIR,
    batch_size=BATCH_SIZE,
    validation_size=0.2,
):
    _validate_dataset_paths(train_csv, image_dir)

    df = pd.read_csv(train_csv)
    required_columns = {"id_code", "diagnosis"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing_columns)}")

    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=RANDOM_SEED,
        stratify=df["diagnosis"],
    )

    train_dataset = APTOSDataset(train_df, image_dir, transform=get_train_transforms())
    val_dataset = APTOSDataset(val_df, image_dir, transform=get_val_transforms())

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
