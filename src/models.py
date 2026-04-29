import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
)

from config import NUM_CLASSES


class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _freeze_except_head(model, head_prefixes):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in head_prefixes):
            parameter.requires_grad = True


def get_model(model_name, freeze_backbone=False, use_pretrained_weights=True):
    model_name = model_name.lower()

    if model_name == "cnn":
        return CustomCNN(num_classes=NUM_CLASSES)

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if use_pretrained_weights else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        if freeze_backbone:
            _freeze_except_head(model, head_prefixes=["fc."])
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained_weights else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        if freeze_backbone:
            _freeze_except_head(model, head_prefixes=["classifier."])
        return model

    if model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if use_pretrained_weights else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        if freeze_backbone:
            _freeze_except_head(model, head_prefixes=["classifier."])
        return model

    raise ValueError(
        "Unsupported model_name. Use one of: cnn, resnet50, efficientnet_b0, mobilenet_v2."
    )
