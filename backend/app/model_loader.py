import sys
from functools import lru_cache
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SAVED_MODELS_DIR = PROJECT_ROOT / "results" / "saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_MODELS: dict[str, str] = {
    "resnet50": "best_resnet50.pth",
    "efficientnet_b0": "best_efficientnet_b0.pth",
    "mobilenet_v2": "best_mobilenet_v2.pth",
    "cnn": "best_cnn.pth",
}
DEFAULT_MODEL_NAME = "resnet50"

MODEL_NAME = DEFAULT_MODEL_NAME
CHECKPOINT_PATH = SAVED_MODELS_DIR / AVAILABLE_MODELS[DEFAULT_MODEL_NAME]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import get_model  # noqa: E402


def checkpoint_path_for(model_name: str) -> Path:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS)}"
        )
    return SAVED_MODELS_DIR / AVAILABLE_MODELS[model_name]


def available_models() -> list[dict]:
    return [
        {
            "name": name,
            "checkpoint_path": str(checkpoint_path_for(name)),
            "checkpoint_exists": checkpoint_path_for(name).exists(),
            "is_default": name == DEFAULT_MODEL_NAME,
        }
        for name in AVAILABLE_MODELS
    ]


def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL_NAME) -> torch.nn.Module:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS)}"
        )

    path = checkpoint_path_for(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint for '{model_name}' was not found at {path}. "
            f"Train it first or copy {AVAILABLE_MODELS[model_name]} there."
        )

    model = get_model(
        model_name,
        freeze_backbone=False,
        use_pretrained_weights=False,
    )
    state_dict = _load_checkpoint(path)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model
