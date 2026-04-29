import sys
from functools import lru_cache
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "saved_models" / "best_resnet50.pth"
MODEL_NAME = "resnet50"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import get_model  # noqa: E402


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


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Model checkpoint was not found. Expected file: "
            f"{CHECKPOINT_PATH}. Train ResNet50 first or copy best_resnet50.pth there."
        )

    model = get_model(
        MODEL_NAME,
        freeze_backbone=False,
        use_pretrained_weights=False,
    )
    state_dict = _load_checkpoint(CHECKPOINT_PATH)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model
