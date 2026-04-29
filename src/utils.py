import json
import random

import numpy as np
import pandas as pd
import torch

from config import FIGURES_DIR, METRICS_DIR, RESULTS_DIR, SAVED_MODELS_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dirs():
    for directory in [RESULTS_DIR, FIGURES_DIR, METRICS_DIR, SAVED_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_history(history, save_path):
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_path, index=False)
    return history_df


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device": "cpu"}

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)

    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device_index),
        "total_memory_gb": round(properties.total_memory / (1024**3), 2),
    }


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
