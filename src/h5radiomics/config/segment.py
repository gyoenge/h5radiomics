from __future__ import annotations

import copy
from .base import get_common_default_config


DEFAULT_CONFIG_SEGMENT = {
    "model_dir": "/root/workspace/h5radiomics/models",
    "model_name": "CellViT-SAM-H-x20.pth",
    "batch_size": 8,
    "num_workers": 0,
    "device": "cuda:0",
    "patch_indices": None,
    "save_png_overlay": True,
    "use_class_color": True,
    "save_geojson_per_patch": False,
    "postprocess_threads": 1,
    "verbose": False,
}


def get_segment_default_config() -> dict:
    config = get_common_default_config()
    config.update(copy.deepcopy(DEFAULT_CONFIG_SEGMENT))
    return config

