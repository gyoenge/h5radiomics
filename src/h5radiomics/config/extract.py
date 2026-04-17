from __future__ import annotations

import copy
from .base import get_common_default_config


DEFAULT_CONFIG_EXTRACT = {
    "num_workers": 0,
    "save_patches": True,
    "label": 255,
    "classes": ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
    "filters": ["Original"],
    "image_type_settings": {
        "LoG": {
            "sigma": [1.0, 2.0, 3.0],
        }
    },
    "processing": {
        "lower_q": 0.01,
        "upper_q": 0.99,
        "save_processed": True,
    },
}


def get_extract_default_config() -> dict:
    config = get_common_default_config()
    config.update(copy.deepcopy(DEFAULT_CONFIG_EXTRACT))
    return config
