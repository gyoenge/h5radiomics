from __future__ import annotations

import copy
from .base import get_common_default_config


DEFAULT_CONFIG_STATISTICS = {
    "status_filter": "ok",
    "drop_diagnostic": True,
    "save_per_sample": True,
    "save_merged": False,
    "save_representatives": True,
    "representative_image_col": "color_path",
    "representative_stats": ["min", "q10", "q25", "q50", "q75", "q90", "max"],
    "save_boxplot": True,
}


def get_statistics_default_config() -> dict:
    config = get_common_default_config()
    config.update(copy.deepcopy(DEFAULT_CONFIG_STATISTICS))
    return config

