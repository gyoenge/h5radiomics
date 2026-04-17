from __future__ import annotations

import copy


DEFAULT_CONFIG_COMMON = {
    "sample_ids": ["TENX95", "NCBI785", "NCBI783", "TENX99"],
    "h5_dir": "/root/workspace/hest-radiomics/data/h5",
    "output_root": "/root/workspace/hest-radiomics/data/outputs",
}


def get_common_default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG_COMMON)

