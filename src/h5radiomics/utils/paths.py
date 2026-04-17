from __future__ import annotations

import os


def get_feature_output_dir(output_root: str, sample_id: str) -> str:
    return os.path.join(output_root, "features", f"{sample_id}_features")


def get_statistics_output_dir(output_root: str, sample_id: str) -> str:
    return os.path.join(output_root, f"{sample_id}_stats")


def get_segment_output_dir(output_root: str, sample_id: str) -> str:
    return os.path.join(output_root, f"{sample_id}_seg")