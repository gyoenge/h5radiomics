from __future__ import annotations

import os


# =========================
# root
# =========================
def get_sample_root(output_dir: str, sample_id: str) -> str:
    return os.path.join(output_dir, sample_id)


# =========================
# patches
# =========================
def get_patches_root(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_sample_root(output_dir, sample_id), "patches")


def get_patch_color_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_patches_root(output_dir, sample_id), "color")


def get_patch_gray_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_patches_root(output_dir, sample_id), "gray")


def get_patch_mask_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_patches_root(output_dir, sample_id), "mask")


def get_patch_masked_color_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_patches_root(output_dir, sample_id), "masked_color")


def get_patch_masked_gray_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_patches_root(output_dir, sample_id), "masked_gray")


# =========================
# features
# =========================
def get_features_root(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_sample_root(output_dir, sample_id), "features")


def get_raw_features_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_features_root(output_dir, sample_id), "raw")


def get_processed_features_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_features_root(output_dir, sample_id), "processed")


def get_raw_features_csv_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_raw_features_dir(output_dir, sample_id), "features.csv")


def get_raw_features_parquet_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_raw_features_dir(output_dir, sample_id), "features.parquet")


def get_processed_features_csv_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_processed_features_dir(output_dir, sample_id), "features.csv")


def get_processed_features_parquet_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_processed_features_dir(output_dir, sample_id), "features.parquet")


def get_processing_stats_csv_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_processed_features_dir(output_dir, sample_id), "processing_stats.csv")


def get_processing_config_json_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_processed_features_dir(output_dir, sample_id), "processing_config.json")


# =========================
# feature statistics
# =========================
def get_statistics_root(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_features_root(output_dir, sample_id), "statistics")


def get_raw_statistics_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_statistics_root(output_dir, sample_id), "raw")


def get_processed_statistics_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_statistics_root(output_dir, sample_id), "processed")


def get_statistics_dir(output_dir: str, sample_id: str, feature_type: str) -> str:
    if feature_type not in ("raw", "processed"):
        raise ValueError(f"feature_type must be 'raw' or 'processed', got: {feature_type}")
    return os.path.join(get_statistics_root(output_dir, sample_id), feature_type)


def get_statistics_csv_path(output_dir: str, sample_id: str, feature_type: str) -> str:
    return os.path.join(get_statistics_dir(output_dir, sample_id, feature_type), "stats.csv")


def get_statistics_parquet_path(output_dir: str, sample_id: str, feature_type: str) -> str:
    return os.path.join(get_statistics_dir(output_dir, sample_id, feature_type), "stats.parquet")


def get_statistics_representative_dir(output_dir: str, sample_id: str, feature_type: str) -> str:
    return os.path.join(get_statistics_dir(output_dir, sample_id, feature_type), "representative")


def get_statistics_boxplots_dir(output_dir: str, sample_id: str, feature_type: str) -> str:
    return os.path.join(get_statistics_dir(output_dir, sample_id, feature_type), "boxplots")


def get_feature_csv_path(output_dir: str, sample_id: str, feature_type: str) -> str:
    if feature_type == "raw":
        return get_raw_features_csv_path(output_dir, sample_id)
    if feature_type == "processed":
        return get_processed_features_csv_path(output_dir, sample_id)
    raise ValueError(f"feature_type must be 'raw' or 'processed', got: {feature_type}")


def get_feature_parquet_path(output_dir: str, sample_id: str, feature_type: str) -> str:
    if feature_type == "raw":
        return get_raw_features_parquet_path(output_dir, sample_id)
    if feature_type == "processed":
        return get_processed_features_parquet_path(output_dir, sample_id)
    raise ValueError(f"feature_type must be 'raw' or 'processed', got: {feature_type}")


# =========================
# cellvit segmentation
# =========================
def get_cellvitseg_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_sample_root(output_dir, sample_id), "cellvitseg")


def get_cellvit_overlay_dir(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_cellvitseg_dir(output_dir, sample_id), "overlay")


def get_cellseg_geojson_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_cellvitseg_dir(output_dir, sample_id), "cellseg.geojson")


def get_cellseg_parquet_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_cellvitseg_dir(output_dir, sample_id), "cellseg.parquet")


def get_cellseg_metadata_csv_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_cellvitseg_dir(output_dir, sample_id), "metadata.csv")


def get_cellseg_summary_json_path(output_dir: str, sample_id: str) -> str:
    return os.path.join(get_cellvitseg_dir(output_dir, sample_id), "summary.json")

