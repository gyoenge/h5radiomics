from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hestradiomics.utils import (
    get_feature_csv_path,
    get_statistics_dir,
)


def load_feature_csv(
    csv_path: str,
    status_filter: Optional[str] = "ok",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if status_filter is not None and "status" in df.columns:
        df = df[df["status"] == status_filter].copy()

    return df


def get_feature_columns(
    df: pd.DataFrame,
    drop_diagnostic: bool = True,
) -> List[str]:
    meta_cols = {
        "patch_idx",
        "barcode",
        "color_path",
        "gray_path",
        "mask_path",
        "x",
        "y",
        "status",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in meta_cols]

    if drop_diagnostic:
        feature_cols = [
            col for col in feature_cols
            if not col.startswith("diagnostics_")
        ]

    return feature_cols


def compute_feature_statistics(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        rows.append(
            {
                "feature_name": col,
                "count": int(s.count()),
                "nan_count": int(s.isna().sum()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "q01": s.quantile(0.01),
                "q05": s.quantile(0.05),
                "q10": s.quantile(0.10),
                "q25": s.quantile(0.25),
                "median": s.median(),
                "q50": s.quantile(0.50),
                "q75": s.quantile(0.75),
                "q90": s.quantile(0.90),
                "q95": s.quantile(0.95),
                "q99": s.quantile(0.99),
                "max": s.max(),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "abs_mean": s.abs().mean(),
                "zero_count": int((s == 0).sum()),
                "positive_count": int((s > 0).sum()),
                "negative_count": int((s < 0).sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("feature_name")
        .reset_index(drop=True)
    )


def summarize_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, int]:
    return {
        "num_rows": len(df),
        "num_feature_columns": len(feature_cols),
        "num_total_columns": len(df.columns),
    }


def save_statistics(
    stats_df: pd.DataFrame,
    output_dir: str,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "stats.csv")
    parquet_path = os.path.join(output_dir, "stats.parquet")

    stats_df.to_csv(csv_path, index=False)
    stats_df.to_parquet(parquet_path, index=False)

    return csv_path, parquet_path


def sanitize_filename(text) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180] if len(text) > 180 else text


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_patch_path(
    row: pd.Series,
    preferred_col: str = "color_path",
) -> Optional[str]:
    candidates = [preferred_col, "color_path", "gray_path", "mask_path"]
    seen = set()

    for col in candidates:
        if col in seen:
            continue

        seen.add(col)

        if col in row.index:
            value = row[col]

            if pd.notna(value) and str(value).strip() != "":
                return str(value)

    return None


def get_target_stat_values(
    series: pd.Series,
) -> Optional[Dict[str, float]]:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return None

    return {
        "min": s.min(),
        "q10": s.quantile(0.10),
        "q25": s.quantile(0.25),
        "q50": s.quantile(0.50),
        "q75": s.quantile(0.75),
        "q90": s.quantile(0.90),
        "max": s.max(),
    }


def select_representative_row(
    df: pd.DataFrame,
    feature_col: str,
    target_value: float,
    stat_name: str,
):
    temp = df.copy()
    temp[feature_col] = pd.to_numeric(temp[feature_col], errors="coerce")
    temp = temp[temp[feature_col].notna()].copy()

    if len(temp) == 0:
        return None, None

    if stat_name == "min":
        idx = temp[feature_col].idxmin()
    elif stat_name == "max":
        idx = temp[feature_col].idxmax()
    else:
        temp["_abs_diff_"] = (temp[feature_col] - target_value).abs()
        idx = temp["_abs_diff_"].idxmin()

    row = temp.loc[idx]
    abs_diff = abs(float(row[feature_col]) - float(target_value))

    return row, abs_diff


def save_representative_patches(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str,
    config: dict,
    prefix: str = "sample",
) -> str:
    rep_root = os.path.join(output_dir, "representative")
    ensure_dir(rep_root)

    requested_stats = config.get(
        "representative_stats",
        ["min", "q10", "q25", "q50", "q75", "q90", "max"],
    )
    preferred_image_col = config.get("representative_image_col", "color_path")

    manifest_rows = []

    for feature_col in feature_cols:
        stat_targets = get_target_stat_values(df[feature_col])

        if stat_targets is None:
            continue

        feature_dir = os.path.join(rep_root, sanitize_filename(feature_col))
        ensure_dir(feature_dir)

        for stat_name in requested_stats:
            if stat_name not in stat_targets:
                continue

            target_value = stat_targets[stat_name]

            row, abs_diff = select_representative_row(
                df=df,
                feature_col=feature_col,
                target_value=target_value,
                stat_name=stat_name,
            )

            if row is None:
                manifest_rows.append(
                    {
                        "feature_name": feature_col,
                        "stat_name": stat_name,
                        "target_value": target_value,
                        "selected_value": np.nan,
                        "abs_diff": np.nan,
                        "patch_idx": np.nan,
                        "barcode": "",
                        "sample_id": prefix,
                        "source_path": "",
                        "saved_path": "",
                        "status": "no_valid_row",
                    }
                )
                continue

            src_path = resolve_patch_path(
                row,
                preferred_col=preferred_image_col,
            )

            if src_path is None or not os.path.exists(src_path):
                manifest_rows.append(
                    {
                        "feature_name": feature_col,
                        "stat_name": stat_name,
                        "target_value": float(target_value),
                        "selected_value": float(row[feature_col]),
                        "abs_diff": float(abs_diff),
                        "patch_idx": row.get("patch_idx", np.nan),
                        "barcode": row.get("barcode", ""),
                        "sample_id": row.get("sample_id", prefix),
                        "source_path": src_path or "",
                        "saved_path": "",
                        "status": "missing_source_image",
                    }
                )
                continue

            patch_idx = row.get("patch_idx", "na")
            sample_id = row.get("sample_id", prefix)
            barcode = row.get("barcode", "")

            src_ext = os.path.splitext(src_path)[1] or ".png"

            out_name = (
                f"{stat_name}__sample_{sanitize_filename(sample_id)}"
                f"__patch_{patch_idx}{src_ext}"
            )
            dst_path = os.path.join(feature_dir, out_name)

            shutil.copy2(src_path, dst_path)

            manifest_rows.append(
                {
                    "feature_name": feature_col,
                    "stat_name": stat_name,
                    "target_value": float(target_value),
                    "selected_value": float(row[feature_col]),
                    "abs_diff": float(abs_diff),
                    "patch_idx": patch_idx,
                    "barcode": barcode,
                    "sample_id": sample_id,
                    "source_path": src_path,
                    "saved_path": dst_path,
                    "status": "ok",
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(rep_root, "representative_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)

    return manifest_csv


def extract_feature_class(
    feature_name: str,
) -> str:
    parts = str(feature_name).split("_")

    if len(parts) >= 2:
        return parts[1].lower()

    return "unknown"


def save_sample_feature_boxplot(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str,
    prefix: str,
):
    if len(feature_cols) == 0:
        return None

    boxplot_dir = os.path.join(output_dir, "boxplots")
    ensure_dir(boxplot_dir)

    requested_stats = ["min", "q10", "q25", "q50", "q75", "q90", "max"]

    data_for_boxplot = []
    scatter_x = []
    scatter_y = []

    for i, feature_col in enumerate(feature_cols, start=1):
        s = pd.to_numeric(df[feature_col], errors="coerce").dropna()

        if len(s) == 0:
            data_for_boxplot.append(np.array([np.nan]))
            continue

        data_for_boxplot.append(s.values)

        stat_targets = get_target_stat_values(df[feature_col])

        if stat_targets is None:
            continue

        for stat_name in requested_stats:
            row, _ = select_representative_row(
                df=df,
                feature_col=feature_col,
                target_value=stat_targets[stat_name],
                stat_name=stat_name,
            )

            if row is None:
                continue

            actual_value = pd.to_numeric(row[feature_col], errors="coerce")

            if pd.isna(actual_value):
                continue

            scatter_x.append(i)
            scatter_y.append(float(actual_value))

    fig_width = max(16, len(feature_cols) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.boxplot(
        data_for_boxplot,
        showfliers=False,
        patch_artist=False,
        widths=0.5,
    )

    if scatter_x:
        ax.scatter(scatter_x, scatter_y, s=10, alpha=0.9)

    ax.set_title(f"Radiomics Feature Distribution ({prefix})")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature value")
    ax.set_xticks(range(1, len(feature_cols) + 1))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(boxplot_dir, f"{prefix}_feature_boxplot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path


def save_sample_feature_boxplots_by_class(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str,
    prefix: str,
):
    if len(feature_cols) == 0:
        return []

    boxplot_dir = os.path.join(output_dir, "boxplots")
    ensure_dir(boxplot_dir)

    class_to_features = {}

    for col in feature_cols:
        cls = extract_feature_class(col)
        class_to_features.setdefault(cls, []).append(col)

    saved_paths = []

    for feature_class, class_features in sorted(class_to_features.items()):
        path = save_sample_feature_boxplot(
            df=df,
            feature_cols=class_features,
            output_dir=output_dir,
            prefix=f"{prefix}_{sanitize_filename(feature_class)}",
        )

        if path is not None:
            saved_paths.append(path)

    return saved_paths


def process_single_feature_table(
    sample_id: str,
    config: dict,
    feature_type: str,
):
    if feature_type not in ("raw", "processed"):
        raise ValueError(
            f"feature_type must be 'raw' or 'processed', got: {feature_type}"
        )

    csv_path = get_feature_csv_path(
        config["output_dir"],
        sample_id,
        feature_type,
    )
    output_dir = get_statistics_dir(
        config["output_dir"],
        sample_id,
        feature_type,
    )

    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found: {csv_path}")
        return None

    df = load_feature_csv(
        csv_path=csv_path,
        status_filter=config["status_filter"],
    )

    feature_cols = get_feature_columns(
        df=df,
        drop_diagnostic=config["drop_diagnostic"],
    )

    if len(feature_cols) == 0:
        print(f"[WARNING] No feature columns found: {sample_id}, {feature_type}")
        return None

    stats_df = compute_feature_statistics(df, feature_cols)
    csv_out, parquet_out = save_statistics(stats_df, output_dir)

    df_for_vis = df.copy()

    if "sample_id" not in df_for_vis.columns:
        df_for_vis["sample_id"] = sample_id

    if config.get("save_representatives", False):
        save_representative_patches(
            df=df_for_vis,
            feature_cols=feature_cols,
            output_dir=output_dir,
            config=config,
            prefix=f"{sample_id}_{feature_type}",
        )

    if config.get("save_boxplot", False):
        save_sample_feature_boxplot(
            df=df_for_vis,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix=f"{sample_id}_{feature_type}",
        )

        save_sample_feature_boxplots_by_class(
            df=df_for_vis,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix=f"{sample_id}_{feature_type}",
        )

    return {
        "sample_id": sample_id,
        "feature_type": feature_type,
        "df": df,
        "feature_cols": feature_cols,
        "stats_df": stats_df,
        "output_dir": output_dir,
        "csv_out": csv_out,
        "parquet_out": parquet_out,
    }


def process_single_sample(
    sample_id: str,
    config: dict,
):
    results = {}

    for feature_type in ("raw", "processed"):
        results[feature_type] = process_single_feature_table(
            sample_id=sample_id,
            config=config,
            feature_type=feature_type,
        )

    return {
        "sample_id": sample_id,
        "raw": results.get("raw"),
        "processed": results.get("processed"),
    }


def process_merged_samples(
    results,
    config: dict,
):
    if not config.get("save_merged", False):
        return

    for feature_type in ("raw", "processed"):
        collected = []

        for result in results:
            if result is None:
                continue

            part = result.get(feature_type)

            if part is None:
                continue

            temp = part["df"].copy()
            temp["sample_id"] = result["sample_id"]
            collected.append(temp)

        if not collected:
            print(f"[WARNING] No valid merged data for {feature_type}")
            continue

        merged_df = pd.concat(
            collected,
            axis=0,
            ignore_index=True,
        )

        feature_cols = get_feature_columns(
            df=merged_df,
            drop_diagnostic=config["drop_diagnostic"],
        )

        merged_stats_df = compute_feature_statistics(
            merged_df,
            feature_cols,
        )

        output_dir = os.path.join(
            config["output_dir"],
            "_merged",
            "statistics",
            feature_type,
        )

        save_statistics(
            merged_stats_df,
            output_dir,
        )

        if config.get("save_representatives", False):
            save_representative_patches(
                df=merged_df,
                feature_cols=feature_cols,
                output_dir=output_dir,
                config=config,
                prefix=f"merged_{feature_type}",
            )

        if config.get("save_boxplot", False):
            save_sample_feature_boxplot(
                df=merged_df,
                feature_cols=feature_cols,
                output_dir=output_dir,
                prefix=f"merged_{feature_type}",
            )

            save_sample_feature_boxplots_by_class(
                df=merged_df,
                feature_cols=feature_cols,
                output_dir=output_dir,
                prefix=f"merged_{feature_type}",
            )
