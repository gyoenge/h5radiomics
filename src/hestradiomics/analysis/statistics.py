from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hestradiomics.utils import filter_sample_ids


# =============================================================================
# Config
# =============================================================================

HEST_ROOT = Path("/root/workspace/hest-radiomics/data/hestradiomics")
ONCOTREE = "IDC"
SAMPLE_IDS = ["TENX95"]

RADIOMICS_DIR = HEST_ROOT / ONCOTREE / "radiomics"
PATCH_DIR = HEST_ROOT / ONCOTREE / "patches"
OUTPUT_DIR = HEST_ROOT / ONCOTREE / "radiomics_statistics"

REPRESENTATIVE_STATS = ["min", "q25", "q50", "q75", "max"]

META_COLS = {
    "sample_id",
    "patch_idx",
    "barcode",
    "mask_path",
    "color_path",
    "gray_path",
    "x",
    "y",
    "coord_x",
    "coord_y",
    "status",
}

FEATURE_CLASS_PREFIXES = {
    "cell_count": ["n_cells_total", "cellseg_mask_area"],
    "cellseg_firstorder": ["cellseg_all_original_firstorder_"],
    "cellseg_glcm": ["cellseg_all_original_glcm_"],
    "cellseg_glrlm": ["cellseg_all_original_glrlm_"],
    "cellseg_glszm": ["cellseg_all_original_glszm_"],
    "cellseg_gldm": ["cellseg_all_original_gldm_"],
    "cellseg_ngtdm": ["cellseg_all_original_ngtdm_"],
    "morph": ["morph_"],
    "dist": ["dist_"],
}


# =============================================================================
# Utils
# =============================================================================

def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180]


def load_radiomics_parquet(parquet_path: Path, status_filter: Optional[str] = "ok") -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    if status_filter is not None and "status" in df.columns:
        df = df[df["status"] == status_filter].copy()

    return df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c not in META_COLS and not c.startswith("diagnostics_")
    ]

    return feature_cols


def assign_feature_class(feature_name: str) -> str:
    for class_name, prefixes in FEATURE_CLASS_PREFIXES.items():
        if any(feature_name.startswith(prefix) for prefix in prefixes):
            return class_name
    return "others"


def group_features_by_class(feature_cols: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}

    for col in feature_cols:
        cls = assign_feature_class(col)
        grouped.setdefault(cls, []).append(col)

    return grouped


# =============================================================================
# Statistics
# =============================================================================

def compute_feature_statistics(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        rows.append({
            "feature_class": assign_feature_class(col),
            "feature_name": col,
            "count": int(s.count()),
            "nan_count": int(s.isna().sum()),
            "nan_ratio": float(s.isna().mean()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q01": s.quantile(0.01),
            "q05": s.quantile(0.05),
            "q10": s.quantile(0.10),
            "q25": s.quantile(0.25),
            "median": s.median(),
            "q75": s.quantile(0.75),
            "q90": s.quantile(0.90),
            "q95": s.quantile(0.95),
            "q99": s.quantile(0.99),
            "max": s.max(),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
            "zero_count": int((s == 0).sum()),
            "positive_count": int((s > 0).sum()),
            "negative_count": int((s < 0).sum()),
        })

    return pd.DataFrame(rows).sort_values(["feature_class", "feature_name"])


def save_feature_statistics(stats_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    stats_df.to_csv(output_dir / "feature_statistics.csv", index=False)
    stats_df.to_parquet(output_dir / "feature_statistics.parquet", index=False)


# =============================================================================
# Boxplot by Feature Class
# =============================================================================

def save_boxplots_by_class(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    sample_id: str,
) -> None:
    boxplot_dir = output_dir / "boxplots_by_class"
    ensure_dir(boxplot_dir)

    grouped = group_features_by_class(feature_cols)

    for class_name, cols in sorted(grouped.items()):
        data = []

        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            data.append(s.values if len(s) > 0 else np.array([np.nan]))

        fig_width = max(14, len(cols) * 0.28)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        ax.boxplot(data, showfliers=False, widths=0.5)
        ax.set_title(f"{sample_id} | {class_name} Feature Distribution")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.set_xticks(range(1, len(cols) + 1))
        ax.set_xticklabels(cols, rotation=90, fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(boxplot_dir / f"{sample_id}__{class_name}__boxplot.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Patch Image Loader
# =============================================================================

def get_h5_key(h5: h5py.File, candidates: Tuple[str, ...]) -> str:
    for key in candidates:
        if key in h5:
            return key
    raise KeyError(f"Cannot find any key from candidates={candidates}. Available keys={list(h5.keys())}")


def load_patch_image_from_h5(patch_h5_path: Path, patch_idx: int) -> np.ndarray:
    with h5py.File(patch_h5_path, "r") as h5:
        img_key = get_h5_key(h5, ("img", "imgs", "images"))
        img = h5[img_key][int(patch_idx)]

    return img


# =============================================================================
# Representative Selection
# =============================================================================

def get_target_values(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()

    return {
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def select_representative_row(
    df: pd.DataFrame,
    feature_col: str,
    stat_name: str,
    target_value: float,
) -> Optional[pd.Series]:
    temp = df.copy()
    temp[feature_col] = pd.to_numeric(temp[feature_col], errors="coerce")
    temp = temp[temp[feature_col].notna()].copy()

    if len(temp) == 0:
        return None

    if stat_name == "min":
        return temp.loc[temp[feature_col].idxmin()]

    if stat_name == "max":
        return temp.loc[temp[feature_col].idxmax()]

    temp["_abs_diff"] = (temp[feature_col] - target_value).abs()
    return temp.loc[temp["_abs_diff"].idxmin()]


# =============================================================================
# Representative Visualization
# =============================================================================

def plot_representative_summary(
    df: pd.DataFrame,
    feature_col: str,
    selected_row: pd.Series,
    selected_value: float,
    stat_name: str,
    patch_img: np.ndarray,
    save_path: Path,
    sample_id: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1) patch image
    axes[0].imshow(patch_img)
    axes[0].set_title(
        f"Representative Patch\n"
        f"{stat_name} | patch_idx={selected_row.get('patch_idx')}"
    )
    axes[0].axis("off")

    # 2) whole slide coordinate position
    if "x" in df.columns and "y" in df.columns:
        axes[1].scatter(df["x"], df["y"], s=4, alpha=0.25)
        axes[1].scatter(
            selected_row["x"],
            selected_row["y"],
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
        )
        axes[1].invert_yaxis()
        axes[1].set_title("Position in Whole Slide")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
    else:
        axes[1].text(0.5, 0.5, "No x/y columns", ha="center", va="center")
        axes[1].set_axis_off()

    # 3) value position in boxplot
    s = pd.to_numeric(df[feature_col], errors="coerce").dropna()

    axes[2].boxplot(s.values, vert=False, showfliers=False)
    axes[2].scatter([selected_value], [1], s=80, marker="*", edgecolors="black", linewidths=0.8)
    axes[2].set_title("Value Position in Distribution")
    axes[2].set_xlabel(feature_col)
    axes[2].set_yticks([])

    fig.suptitle(
        f"{sample_id} | {feature_col}\n"
        f"{stat_name}: selected={selected_value:.6g}",
        fontsize=11,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_representative_visualizations(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    sample_id: str,
    patch_h5_path: Path,
    representative_stats: List[str],
) -> None:
    rep_dir = output_dir / "representatives"
    ensure_dir(rep_dir)

    manifest_rows = []

    for feature_col in feature_cols:
        s = pd.to_numeric(df[feature_col], errors="coerce").dropna()

        if len(s) == 0:
            continue

        targets = get_target_values(df[feature_col])
        feature_class = assign_feature_class(feature_col)

        feature_dir = rep_dir / feature_class / sanitize_filename(feature_col)
        ensure_dir(feature_dir)

        for stat_name in representative_stats:
            if stat_name not in targets:
                continue

            target_value = targets[stat_name]

            row = select_representative_row(
                df=df,
                feature_col=feature_col,
                stat_name=stat_name,
                target_value=target_value,
            )

            if row is None:
                continue

            patch_idx = int(row["patch_idx"])
            selected_value = float(row[feature_col])

            try:
                patch_img = load_patch_image_from_h5(patch_h5_path, patch_idx)
                status = "ok"
            except Exception as e:
                patch_img = np.zeros((224, 224, 3), dtype=np.uint8)
                status = f"patch_load_failed: {e}"

            save_path = feature_dir / f"{stat_name}__patch_{patch_idx}.png"

            if status == "ok":
                plot_representative_summary(
                    df=df,
                    feature_col=feature_col,
                    selected_row=row,
                    selected_value=selected_value,
                    stat_name=stat_name,
                    patch_img=patch_img,
                    save_path=save_path,
                    sample_id=sample_id,
                )

            manifest_rows.append({
                "sample_id": sample_id,
                "feature_class": feature_class,
                "feature_name": feature_col,
                "stat_name": stat_name,
                "target_value": target_value,
                "selected_value": selected_value,
                "abs_diff": abs(selected_value - target_value),
                "patch_idx": patch_idx,
                "barcode": row.get("barcode", ""),
                "x": row.get("x", np.nan),
                "y": row.get("y", np.nan),
                "figure_path": str(save_path) if status == "ok" else "",
                "status": status,
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(rep_dir / "representative_manifest.csv", index=False)


# =============================================================================
# Main Pipeline
# =============================================================================


def process_single_sample_statistics(
    sample_id: str,
    parquet_path: str | Path,
    patch_h5_path: Optional[str | Path],
    output_dir: str | Path,
    status_filter: Optional[str] = "ok",
    save_boxplot: bool = True,
    save_representative: bool = True,
    representative_stats: Optional[List[str]] = None,
) -> None:
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)

    if representative_stats is None:
        representative_stats = ["min", "q25", "q50", "q75", "max"]

    ensure_dir(output_dir)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")

    print(f"[PARQUET] {parquet_path}")
    print(f"[OUTPUT] {output_dir}")

    df = load_radiomics_parquet(
        parquet_path=parquet_path,
        status_filter=status_filter,
    )

    feature_cols = get_feature_columns(df)

    print(f"[ROWS] {len(df)}")
    print(f"[FEATURE COLS] {len(feature_cols)}")

    stats_df = compute_feature_statistics(df, feature_cols)
    save_feature_statistics(stats_df, output_dir)

    if save_boxplot:
        save_boxplots_by_class(
            df=df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            sample_id=sample_id,
        )

    if save_representative and patch_h5_path is not None:
        save_representative_visualizations(
            df=df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            sample_id=sample_id,
            patch_h5_path=Path(patch_h5_path),
            representative_stats=representative_stats,
        )

    print("[DONE]")


def statistics_analysis_from_oncotrees(
    oncotrees: List[str],
    download_dir: str | Path,
    sample_ids: Optional[tuple[str, ...]] = None,
    radiomics_dirname: str = "radiomics",
    patch_dirname: str = "patches",
    statistics_dirname: str = "radiomics_statistics",
    status_filter: Optional[str] = "ok",
    save_boxplot: bool = True,
    save_representative: bool = True,
    representative_stats: Optional[List[str]] = None,
) -> None:
    download_dir = Path(download_dir)

    if representative_stats is None:
        representative_stats = ["min", "q25", "q50", "q75", "max"]

    for oncotree in oncotrees:
        oncotree_root = download_dir / oncotree

        radiomics_dir = oncotree_root / radiomics_dirname
        patch_dir = oncotree_root / patch_dirname
        statistics_root = oncotree_root / statistics_dirname

        if not radiomics_dir.exists():
            print(f"[SKIP] Radiomics dir not found: {radiomics_dir}")
            continue

        parquet_paths = sorted(radiomics_dir.glob("*.parquet"))

        if not parquet_paths:
            print(f"[SKIP] No parquet files found: {radiomics_dir}")
            continue

        all_sample_ids = [path.stem for path in parquet_paths]

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        if not target_sample_ids:
            print(f"[SKIP] No selected samples found for {oncotree}")
            continue

        target_sample_id_set = set(target_sample_ids)

        parquet_paths = [
            path for path in parquet_paths
            if path.stem in target_sample_id_set
        ]

        print("=" * 80)
        print(f"[ONCOTREE] {oncotree}")
        print(f"[RADIOMICS DIR] {radiomics_dir}")
        print(f"[PATCH DIR] {patch_dir}")
        print(f"[OUTPUT DIR] {statistics_root}")
        print(f"[NUM SAMPLES] {len(parquet_paths)} / {len(all_sample_ids)}")
        print(f"[STATUS FILTER] {status_filter}")
        print(f"[SAVE BOXPLOT] {save_boxplot}")
        print(f"[SAVE REPRESENTATIVE] {save_representative}")
        print(f"[REPRESENTATIVE STATS] {representative_stats}")
        print("=" * 80)

        for parquet_path in parquet_paths:
            sample_id = parquet_path.stem
            patch_h5_path = patch_dir / f"{sample_id}.h5"
            sample_output_dir = statistics_root / sample_id

            print(f"\n[SAMPLE] {sample_id}")

            if not patch_h5_path.exists():
                print(f"[WARN] Patch h5 not found: {patch_h5_path}")
                print("[WARN] Representative patch visualization will be skipped.")

            process_single_sample_statistics(
                sample_id=sample_id,
                parquet_path=parquet_path,
                patch_h5_path=patch_h5_path if patch_h5_path.exists() else None,
                output_dir=sample_output_dir,
                status_filter=status_filter,
                save_boxplot=save_boxplot,
                save_representative=save_representative,
                representative_stats=representative_stats,
            )