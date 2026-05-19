from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm

from hestradiomics.segment.adapter import iter_polygons
from hestradiomics.segment.io import (
    H5PatchDataset,
    build_sample_paths,
    list_sample_ids_from_patches,
    load_cellseg_parquet,
)
from hestradiomics.utils import ensure_uint8_rgb, filter_sample_ids


CLASS_COLOR_MAP = {
    "neoplastic": "#ff4d4d",
    "inflammatory": "#4da6ff",
    "connective": "#00c853",
    "dead": "#ffd54f",
    "epithelial": "#ab47bc",
    "background": "#9e9e9e",
    "unknown": "#00ffff",
}

DEFAULT_OVERLAY_COLOR = "#00ffff"


def _select_patch_indices(
    patch_indices_all: List[int],
    vis_ratio: float = 1.0,
    patch_indices: Optional[List[int]] = None,
) -> List[int]:
    if patch_indices is not None:
        return [int(i) for i in patch_indices]

    if not (0 < vis_ratio <= 1):
        raise ValueError(f"vis_ratio must be in (0, 1], got {vis_ratio}")

    patch_indices_all = [int(i) for i in patch_indices_all]
    total_num_patches = len(patch_indices_all)

    if total_num_patches == 0:
        return []

    num_vis = max(1, math.ceil(total_num_patches * vis_ratio))

    if num_vis >= total_num_patches:
        return patch_indices_all

    selected_positions = np.linspace(
        0,
        total_num_patches - 1,
        num=num_vis,
        dtype=int,
    )

    return [patch_indices_all[i] for i in selected_positions]


def _empty_cellseg_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "patch_idx": pd.Series(dtype="int64"),
            "barcode": pd.Series(dtype="str"),
            "cell_id_in_patch": pd.Series(dtype="int64"),
            "class_id": pd.Series(dtype="int64"),
            "class_name": pd.Series(dtype="str"),
            "geometry": [],
        },
        geometry="geometry",
        crs=None,
    )


def load_cellseg_for_overlay(
    seg_parquet_path: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    seg_gdf = load_cellseg_parquet(seg_parquet_path)

    if seg_gdf.empty:
        return _empty_cellseg_gdf(), pd.DataFrame(
            columns=["patch_idx", "barcode", "n_cells"]
        )

    if "patch_idx" not in seg_gdf.columns:
        raise KeyError("segment parquet must contain 'patch_idx' column.")

    if "barcode" not in seg_gdf.columns:
        seg_gdf["barcode"] = seg_gdf["patch_idx"].astype(str)

    if "class_name" not in seg_gdf.columns:
        seg_gdf["class_name"] = "unknown"

    if "class_id" not in seg_gdf.columns:
        seg_gdf["class_id"] = -1

    if "cell_id_in_patch" not in seg_gdf.columns:
        seg_gdf["cell_id_in_patch"] = seg_gdf.groupby("patch_idx").cumcount() + 1

    seg_gdf["patch_idx"] = seg_gdf["patch_idx"].astype(int)

    if "n_cells" in seg_gdf.columns:
        patch_df = (
            seg_gdf[["patch_idx", "barcode", "n_cells"]]
            .drop_duplicates(subset=["patch_idx"])
            .copy()
        )
    else:
        patch_df = (
            seg_gdf.groupby(["patch_idx", "barcode"], as_index=False)
            .size()
            .rename(columns={"size": "n_cells"})
        )

    patch_df["patch_idx"] = patch_df["patch_idx"].astype(int)

    return seg_gdf, patch_df


def save_overlay_png(
    img: np.ndarray,
    gdf: gpd.GeoDataFrame,
    save_path: str,
    title: Optional[str] = None,
    use_class_color: bool = True,
) -> None:
    img = ensure_uint8_rgb(img)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if use_class_color and len(gdf) > 0 and "class_name" in gdf.columns:
        for class_name, sub_gdf in gdf.groupby("class_name"):
            color = CLASS_COLOR_MAP.get(
                str(class_name).lower(),
                DEFAULT_OVERLAY_COLOR,
            )

            patches = []

            for geom in sub_gdf.geometry:
                for poly in iter_polygons(geom):
                    xy = np.asarray(poly.exterior.coords)

                    if len(xy) >= 3:
                        patches.append(MplPolygon(xy, closed=True))

            if patches:
                ax.add_collection(
                    PatchCollection(
                        patches,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.0,
                    )
                )

    else:
        patches = []

        for geom in gdf.geometry:
            for poly in iter_polygons(geom):
                xy = np.asarray(poly.exterior.coords)

                if len(xy) >= 3:
                    patches.append(MplPolygon(xy, closed=True))

        if patches:
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor="none",
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )

    if use_class_color:
        present_classes = (
            set(gdf["class_name"].astype(str).str.lower())
            if len(gdf) > 0 and "class_name" in gdf.columns
            else set()
        )

        handles = [
            Line2D([0], [0], color=color, lw=2, label=class_name)
            for class_name, color in CLASS_COLOR_MAP.items()
            if class_name in present_classes
        ]

        if handles:
            ax.legend(
                handles=handles,
                loc="lower left",
                fontsize=8,
            )

    if title:
        ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )

    plt.close(fig)


def save_overlays_from_cellseg_parquet(
    source_h5_path: str,
    seg_parquet_path: str,
    overlay_dir: str,
    use_class_color: bool = True,
    vis_ratio: float = 1.0,
    patch_indices: Optional[List[int]] = None,
    overwrite: bool = False,
) -> str:
    os.makedirs(overlay_dir, exist_ok=True)

    seg_gdf, patch_df = load_cellseg_for_overlay(seg_parquet_path)

    if patch_df.empty:
        print(f"[WARN] empty segment parquet: {seg_parquet_path}")
        return overlay_dir

    all_patch_indices = patch_df["patch_idx"].astype(int).tolist()

    selected_patch_indices = _select_patch_indices(
        patch_indices_all=all_patch_indices,
        vis_ratio=vis_ratio,
        patch_indices=patch_indices,
    )

    patch_df = patch_df[
        patch_df["patch_idx"].astype(int).isin(selected_patch_indices)
    ].copy()

    seg_gdf = seg_gdf[
        seg_gdf["patch_idx"].astype(int).isin(selected_patch_indices)
    ].copy()

    dataset = H5PatchDataset(
        h5_path=source_h5_path,
        patch_indices=selected_patch_indices,
    )

    gdf_by_patch = {
        int(patch_idx): sub_gdf.copy()
        for patch_idx, sub_gdf in seg_gdf.groupby("patch_idx")
    }

    for i in tqdm(range(len(dataset)), desc="Saving segment overlays"):
        sample = dataset[i]

        patch_idx = int(sample["patch_idx"])
        barcode = str(sample["barcode"])
        img = sample["image"]

        gdf = gdf_by_patch.get(
            patch_idx,
            _empty_cellseg_gdf(),
        )

        save_path = os.path.join(
            overlay_dir,
            f"{barcode}_idx{patch_idx}.png",
        )

        if os.path.exists(save_path) and not overwrite:
            continue

        save_overlay_png(
            img=img,
            gdf=gdf,
            save_path=save_path,
            title=f"idx={patch_idx}, bc={barcode}, cells={len(gdf)}",
            use_class_color=use_class_color,
        )

    return overlay_dir


def save_overlay_one_sample(
    hest_root: str,
    oncotree: str,
    sample_id: str,
    use_class_color: bool = True,
    vis_ratio: float = 1.0,
    overwrite: bool = False,
    patch_indices: Optional[List[int]] = None,
) -> Optional[str]:
    paths = build_sample_paths(
        hest_root=hest_root,
        oncotree=oncotree,
        sample_id=sample_id,
    )

    patch_h5_path = paths["patch_h5_path"]
    seg_parquet_path = paths["seg_parquet_path"]
    overlay_dir = paths["overlay_dir"]

    if not os.path.exists(patch_h5_path):
        print(f"[WARN] patch h5 not found: {patch_h5_path}")
        return None

    if not os.path.exists(seg_parquet_path):
        print(f"[WARN] segment parquet not found: {seg_parquet_path}")
        return None

    if (
        os.path.isdir(overlay_dir)
        and len(os.listdir(overlay_dir)) > 0
        and not overwrite
        and patch_indices is None
        and vis_ratio >= 1.0
    ):
        print(f"[SKIP] overlay exists: {overlay_dir}")
        return overlay_dir

    return save_overlays_from_cellseg_parquet(
        source_h5_path=patch_h5_path,
        seg_parquet_path=seg_parquet_path,
        overlay_dir=overlay_dir,
        use_class_color=use_class_color,
        vis_ratio=vis_ratio,
        patch_indices=patch_indices,
        overwrite=overwrite,
    )


def segment_visualization_from_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    sample_ids: Optional[Tuple[str, ...]] = None,
    use_class_color: bool = True,
    vis_ratio: float = 1.0,
    overwrite: bool = False,
    patch_indices: Optional[List[int]] = None,
) -> List[str]:
    output_dirs: List[str] = []

    for oncotree in oncotrees:
        oncotree_root = os.path.join(hest_root, oncotree)

        if not os.path.isdir(oncotree_root):
            print(f"[SKIP] oncotree root not found: {oncotree_root}")
            continue

        all_sample_ids = list_sample_ids_from_patches(oncotree_root)

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        if not target_sample_ids:
            print(f"[SKIP] No selected samples found for {oncotree}")
            continue

        print("=" * 80)
        print(f"[ONCOTREE] {oncotree}")
        print(f"[ROOT] {oncotree_root}")
        print(f"[NUM SAMPLES] {len(target_sample_ids)} / {len(all_sample_ids)}")
        print(f"[VIS RATIO] {vis_ratio}")
        print(f"[USE CLASS COLOR] {use_class_color}")
        print(f"[OVERWRITE] {overwrite}")
        print("=" * 80)

        for sample_id in target_sample_ids:
            print(f"\n[SAMPLE] {sample_id}")

            overlay_dir = save_overlay_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                use_class_color=use_class_color,
                vis_ratio=vis_ratio,
                overwrite=overwrite,
                patch_indices=patch_indices,
            )

            if overlay_dir is not None:
                output_dirs.append(overlay_dir)

    return output_dirs