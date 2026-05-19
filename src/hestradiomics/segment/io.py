from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from hestradiomics.utils import ensure_uint8_rgb


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape:
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class H5PatchDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        patch_indices: Optional[List[int]] = None,
        img_key_candidates: Tuple[str, ...] = ("img", "imgs", "images"),
        barcode_key_candidates: Tuple[str, ...] = ("barcode", "barcodes"),
    ):
        self.h5_path = h5_path
        self.img_key_candidates = img_key_candidates
        self.barcode_key_candidates = barcode_key_candidates

        with h5py.File(self.h5_path, "r") as f:
            self.img_key = self._find_key(f, self.img_key_candidates)
            self.barcode_key = self._find_key(f, self.barcode_key_candidates)
            self.has_coords = "coords" in f

            n = f[self.img_key].shape[0]
            self.patch_indices = list(range(n)) if patch_indices is None else patch_indices

    @staticmethod
    def _find_key(
        f: h5py.File,
        candidates: Iterable[str],
    ) -> str:
        for key in candidates:
            if key in f:
                return key

        raise KeyError(f"None of keys found: {candidates}")

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(
        self,
        i: int,
    ) -> Dict[str, Any]:
        patch_idx = self.patch_indices[i]

        with h5py.File(self.h5_path, "r") as f:
            image = f[self.img_key][patch_idx]
            barcode_raw = f[self.barcode_key][patch_idx]
            coord = f["coords"][patch_idx] if self.has_coords else None

        return {
            "patch_idx": int(patch_idx),
            "barcode": _decode_scalar(barcode_raw),
            "coord": None if coord is None else np.asarray(coord),
            "image": ensure_uint8_rgb(image),
        }


def collate_patches(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "patch_idx": [item["patch_idx"] for item in batch],
        "barcodes": [item["barcode"] for item in batch],
        "coords": [item["coord"] for item in batch],
    }


def gdf_to_cell_rows(
    gdf: gpd.GeoDataFrame,
    patch_idx: int,
    barcode: str,
) -> List[Dict[str, Any]]:
    rows = []

    if len(gdf) == 0:
        return rows

    gdf = gdf.copy()

    if "cell_id_in_patch" not in gdf.columns:
        gdf["cell_id_in_patch"] = list(range(1, len(gdf) + 1))

    for _, row in gdf.iterrows():
        rows.append(
            {
                "patch_idx": int(patch_idx),
                "barcode": barcode,
                "cell_id_in_patch": int(row["cell_id_in_patch"]),
                "class_id": (
                    int(row["class_id"])
                    if "class_id" in gdf.columns and pd.notna(row["class_id"])
                    else -1
                ),
                "class_name": (
                    str(row["class_name"])
                    if "class_name" in gdf.columns and pd.notna(row["class_name"])
                    else "unknown"
                ),
                "geometry": row.geometry,
            }
        )

    return rows


def save_cellseg_parquet(
    rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    save_path: str,
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if rows:
        cell_gdf = gpd.GeoDataFrame(
            rows,
            geometry="geometry",
            crs=None,
        )
    else:
        cell_gdf = gpd.GeoDataFrame(
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

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        # coord_raw는 list라 parquet 저장은 가능하지만,
        # 안정성을 위해 JSON 문자열 컬럼도 같이 둔다.
        if "coord_raw" in summary_df.columns:
            summary_df["coord_raw_json"] = summary_df["coord_raw"].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if x is not None
                else None
            )

        cell_gdf = cell_gdf.merge(
            summary_df,
            on=["patch_idx", "barcode"],
            how="left",
        )
    else:
        cell_gdf["coord_raw"] = None
        cell_gdf["coord_raw_json"] = None
        cell_gdf["n_cells"] = 0

    cell_gdf.to_parquet(
        save_path,
        index=False,
    )

    return save_path


def load_cellseg_parquet(
    parquet_path: str,
) -> gpd.GeoDataFrame:
    return gpd.read_parquet(parquet_path)


def list_sample_ids_from_patches(
    oncotree_root: str,
) -> List[str]:
    patches_dir = os.path.join(
        oncotree_root,
        "patches",
    )

    if not os.path.isdir(patches_dir):
        print(f"[WARN] patches dir not found: {patches_dir}")
        return []

    return [
        os.path.splitext(filename)[0]
        for filename in sorted(os.listdir(patches_dir))
        if filename.endswith(".h5")
    ]


def build_sample_paths(
    hest_root: str,
    oncotree: str,
    sample_id: str,
) -> Dict[str, str]:
    data_root = os.path.join(
        hest_root,
        oncotree,
    )

    segment_dir = os.path.join(
        data_root,
        "segment",
    )

    segment_vis_dir = os.path.join(
        data_root,
        "segment_vis",
    )

    return {
        "data_root": data_root,
        "patch_h5_path": os.path.join(data_root, "patches", f"{sample_id}.h5"),
        "segment_dir": segment_dir,
        "segment_vis_dir": segment_vis_dir,
        "seg_parquet_path": os.path.join(segment_dir, f"{sample_id}.parquet"),
        "summary_json_path": os.path.join(segment_dir, f"{sample_id}.summary.json"),
        "runtime_dir": os.path.join(segment_dir, "_cellvit_runtime"),
        "overlay_dir": os.path.join(segment_vis_dir, sample_id),
    }
