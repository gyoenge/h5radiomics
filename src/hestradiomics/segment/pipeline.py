from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hestradiomics.segment.adapter import CellViTInferenceAdapter
from hestradiomics.segment.io import (
    H5PatchDataset,
    collate_patches,
    save_cellseg_h5,
    gdf_to_cell_rows, 
    list_sample_ids_from_patches, 
    build_sample_paths
)
from hestradiomics.utils import filter_sample_ids


def segment_h5_patches_with_cellvit(
    h5_path: str,
    seg_h5_path: str,
    model_path: str,
    runtime_dir: str,
    summary_json_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    patch_indices: Optional[List[int]] = None,
    device: str = "cuda:0",
    predictor: Optional[CellViTInferenceAdapter] = None,
) -> str:
    os.makedirs(os.path.dirname(seg_h5_path), exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)

    dataset = H5PatchDataset(
        h5_path=h5_path,
        patch_indices=patch_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patches,
        pin_memory=torch.cuda.is_available(),
    )

    if predictor is None:
        predictor = CellViTInferenceAdapter(
            model_path=model_path,
            model_name=os.path.basename(model_path),
            output_dir=os.path.dirname(seg_h5_path),
            runtime_dir=runtime_dir,
            device=device,
        )

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    pbar = tqdm(
        total=len(dataset),
        desc="Segmenting patches",
    )

    for batch in loader:
        images = batch["images"]
        patch_idxs = batch["patch_idx"]
        barcodes = batch["barcodes"]
        coords = batch["coords"]

        gdfs = predictor.predict_batch_to_gdfs(images)

        for gdf, patch_idx, barcode, coord in zip(
            gdfs,
            patch_idxs,
            barcodes,
            coords,
        ):
            rows = gdf_to_cell_rows(
                gdf=gdf,
                patch_idx=patch_idx,
                barcode=barcode,
            )

            all_rows.extend(rows)

            summary_rows.append(
                {
                    "patch_idx": int(patch_idx),
                    "barcode": barcode,
                    "coord_raw": (
                        None
                        if coord is None
                        else np.asarray(coord).tolist()
                    ),
                    "n_cells": int(len(gdf)),
                }
            )

            pbar.update(1)

    pbar.close()

    save_cellseg_h5(
        rows=all_rows,
        summary_rows=summary_rows,
        save_path=seg_h5_path,
    )

    summary = {
        "h5_path": h5_path,
        "segment_dir": os.path.dirname(seg_h5_path),
        "runtime_dir": runtime_dir,
        "num_patches": len(dataset),
        "num_polygons": int(len(all_rows)),
        "seg_h5_path": seg_h5_path,
    }

    if summary_json_path is not None:
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

    return seg_h5_path


def segment_one_sample(
    hest_root: str,
    oncotree: str,
    sample_id: str,
    model_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Optional[str]:
    paths = build_sample_paths(
        hest_root=hest_root,
        oncotree=oncotree,
        sample_id=sample_id,
    )

    patch_h5_path = paths["patch_h5_path"]
    seg_h5_path = paths["seg_h5_path"]

    if not os.path.exists(patch_h5_path):
        print(f"[WARN] patch h5 not found: {patch_h5_path}")
        return None

    if os.path.exists(seg_h5_path) and not overwrite:
        print(f"[SKIP] segment exists: {seg_h5_path}")
        return seg_h5_path

    return segment_h5_patches_with_cellvit(
        h5_path=patch_h5_path,
        seg_h5_path=seg_h5_path,
        model_path=model_path,
        runtime_dir=paths["runtime_dir"],
        summary_json_path=paths["summary_json_path"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def segment_all_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    model_path: str,
    sample_ids: Optional[Tuple[str, ...]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> List[str]:
    output_paths = []

    for oncotree in oncotrees:
        oncotree_root = os.path.join(
            hest_root,
            oncotree,
        )

        all_sample_ids = list_sample_ids_from_patches(
            oncotree_root
        )

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        for sample_id in target_sample_ids:
            seg_h5_path = segment_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                model_path=model_path,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                overwrite=overwrite,
            )

            if seg_h5_path is not None:
                output_paths.append(seg_h5_path)

    return output_paths


