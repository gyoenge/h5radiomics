from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
from tqdm import tqdm

from hestradiomics.extractors import (
    process_single_patch,
)
from hestradiomics.extractors.constants import *
from hestradiomics.extractors.builders import (
    _get_worker_radiomics_extractor,
    _get_worker_shape2d_extractor,
    build_radiomics_extractor,
    build_shape2d_extractor,
)
from hestradiomics.utils import (
    get_barcodes_key,
    get_coords_key,
    get_img_key,
    load_cellseg_dataframe,
    make_error_row,
)

logging.getLogger("radiomics").setLevel(logging.ERROR)


# ------------------------------------------------------------------------------
# chunk / pipeline
# ------------------------------------------------------------------------------

def process_patch_chunk(
    h5_path,
    patch_indices,
    output_dir: str,
    sample_id: str,
    classes,
    filters,
    label,
    save_patches,
    image_type_settings=None,
    mask_source: str = "threshold",
    cellseg_path: Optional[str] = None,
):
    rows = []

    extractor = _get_worker_radiomics_extractor(
        classes=classes,
        filters=filters,
        label=label,
        image_type_settings=image_type_settings,
    )
    shape_extractor = _get_worker_shape2d_extractor(label)

    cellseg_df = load_cellseg_dataframe(cellseg_path) if mask_source == "cellseg" else None

    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        for i in patch_indices:
            try:
                row = process_single_patch(
                    f=f,
                    img_key=img_key,
                    coords_key=coords_key,
                    barcodes_key=barcodes_key,
                    i=i,
                    output_dir=output_dir,
                    sample_id=sample_id,
                    extractor=extractor,
                    label=label,
                    save_patches=save_patches,
                    mask_source=mask_source,
                    cellseg_df=cellseg_df,
                    shape_extractor=shape_extractor,
                )
                rows.append(row)
            except Exception as e:
                rows.append(make_error_row(i, str(e)))

    return rows


def split_indices(indices, num_chunks):
    chunk_size = math.ceil(len(indices) / num_chunks)
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def extract_radiomics(
    h5_path,
    output_dir: str,
    sample_id: str,
    extractor=None,
    label=EXTRACTOR_DEFAULT_LABEL,
    save_patches=True,
    num_workers=0,
    classes=None,
    filters=None,
    image_type_settings=None,
    mask_source: str = "threshold",
    cellseg_path: Optional[str] = None,
    celltype_mode: str = "merged",
    target_cell_type: Optional[str] = None,
):
    """
    New design:
      - threshold: one row per patch, patch radiomics only
      - cellseg: one row per patch, patch + cellseg + morphology + distribution
    """
    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        total_num_patches = len(f[img_key])

    patch_indices = list(range(total_num_patches))

    if mask_source == "cellseg" and not cellseg_path:
        raise ValueError("cellseg_path is required when mask_source='cellseg'")

    if num_workers is None or num_workers <= 1:
        rows = []

        if extractor is None:
            extractor = build_radiomics_extractor(
                classes=classes,
                filters=filters,
                label=label,
                image_type_settings=image_type_settings,
            )

        shape_extractor = build_shape2d_extractor(label=label)
        cellseg_df = load_cellseg_dataframe(cellseg_path) if mask_source == "cellseg" else None

        with h5py.File(h5_path, "r") as f:
            img_key = get_img_key(f)
            coords_key = get_coords_key(f)
            barcodes_key = get_barcodes_key(f)

            for i in tqdm(patch_indices, desc=f"[Processing patches] {sample_id}"):
                try:
                    row = process_single_patch(
                        f=f,
                        img_key=img_key,
                        coords_key=coords_key,
                        barcodes_key=barcodes_key,
                        i=i,
                        output_dir=output_dir,
                        sample_id=sample_id,
                        extractor=extractor,
                        label=label,
                        save_patches=save_patches,
                        mask_source=mask_source,
                        cellseg_df=cellseg_df,
                        shape_extractor=shape_extractor,
                    )
                    rows.append(row)
                except Exception as e:
                    rows.append(make_error_row(i, str(e)))

        return {
            "total_num_patches": total_num_patches,
            "rows": rows,
        }

    num_workers = min(num_workers, os.cpu_count() or 1)
    chunks = split_indices(patch_indices, num_workers * 64)
    rows = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk_size = {}

        for chunk in chunks:
            future = executor.submit(
                process_patch_chunk,
                h5_path,
                chunk,
                output_dir,
                sample_id,
                classes,
                filters,
                label,
                save_patches,
                image_type_settings,
                mask_source,
                cellseg_path,
            )
            future_to_chunk_size[future] = len(chunk)

        with tqdm(total=len(chunks), desc=f"[Processing chunks] {sample_id}", position=0) as chunk_pbar, tqdm(
            total=total_num_patches, desc=f"[Processing patches] {sample_id}", position=1
        ) as patch_pbar:
            for future in as_completed(future_to_chunk_size):
                chunk_rows = future.result()
                rows.extend(chunk_rows)

                chunk_pbar.update(1)
                patch_pbar.update(future_to_chunk_size[future])

    rows.sort(key=lambda x: x["patch_idx"])

    return {
        "total_num_patches": total_num_patches,
        "rows": rows,
    }


# ------------------------------------------------------------------------------
# save
# ------------------------------------------------------------------------------

def save_radiomics_result_as_parquet(result: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(result["rows"])
    df.to_parquet(save_path, index=False)

    print(f"[SAVE] {save_path} | rows={len(df)}")


# ------------------------------------------------------------------------------
# config-based wrapper
# ------------------------------------------------------------------------------

def run_radiomics_extraction_for_oncotree(
    oncotree: str,
    data_root: str | Path,
    patch_dirname: str = "patches",
    cellseg_dirname: str = "cellseg",
    output_dirname: str = "radiomics_features",
    mask_source: str = "cellseg",
    save_patches: bool = False,
    num_workers: int = 0,
    overwrite: bool = False,
    classes=None,
    filters=None,
    image_type_settings=None,
    label=EXTRACTOR_DEFAULT_LABEL,
) -> None:
    """
    Expected structure:

    data_root/
      IDC/
        patches/
          sample1.h5
        cellseg/
          sample1.parquet
        radiomics_features/
          sample1.parquet
    """
    data_root = Path(data_root)
    oncotree_root = data_root / oncotree

    patch_dir = oncotree_root / patch_dirname
    cellseg_dir = oncotree_root / cellseg_dirname
    output_dir = oncotree_root / output_dirname

    output_dir.mkdir(parents=True, exist_ok=True)

    if not patch_dir.exists():
        print(f"[SKIP] Patch dir not found: {patch_dir}")
        return

    h5_paths = sorted(patch_dir.glob("*.h5"))

    if len(h5_paths) == 0:
        print(f"[SKIP] No h5 files found: {patch_dir}")
        return

    print("=" * 80)
    print(f"[ONCOTREE] {oncotree}")
    print(f"[PATCH DIR] {patch_dir}")
    print(f"[OUTPUT DIR] {output_dir}")
    print(f"[NUM SAMPLES] {len(h5_paths)}")
    print("=" * 80)

    for h5_path in h5_paths:
        sample_id = h5_path.stem
        save_path = output_dir / f"{sample_id}.parquet"

        if save_path.exists() and not overwrite:
            print(f"[SKIP] Already exists: {save_path}")
            continue

        cellseg_path = None
        if mask_source == "cellseg":
            cellseg_path = cellseg_dir / f"{sample_id}.parquet"

            if not cellseg_path.exists():
                print(f"[SKIP] Cellseg not found: {cellseg_path}")
                continue

        print(f"[RUN] {oncotree}/{sample_id}")

        result = extract_radiomics(
            h5_path=str(h5_path),
            output_dir=str(output_dir),
            sample_id=sample_id,
            label=label,
            save_patches=save_patches,
            num_workers=num_workers,
            classes=classes,
            filters=filters,
            image_type_settings=image_type_settings,
            mask_source=mask_source,
            cellseg_path=str(cellseg_path) if cellseg_path is not None else None,
        )

        save_radiomics_result_as_parquet(result, save_path)


def run_radiomics_extraction_from_config(config) -> None:
    """
    Required:
      config.download.download_dir
      config.download.oncotrees
      config.radiomics.*
    """
    for oncotree in config.download.oncotrees:
        run_radiomics_extraction_for_oncotree(
            oncotree=oncotree,
            data_root=config.download.download_dir,
            patch_dirname=config.radiomics.patch_dirname,
            cellseg_dirname=config.radiomics.segment_dirname,
            output_dirname=config.radiomics.output_dirname,
            mask_source=config.radiomics.mask_source,
            save_patches=config.radiomics.save_patches,
            num_workers=config.radiomics.num_workers,
            overwrite=config.radiomics.overwrite,
        )


if __name__ == "__main__": 
    from hestradiomics.config import CONFIG
    run_radiomics_extraction_from_config(config=CONFIG)

