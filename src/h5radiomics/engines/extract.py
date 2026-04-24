from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import h5py
from tqdm import tqdm

from h5radiomics.engines.extractors.constants import *
from h5radiomics.utils import (
    get_barcodes_key, get_coords_key, get_img_key,
    load_cellseg_dataframe, 
    make_error_row, 
)
from h5radiomics.engines.extractors.builders import (
    _get_worker_radiomics_extractor, 
    _get_worker_shape2d_extractor, 
    build_radiomics_extractor, 
    build_shape2d_extractor, 
)
from h5radiomics.engines.extractors import (
    process_single_patch, 
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
    celltype_mode: str = "merged",   # kept for backward compatibility, unused in new design
    target_cell_type: Optional[str] = None,  # kept for backward compatibility, unused
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

            for i in tqdm(patch_indices, desc="[Processing patches]"):
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

        with tqdm(total=len(chunks), desc="[Processing chunks]", position=0) as chunk_pbar, tqdm(
            total=total_num_patches, desc="[Processing patches]", position=1
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

