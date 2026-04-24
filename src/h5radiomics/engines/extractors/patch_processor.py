from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import geopandas as gpd
import numpy as np
from h5radiomics.engines.extractors.constants import *
from h5radiomics.utils import (
    build_threshold_mask, 
    rasterize_geometries_to_mask, 
    normalize_class_name, 
    save_region_mask_images, safe_update_features, 
    load_patch_data, 
    build_patch_row_base, 
)
from h5radiomics.engines.extractors import (
    extract_patch_level_radiomics,
    extract_cellseg_level_radiomics,
    extract_morphology_aggregates, 
    extract_cell_type_distribution, 
)
from h5radiomics.engines.extractors.builders import (
    _get_worker_shape2d_extractor, 
)
from h5radiomics.engines.extractors.intensity_texture import (
    _add_prefix_to_keys, 
    _execute_radiomics_on_mask, 
)


@dataclass
class PatchData:
    patch_idx: int
    color_patch: np.ndarray
    gray_patch: np.ndarray
    coords: Optional[Any]
    barcode: Optional[str]
    base_filename: str


# ------------------------------------------------------------------------------
# patch processors
# ------------------------------------------------------------------------------

def get_patch_cellseg(
    cellseg_df: gpd.GeoDataFrame,
    patch_idx: int,
) -> gpd.GeoDataFrame:
    patch_cellseg = cellseg_df[cellseg_df[PATCH_IDX_COLUMN] == patch_idx].copy()
    patch_cellseg = patch_cellseg[patch_cellseg.geometry.notnull()].copy()

    if len(patch_cellseg) > 0:
        patch_cellseg[CELL_CLASS_COLUMN] = patch_cellseg[CELL_CLASS_COLUMN].map(normalize_class_name)

    return patch_cellseg


def process_threshold_patch(
    patch: PatchData,
    row: Dict[str, Any],
    extractor,
    output_dir: str,
    sample_id: str,
    label: int,
    save_patches: bool,
) -> Dict[str, Any]:
    patch_mask = build_threshold_mask(patch.gray_patch, label=label)
    row[PATCH_MASK_AREA_COLUMN] = int(np.count_nonzero(patch_mask > 0))

    if save_patches:
        row[MASK_PATH_COLUMN] = save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=patch_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}{THRESHOLD_MASK_SUFFIX}",
        )

    if row[PATCH_MASK_AREA_COLUMN] < PATCH_MASK_AREA_MIN_THRESHOLD:
        row[STATUS_COLUMN] = STATUS_SKIPPED_SMALL_MASK
        return row

    safe_update_features(
        row,
        lambda: _add_prefix_to_keys(
            _execute_radiomics_on_mask(patch.gray_patch, patch_mask, extractor),
            "patch_",
        ),
        ERROR_PATCH_RADIOMICS,
    )
    return row


def process_cellseg_patch(
    patch: PatchData,
    row: Dict[str, Any],
    extractor,
    shape_extractor,
    output_dir: str,
    sample_id: str,
    label: int,
    save_patches: bool,
    cellseg_df: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    patch_cellseg = get_patch_cellseg(cellseg_df, patch.patch_idx)
    row[N_CELLS_TOTAL_COLUMN] = int(len(patch_cellseg))

    if len(patch_cellseg) == 0:
        row[STATUS_COLUMN] = STATUS_SKIPPED_NO_CELLSEG
        row.update(extract_cell_type_distribution(patch_cellseg))
        return row

    merged_mask = rasterize_geometries_to_mask(
        patch_cellseg.geometry.tolist(),
        image_shape=patch.gray_patch.shape,
        label=label,
    )
    row[CELLSEG_MASK_AREA_COLUMN] = int(np.count_nonzero(merged_mask > 0))

    if save_patches:
        row[MASK_PATH_COLUMN] = save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=merged_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}{CELLSEG_ALL_MASK_SUFFIX}",
        )

    # if save_patches:
    #     for class_name, sub in patch_cellseg.groupby(CELL_CLASS_COLUMN):
    #         if len(sub) == 0:
    #             continue

    #         mask_cls = rasterize_geometries_to_mask(
    #             sub.geometry.tolist(),
    #             image_shape=patch.gray_patch.shape,
    #             label=label,
    #         )

    #         save_region_mask_images(
    #             color_patch=patch.color_patch,
    #             gray_patch=patch.gray_patch,
    #             mask_patch=mask_cls,
    #             output_dir=output_dir,
    #             sample_id=sample_id,
    #             mask_filename=f"{patch.base_filename}__cellseg_{class_name}",
    #         )

    if save_patches:
        threshold_mask = build_threshold_mask(patch.gray_patch, label=label)

        save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=threshold_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}{THRESHOLD_MASK_SUFFIX}",
        )

    safe_update_features(
        row,
        lambda: extract_patch_level_radiomics(patch.gray_patch, extractor, label=label),
        ERROR_PATCH_RADIOMICS,
    )

    safe_update_features(
        row,
        lambda: extract_cellseg_level_radiomics(
            patch.gray_patch, patch_cellseg, extractor, label=label
        ),
        ERROR_CELLSEG_RADIOMICS,
    )

    safe_update_features(
        row,
        lambda: extract_morphology_aggregates(
            patch.gray_patch,
            patch_cellseg,
            label=label,
            shape_extractor=shape_extractor,
        ),
        ERROR_MORPHOLOGY,
    )

    safe_update_features(
        row,
        lambda: extract_cell_type_distribution(patch_cellseg),
        ERROR_DISTRIBUTION,
    )

    return row


def process_single_patch(
    f,
    img_key,
    coords_key,
    barcodes_key,
    i,
    output_dir: str,
    sample_id: str,
    extractor,
    label=EXTRACTOR_DEFAULT_LABEL,
    save_patches=True,
    mask_source: str = MASK_SOURCE_THRESHOLD,
    cellseg_df: Optional[gpd.GeoDataFrame] = None,
    shape_extractor=None,
):
    patch = load_patch_data(
        f=f,
        img_key=img_key,
        coords_key=coords_key,
        barcodes_key=barcodes_key,
        patch_idx=i,
    )
    row = build_patch_row_base(
        patch=patch,
        output_dir=output_dir,
        sample_id=sample_id,
        save_patches=save_patches,
    )

    if mask_source == MASK_SOURCE_THRESHOLD:
        return process_threshold_patch(
            patch=patch,
            row=row,
            extractor=extractor,
            output_dir=output_dir,
            sample_id=sample_id,
            label=label,
            save_patches=save_patches,
        )

    if mask_source == MASK_SOURCE_CELLSEG:
        if cellseg_df is None:
            raise ValueError(f"mask_source='{MASK_SOURCE_CELLSEG}' requires cellseg_df")

        if shape_extractor is None:
            shape_extractor = _get_worker_shape2d_extractor(label)

        return process_cellseg_patch(
            patch=patch,
            row=row,
            extractor=extractor,
            shape_extractor=shape_extractor,
            output_dir=output_dir,
            sample_id=sample_id,
            label=label,
            save_patches=save_patches,
            cellseg_df=cellseg_df,
        )

    raise ValueError(f"Unsupported mask_source: {mask_source}")


