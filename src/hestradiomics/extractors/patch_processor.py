from __future__ import annotations

from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np

from hestradiomics.extractors._intensity_texture import (
    _add_prefix_to_keys,
    _execute_radiomics_on_mask,
)
from hestradiomics.extractors import (
    extract_patch_level_radiomics,
    extract_cellseg_level_radiomics,
    extract_morphology_aggregates,
    extract_cell_type_distribution,
)
from hestradiomics.extractors.constants import *
from hestradiomics.utils import (
    build_threshold_mask,
    rasterize_geometries_to_mask,
    normalize_class_name,
    save_region_mask_images,
    safe_update_features,
    load_patch_data,
    build_patch_row_base,
    PatchData,
)


# ------------------------------------------------------------------------------
# Patch processors
# ------------------------------------------------------------------------------
# This module contains the main patch-level processing pipeline used for:
#
#   1. Threshold-based radiomics extraction
#   2. Cell segmentation-based radiomics extraction
#   3. Cell morphology aggregation
#   4. Cell-type distribution analysis
#
# Processing flow:
#
#   HDF5 Patch
#       ↓
#   Load patch image / metadata
#       ↓
#   Build mask (threshold or cell segmentation)
#       ↓
#   Extract radiomics features
#       ↓
#   Extract morphology features
#       ↓
#   Aggregate outputs into row dictionary
#
# ------------------------------------------------------------------------------


def get_patch_cellseg(
    cellseg_df: gpd.GeoDataFrame,
    patch_idx: int,
) -> gpd.GeoDataFrame:
    """
    Retrieve cell segmentation polygons belonging to a specific patch.

    Steps:
        1. Filter segmentation rows by patch index
        2. Remove invalid/null geometries
        3. Normalize cell class names

    Args:
        cellseg_df:
            Full cell segmentation dataframe.

        patch_idx:
            Target patch index.

    Returns:
        GeoDataFrame containing only the cells inside the patch.
    """

    # Select segmentation rows belonging to this patch
    patch_cellseg = cellseg_df[
        cellseg_df[PATCH_IDX_COLUMN] == patch_idx
    ].copy()

    # Remove invalid geometries
    patch_cellseg = patch_cellseg[
        patch_cellseg.geometry.notnull()
    ].copy()

    # Normalize cell class labels if cells exist
    if len(patch_cellseg) > 0:
        patch_cellseg[CELL_CLASS_COLUMN] = (
            patch_cellseg[CELL_CLASS_COLUMN]
            .map(normalize_class_name)
        )

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
    """
    Process a patch using threshold-based foreground masking.

    Workflow:
        1. Build binary threshold mask
        2. Save visualization masks (optional)
        3. Skip if mask area is too small
        4. Extract patch-level radiomics features

    Args:
        patch:
            Loaded patch object.

        row:
            Output feature dictionary.

        extractor:
            PyRadiomics extractor instance.

        output_dir:
            Directory for saving outputs.

        sample_id:
            Sample identifier.

        label:
            Foreground mask label value.

        save_patches:
            Whether to save visualization images.

    Returns:
        Updated feature dictionary.
    """

    # Build threshold-based foreground mask
    patch_mask = build_threshold_mask(
        patch.gray_patch,
        label=label,
    )

    # Compute foreground area
    row[PATCH_MASK_AREA_COLUMN] = int(
        np.count_nonzero(patch_mask > 0)
    )

    # Save mask visualization if enabled
    if save_patches:
        row[MASK_PATH_COLUMN] = save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=patch_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}{THRESHOLD_MASK_SUFFIX}",
        )

    # Skip patch if foreground area is too small
    if row[PATCH_MASK_AREA_COLUMN] < PATCH_MASK_AREA_MIN_THRESHOLD:
        row[STATUS_COLUMN] = STATUS_SKIPPED_SMALL_MASK
        return row

    # Extract radiomics features on threshold mask
    safe_update_features(
        row,
        lambda: _add_prefix_to_keys(
            _execute_radiomics_on_mask(
                patch.gray_patch,
                patch_mask,
                extractor,
            ),
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
    """
    Process a patch using cell segmentation masks.

    Workflow:
        1. Load patch-specific cell polygons
        2. Rasterize polygons into merged mask
        3. Save visualization masks
        4. Extract:
            - Patch-level radiomics
            - Cellseg-level radiomics
            - Morphology aggregates
            - Cell-type distributions

    Args:
        patch:
            Loaded patch object.

        row:
            Output feature dictionary.

        extractor:
            Standard radiomics extractor.

        shape_extractor:
            Shape2D extractor for morphology features.

        output_dir:
            Output directory.

        sample_id:
            Sample identifier.

        label:
            Mask label value.

        save_patches:
            Whether to save visualization images.

        cellseg_df:
            Full segmentation dataframe.

    Returns:
        Updated feature dictionary.
    """

    # Retrieve cells belonging to this patch
    patch_cellseg = get_patch_cellseg(
        cellseg_df,
        patch.patch_idx,
    )

    # Store total number of cells
    row[N_CELLS_TOTAL_COLUMN] = int(len(patch_cellseg))

    # Handle empty segmentation case
    if len(patch_cellseg) == 0:
        row[STATUS_COLUMN] = STATUS_SKIPPED_NO_CELLSEG

        row.update(
            extract_cell_type_distribution(
                patch_cellseg
            )
        )

        return row

    # Merge all cell polygons into a single binary mask
    merged_mask = rasterize_geometries_to_mask(
        patch_cellseg.geometry.tolist(),
        image_shape=patch.gray_patch.shape,
        label=label,
    )

    # Compute merged mask area
    row[CELLSEG_MASK_AREA_COLUMN] = int(
        np.count_nonzero(merged_mask > 0)
    )

    # Save merged segmentation mask visualization
    if save_patches:
        row[MASK_PATH_COLUMN] = save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=merged_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}{CELLSEG_ALL_MASK_SUFFIX}",
        )

    # --------------------------------------------------------------------------
    # Save class-specific masks
    # --------------------------------------------------------------------------
    if save_patches:

        for class_name, sub in patch_cellseg.groupby(CELL_CLASS_COLUMN):

            if len(sub) == 0:
                continue

            # Rasterize only cells of this class
            mask_cls = rasterize_geometries_to_mask(
                sub.geometry.tolist(),
                image_shape=patch.gray_patch.shape,
                label=label,
            )

            # Save visualization
            save_region_mask_images(
                color_patch=patch.color_patch,
                gray_patch=patch.gray_patch,
                mask_patch=mask_cls,
                output_dir=output_dir,
                sample_id=sample_id,
                mask_filename=f"{patch.base_filename}__cellseg_{class_name}",
            )

    # --------------------------------------------------------------------------
    # Save threshold mask for comparison/debugging
    # --------------------------------------------------------------------------
    if save_patches:

        threshold_mask = build_threshold_mask(
            patch.gray_patch,
            label=label,
        )

        save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=threshold_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}{THRESHOLD_MASK_SUFFIX}",
        )

    # --------------------------------------------------------------------------
    # Extract patch-level radiomics
    # --------------------------------------------------------------------------
    safe_update_features(
        row,
        lambda: extract_patch_level_radiomics(
            patch.gray_patch,
            extractor,
            label=label,
        ),
        ERROR_PATCH_RADIOMICS,
    )

    # --------------------------------------------------------------------------
    # Extract cell segmentation-based radiomics
    # --------------------------------------------------------------------------
    safe_update_features(
        row,
        lambda: extract_cellseg_level_radiomics(
            patch.gray_patch,
            patch_cellseg,
            extractor,
            label=label,
        ),
        ERROR_CELLSEG_RADIOMICS,
    )

    # --------------------------------------------------------------------------
    # Extract morphology aggregate statistics
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Extract cell-type distribution statistics
    # --------------------------------------------------------------------------
    safe_update_features(
        row,
        lambda: extract_cell_type_distribution(
            patch_cellseg
        ),
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
    """
    Main entry point for processing a single patch.

    This function:
        1. Loads patch image + metadata
        2. Initializes output row
        3. Selects processing strategy
            - threshold-based
            - cellseg-based
        4. Returns extracted feature dictionary

    Args:
        f:
            Open HDF5 file handle.

        img_key:
            HDF5 image dataset key.

        coords_key:
            HDF5 coordinate dataset key.

        barcodes_key:
            HDF5 barcode dataset key.

        i:
            Patch index.

        output_dir:
            Output directory.

        sample_id:
            Sample identifier.

        extractor:
            Standard radiomics extractor.

        label:
            Mask label value.

        save_patches:
            Whether to save visualization images.

        mask_source:
            Mask generation strategy.
            Supported:
                - threshold
                - cellseg

        cellseg_df:
            Cell segmentation dataframe.

        shape_extractor:
            Shape2D extractor.

    Returns:
        Dictionary containing extracted features.
    """

    # Load patch image + metadata from HDF5
    patch = load_patch_data(
        f=f,
        img_key=img_key,
        coords_key=coords_key,
        barcodes_key=barcodes_key,
        patch_idx=i,
    )

    # Build base output row
    row = build_patch_row_base(
        patch=patch,
        output_dir=output_dir,
        sample_id=sample_id,
        save_patches=save_patches,
    )

    # --------------------------------------------------------------------------
    # Threshold-based processing
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Cell segmentation-based processing
    # --------------------------------------------------------------------------
    if mask_source == MASK_SOURCE_CELLSEG:

        # Cell segmentation dataframe is required
        if cellseg_df is None:
            raise ValueError(
                f"mask_source='{MASK_SOURCE_CELLSEG}' "
                f"requires cellseg_df"
            )

        # Lazily initialize Shape2D extractor
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

    # Unsupported processing mode
    raise ValueError(
        f"Unsupported mask_source: {mask_source}"
    )

