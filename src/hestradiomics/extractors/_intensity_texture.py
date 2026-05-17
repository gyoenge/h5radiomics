from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd
import numpy as np
import SimpleITK as sitk

from hestradiomics.extractors.constants import *

from hestradiomics.utils import (
    build_threshold_mask,
    rasterize_geometries_to_mask,
    make_feature_prefix,
)


# ------------------------------------------------------------------------------
# Radiomics execution helpers
# ------------------------------------------------------------------------------
# This module provides low-level helper utilities for:
#
#   1. Executing PyRadiomics extraction
#   2. Cleaning radiomics outputs
#   3. Applying feature prefixes
#   4. Extracting:
#       - patch-level radiomics
#       - cell segmentation-level radiomics
#
# Main workflow:
#
#   grayscale patch
#       ↓
#   build ROI mask
#       ↓
#   run PyRadiomics
#       ↓
#   clean valid features
#       ↓
#   apply feature prefixes
#
# ------------------------------------------------------------------------------


def _is_radiomics_feature_key(k: str) -> bool:
    """
    Check whether a feature key belongs to a valid
    radiomics image type.

    Example valid prefixes:
        - original_
        - wavelet-
        - log-sigma-
        - square_

    Args:
        k:
            Feature key.

    Returns:
        True if feature key is valid.
    """

    k = k.lower()

    return any(
        k.startswith(prefix)
        for prefix in RADIOMICS_IMAGE_PREFIXES
    )


def _clean_radiomics_result(
    feature_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Clean raw PyRadiomics output dictionary.

    Processing:
        1. Keep only valid radiomics feature keys
        2. Convert values safely to float
        3. Skip invalid/non-numeric entries

    Args:
        feature_dict:
            Raw PyRadiomics output.

    Returns:
        Cleaned radiomics feature dictionary.
    """

    out = {}

    for k, v in feature_dict.items():

        # Skip non-radiomics metadata fields
        if not _is_radiomics_feature_key(k):
            continue

        try:
            out[k] = float(v)

        except Exception:
            # Skip invalid values safely
            continue

    return out


def _add_prefix_to_keys(
    d: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    Add a prefix to all dictionary keys.

    Example:
        Input:
            {"original_firstorder_mean": 1.2}

        Prefix:
            "patch_"

        Output:
            {"patch_original_firstorder_mean": 1.2}

    Args:
        d:
            Input dictionary.

        prefix:
            Prefix string.

    Returns:
        Dictionary with prefixed keys.
    """

    return {
        f"{prefix}{k}": v
        for k, v in d.items()
    }


def _execute_radiomics_on_mask(
    gray_patch: np.ndarray,
    mask_patch: np.ndarray,
    extractor,
) -> Dict[str, float]:
    """
    Execute PyRadiomics feature extraction on a mask ROI.

    Safety checks:
        - mask must exist
        - ROI area must be sufficiently large

    Workflow:
        1. Validate ROI area
        2. Convert arrays to SimpleITK images
        3. Run PyRadiomics extractor
        4. Clean extracted features

    Args:
        gray_patch:
            Grayscale image patch.

        mask_patch:
            Binary ROI mask.

        extractor:
            Initialized PyRadiomics extractor.

    Returns:
        Cleaned radiomics feature dictionary.
    """

    # Skip invalid or too-small masks
    if (
        mask_patch is None
        or np.count_nonzero(mask_patch > 0)
        < EXTRACTOR_DEFAULT_MASK_ROI_AREA_THRESHOLD
    ):
        return {}

    # Convert numpy arrays to SimpleITK images
    image_sitk = sitk.GetImageFromArray(gray_patch)
    mask_sitk = sitk.GetImageFromArray(mask_patch)

    # Execute PyRadiomics extraction
    features = extractor.execute(
        image_sitk,
        mask_sitk,
    )

    # Clean output features
    return _clean_radiomics_result(features)


# ------------------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------------------

def extract_patch_level_radiomics(
    gray_patch: np.ndarray,
    extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    """
    Extract patch-level radiomics features using
    threshold-based foreground masking.

    Workflow:
        1. Build threshold mask
        2. Execute PyRadiomics
        3. Apply patch feature prefix

    Args:
        gray_patch:
            Grayscale patch image.

        extractor:
            PyRadiomics extractor.

        label:
            Foreground mask label value.

    Returns:
        Patch-level radiomics feature dictionary.
    """

    # Build threshold foreground mask
    patch_mask = build_threshold_mask(
        gray_patch,
        label=label,
    )

    # Extract radiomics features
    features = _execute_radiomics_on_mask(
        gray_patch,
        patch_mask,
        extractor,
    )

    # Add patch-level feature prefix
    return _add_prefix_to_keys(
        features,
        PATCH_FEATURE_PREFIX,
    )


def extract_cellseg_level_radiomics(
    gray_patch: np.ndarray,
    patch_cellseg: gpd.GeoDataFrame,
    extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    """
    Extract radiomics features from cell segmentation regions.

    Current implementation:
        - Uses all cell polygons merged together

    Optional future extension:
        - Per-cell-class radiomics extraction

    Workflow:
        1. Validate segmentation polygons
        2. Rasterize polygons into mask
        3. Execute PyRadiomics
        4. Apply cellseg feature prefixes

    Args:
        gray_patch:
            Grayscale patch image.

        patch_cellseg:
            Cell segmentation dataframe.

        extractor:
            PyRadiomics extractor.

        label:
            Foreground label value.

    Returns:
        Cell segmentation-level radiomics features.
    """

    out = {}

    # --------------------------------------------------------------------------
    # Handle empty segmentation safely
    # --------------------------------------------------------------------------
    if patch_cellseg is None or len(patch_cellseg) == 0:
        return out

    patch_cellseg = patch_cellseg.copy()

    # Remove invalid geometries
    patch_cellseg = patch_cellseg[
        patch_cellseg.geometry.notnull()
    ]

    if len(patch_cellseg) == 0:
        return out

    # --------------------------------------------------------------------------
    # Build merged cell segmentation mask
    # --------------------------------------------------------------------------
    mask_all = rasterize_geometries_to_mask(
        patch_cellseg.geometry.tolist(),
        image_shape=gray_patch.shape,
        label=label,
    )

    # --------------------------------------------------------------------------
    # Extract radiomics on merged cell mask
    # --------------------------------------------------------------------------
    all_feats = _execute_radiomics_on_mask(
        gray_patch,
        mask_all,
        extractor,
    )

    # Store features with standardized prefix
    out.update(
        _add_prefix_to_keys(
            all_feats,
            make_feature_prefix(
                CELLSEG_FEATURE_PREFIX,
                CELLSEG_ALL_SUFFIX,
            ),
        )
    )

    # --------------------------------------------------------------------------
    # Optional class-specific extraction
    # --------------------------------------------------------------------------
    # Example:
    #     cellseg_neoplastic_original_glcm_contrast
    #     cellseg_epithelial_original_firstorder_entropy
    #
    # Currently disabled to reduce feature dimensionality.
    # --------------------------------------------------------------------------
    # for class_name, sub in patch_cellseg.groupby(CELL_CLASS_COLUMN):
    #
    #     if len(sub) == 0:
    #         continue
    #
    #     mask_cls = rasterize_geometries_to_mask(
    #         sub.geometry.tolist(),
    #         image_shape=gray_patch.shape,
    #         label=label,
    #     )
    #
    #     cls_feats = _execute_radiomics_on_mask(
    #         gray_patch,
    #         mask_cls,
    #         extractor,
    #     )
    #
    #     out.update(
    #         _add_prefix_to_keys(
    #             cls_feats,
    #             make_feature_prefix(
    #                 CELLSEG_FEATURE_PREFIX,
    #                 class_name,
    #             ),
    #         )
    #     )

    return out