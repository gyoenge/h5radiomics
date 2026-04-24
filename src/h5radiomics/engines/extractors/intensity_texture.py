from __future__ import annotations

from typing import Any, Dict
import geopandas as gpd
import numpy as np
import SimpleITK as sitk
from h5radiomics.engines.constants import *
from h5radiomics.utils import (
    build_threshold_mask, 
    rasterize_geometries_to_mask, 
    make_feature_prefix, 
)


# ------------------------------------------------------------------------------
# radiomics execution helpers
# ------------------------------------------------------------------------------

def _is_radiomics_feature_key(k: str) -> bool:
    """check if given key has the expected feature prefix (e.g. original_)"""
    k = k.lower()
    return any(k.startswith(prefix) for prefix in RADIOMICS_IMAGE_PREFIXES)


def _clean_radiomics_result(feature_dict: Dict[str, Any]) -> Dict[str, float]:
    """clean radiomics result into float"""
    out = {}
    for k, v in feature_dict.items():
        if not _is_radiomics_feature_key(k):
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _add_prefix_to_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """add prefix into the given dictionary keys""" 
    return {f"{prefix}{k}": v for k, v in d.items()}


def _execute_radiomics_on_mask(
    gray_patch: np.ndarray,
    mask_patch: np.ndarray,
    extractor,
) -> Dict[str, float]:
    """
    Wrapper of the radiomics extractor execution
    It executes only on a valid ROI mask (need enoughly large ROI area)
    """
    if mask_patch is None or np.count_nonzero(mask_patch > 0) < EXTRACTOR_DEFAULT_MASK_ROI_AREA_THRESHOLD:
        return {}

    image_sitk = sitk.GetImageFromArray(gray_patch)
    mask_sitk = sitk.GetImageFromArray(mask_patch)
    features = extractor.execute(image_sitk, mask_sitk)
    return _clean_radiomics_result(features)



# ------------------------------------------------------------------------------
# feature extraction
# ------------------------------------------------------------------------------

def extract_patch_level_radiomics(
    gray_patch: np.ndarray,
    extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    patch_mask = build_threshold_mask(gray_patch, label=label)
    features = _execute_radiomics_on_mask(gray_patch, patch_mask, extractor)
    return _add_prefix_to_keys(features, "patch_")


def extract_cellseg_level_radiomics(
    gray_patch: np.ndarray,
    patch_cellseg: gpd.GeoDataFrame,
    extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    out = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        return out

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg = patch_cellseg[patch_cellseg.geometry.notnull()]
    if len(patch_cellseg) == 0:
        return out

    mask_all = rasterize_geometries_to_mask(
        patch_cellseg.geometry.tolist(),
        image_shape=gray_patch.shape,
        label=label,
    )
    all_feats = _execute_radiomics_on_mask(gray_patch, mask_all, extractor)
    out.update(_add_prefix_to_keys(all_feats, make_feature_prefix("cellseg", "all")))

    for class_name, sub in patch_cellseg.groupby("class_name"):
        if len(sub) == 0:
            continue

        mask_cls = rasterize_geometries_to_mask(
            sub.geometry.tolist(),
            image_shape=gray_patch.shape,
            label=label,
        )
        cls_feats = _execute_radiomics_on_mask(gray_patch, mask_cls, extractor)
        out.update(_add_prefix_to_keys(cls_feats, make_feature_prefix("cellseg", class_name)))

    return out
