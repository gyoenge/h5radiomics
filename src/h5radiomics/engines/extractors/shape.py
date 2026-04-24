from __future__ import annotations

from typing import Dict, List
import geopandas as gpd
import numpy as np
import pandas as pd 
import SimpleITK as sitk
from h5radiomics.engines.extractors.constants import *
from h5radiomics.utils import (
    build_local_polygon_mask, 
    crop_patch_by_bbox, align_local_mask_to_crop, 
    normalize_class_name, strip_shape2d_prefix,  
)
from h5radiomics.engines.extractors.builders import (
    build_shape2d_extractor
)


# ------------------------------------------------------------------------------
# radiomics execution helpers
# ------------------------------------------------------------------------------

def _execute_firstorder_aggregation(values: List[float]) -> Dict[str, float]:
    """
    Safe manual first-order aggregation for morphology feature vectors.
    Avoid PyRadiomics firstorder here because it may emit RuntimeWarning
    for short / degenerate vectors.
    """
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) == 0:
        return {}

    arr = s.to_numpy(dtype=np.float64)
    n = len(arr)

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = float(max_val - min_val)
    var_val = float(np.var(arr, ddof=0))
    std_val = float(np.std(arr, ddof=0))
    p10_val = float(np.percentile(arr, 10))
    p25_val = float(np.percentile(arr, 25))
    p75_val = float(np.percentile(arr, 75))
    p90_val = float(np.percentile(arr, 90))
    iqr_val = float(p75_val - p25_val)

    if std_val == 0.0:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        z = (arr - mean_val) / std_val
        skew_val = float(np.mean(z ** 3))
        kurt_val = float(np.mean(z ** 4) - 3.0)

    if n <= 1 or min_val == max_val:
        entropy_val = 0.0
    else:
        hist, _ = np.histogram(arr, bins=min(16, max(2, n)))
        prob = hist.astype(np.float64)
        prob = prob[prob > 0]
        prob = prob / prob.sum()
        entropy_val = float(-np.sum(prob * np.log2(prob)))

    mad_val = float(np.mean(np.abs(arr - mean_val)))
    energy_val = float(np.sum(arr ** 2))
    rms_val = float(np.sqrt(np.mean(arr ** 2)))

    return {
        # "count": float(n),
        "mean": mean_val,
        "median": median_val,
        "minimum": min_val,
        "maximum": max_val,
        "range": range_val,
        "variance": var_val,
        "standarddeviation": std_val,
        "p10": p10_val,
        "p25": p25_val,
        "p75": p75_val,
        "p90": p90_val,
        "iqr": iqr_val, # inter quartile range 
        "meanabsolutedeviation": mad_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "entropy": entropy_val,
        "energy": energy_val,
        "rootmeansquared": rms_val,
    }



# ------------------------------------------------------------------------------
# morphology features
# ------------------------------------------------------------------------------

def extract_single_cell_shape_features(
    gray_patch: np.ndarray,
    geom,
    shape_extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    mask_local, bbox_xyxy = build_local_polygon_mask(geom, label=label, margin=1)
    gray_crop = crop_patch_by_bbox(gray_patch, bbox_xyxy)
    mask_crop = align_local_mask_to_crop(mask_local, bbox_xyxy, gray_patch.shape)

    if gray_crop.size == 0 or mask_crop.size == 0:
        return {}
    if gray_crop.shape != mask_crop.shape:
        return {}
    if np.count_nonzero(mask_crop > 0) < 3:
        return {}

    image_sitk = sitk.GetImageFromArray(gray_crop)
    mask_sitk = sitk.GetImageFromArray(mask_crop)

    result = shape_extractor.execute(image_sitk, mask_sitk)
    out = {}
    for k, v in result.items():
        if "shape2d" not in str(k).lower():
            continue
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def extract_morphology_aggregates(
    gray_patch: np.ndarray,
    patch_cellseg: gpd.GeoDataFrame,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    shape_extractor=None,
) -> Dict[str, float]:
    """
    1) per-cell shape2D extraction
    2) patch-level first-order aggregation over each morphology feature vector
    3) also per-class aggregation
    """
    out = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        return out

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg = patch_cellseg[patch_cellseg.geometry.notnull()]
    if len(patch_cellseg) == 0:
        return out

    if shape_extractor is None:
        shape_extractor = build_shape2d_extractor(label=label)

    per_cell_rows = []
    for _, r in patch_cellseg.iterrows():
        geom = r.geometry
        class_name = normalize_class_name(r.get(CELL_CLASS_COLUMN, "unknown"))
        try:
            feats = extract_single_cell_shape_features(
                gray_patch=gray_patch,
                geom=geom,
                shape_extractor=shape_extractor,
                label=label,
            )
            if not feats:
                continue
            feats[CELL_CLASS_COLUMN] = class_name
            per_cell_rows.append(feats)
        except Exception:
            continue

    if not per_cell_rows:
        return out

    cell_df = pd.DataFrame(per_cell_rows)
    morph_cols = [c for c in cell_df.columns if c != CELL_CLASS_COLUMN]

    for col in morph_cols:
        vals = pd.to_numeric(cell_df[col], errors="coerce").dropna().tolist()
        agg = _execute_firstorder_aggregation(vals)
        base_name = strip_shape2d_prefix(col)
        for stat_name, stat_val in agg.items():
            out[f"morph_{base_name}_{stat_name.lower()}"] = stat_val

    for class_name, sub in cell_df.groupby(CELL_CLASS_COLUMN):
        safe_class = normalize_class_name(class_name)
        for col in morph_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().tolist()
            agg = _execute_firstorder_aggregation(vals)
            base_name = strip_shape2d_prefix(col)
            for stat_name, stat_val in agg.items():
                out[f"morph_{safe_class}_{base_name}_{stat_name.lower()}"] = stat_val

    return out


