from __future__ import annotations

from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # pyright: ignore[reportMissingImports]

from hestradiomics.extract.constants import *
from hestradiomics.utils import (
    align_local_mask_to_crop,
    build_local_polygon_mask,
    crop_patch_by_bbox,
    normalize_class_name,
    strip_shape2d_prefix,
)


class MorphologyExtractor:
    _PROCESS_LOCAL_CACHE: Dict[Any, Any] = {}

    def __init__(
        self,
        label: int = EXTRACTOR_DEFAULT_LABEL,
    ):
        self.label = label

    def build_shape_extractor(self):
        settings = {
            "label": self.label,
            "force2D": True,
            "force2Ddimension": 0,
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        extractor.enableImageTypeByName(RADIOMICS_IMAGE_TYPE_ORIGINAL)
        extractor.enableFeatureClassByName(RADIOMICS_FEATURE_CLASS_SHAPE2D)

        return extractor

    def get_cached_extractor(self):
        key = (
            RADIOMICS_FEATURE_CLASS_SHAPE2D,
            self.label,
        )

        if key not in self._PROCESS_LOCAL_CACHE:
            self._PROCESS_LOCAL_CACHE[key] = self.build_shape_extractor()

        return self._PROCESS_LOCAL_CACHE[key]

    def extract(
        self,
        gray_patch: np.ndarray,
        patch_cellseg: gpd.GeoDataFrame,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        if patch_cellseg is None or len(patch_cellseg) == 0:
            return {}

        shape_extractor = (
            self.get_cached_extractor()
            if use_cache
            else self.build_shape_extractor()
        )

        return extract_morphology_aggregates(
            gray_patch=gray_patch,
            patch_cellseg=patch_cellseg,
            label=self.label,
            shape_extractor=shape_extractor,
        )


def extract_single_cell_shape_features(
    gray_patch: np.ndarray,
    geom,
    shape_extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    mask_local, bbox_xyxy = build_local_polygon_mask(
        geom,
        label=label,
        margin=LOCAL_POLYGON_MASK_MARGIN,
    )

    gray_crop = crop_patch_by_bbox(
        gray_patch,
        bbox_xyxy,
    )

    mask_crop = align_local_mask_to_crop(
        mask_local,
        bbox_xyxy,
        gray_patch.shape,
    )

    if gray_crop.size == 0 or mask_crop.size == 0:
        return {}

    if gray_crop.shape != mask_crop.shape:
        return {}

    if np.count_nonzero(mask_crop > 0) < LOCAL_MASK_MIN_PIXELS:
        return {}

    image_sitk = sitk.GetImageFromArray(gray_crop)
    mask_sitk = sitk.GetImageFromArray(mask_crop)

    result = shape_extractor.execute(
        image_sitk,
        mask_sitk,
    )

    out: Dict[str, float] = {}

    for k, v in result.items():
        if RADIOMICS_FEATURE_CLASS_SHAPE2D.lower() not in str(k).lower():
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
    out: Dict[str, float] = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        return out

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg = patch_cellseg[patch_cellseg.geometry.notnull()]

    if len(patch_cellseg) == 0:
        return out

    if shape_extractor is None:
        shape_extractor = MorphologyExtractor(
            label=label,
        ).get_cached_extractor()

    per_cell_rows = []

    for _, row in patch_cellseg.iterrows():
        geom = row.geometry

        class_name = normalize_class_name(
            row.get(
                CELL_CLASS_COLUMN,
                UNKNOWN_CELL_CLASS,
            )
        )

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

    morph_cols = [
        col
        for col in cell_df.columns
        if col != CELL_CLASS_COLUMN
    ]

    for col in morph_cols:
        values = (
            pd.to_numeric(
                cell_df[col],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )

        agg = _execute_firstorder_aggregation(values)
        base_name = strip_shape2d_prefix(col)

        for stat_name, stat_value in agg.items():
            out[
                f"{MORPH_FEATURE_PREFIX}{base_name}_{stat_name.lower()}"
            ] = stat_value

    return out


def _execute_firstorder_aggregation(
    values: List[float],
) -> Dict[str, float]:
    s = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    s = (
        s.replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

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
        hist, _ = np.histogram(
            arr,
            bins=min(
                FIRSTORDER_HIST_MAX_BINS,
                max(FIRSTORDER_HIST_MIN_BINS, n),
            ),
        )

        prob = hist.astype(np.float64)
        prob = prob[prob > 0]
        prob = prob / prob.sum()

        entropy_val = float(
            -np.sum(prob * np.log2(prob))
        )

    mad_val = float(np.mean(np.abs(arr - mean_val)))
    energy_val = float(np.sum(arr ** 2))
    rms_val = float(np.sqrt(np.mean(arr ** 2)))

    stat_map = {
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
        "iqr": iqr_val,
        "meanabsolutedeviation": mad_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "entropy": entropy_val,
        "energy": energy_val,
        "rootmeansquared": rms_val,
    }

    return {
        key: stat_map[key]
        for key in MORPH_AGG_STAT_KEYS
    }

