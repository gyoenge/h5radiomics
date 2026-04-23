from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor  # pyright: ignore[reportMissingImports]
from tqdm import tqdm

from h5radiomics.engines.constants import *
from h5radiomics.engines.extract import (
    PatchData,
    _WORKER_EXTRACTOR_CACHE,
    _WORKER_SHAPE_EXTRACTOR_CACHE,
)
from h5radiomics.utils import (
    get_barcodes_key, get_coords_key, get_img_key,
    build_threshold_mask, build_local_polygon_mask, 
    rasterize_geometries_to_mask, crop_patch_by_bbox, align_local_mask_to_crop, 
    make_feature_prefix, normalize_class_name, strip_shape2d_prefix, 
    save_region_mask_images, safe_update_features, 
    load_patch_data, load_cellseg_dataframe, 
    build_patch_row_base, make_error_row, 
)

logging.getLogger("radiomics").setLevel(logging.ERROR)

### 

_WORKER_EXTRACTOR_CACHE: Dict[Any, Any] = {}
_WORKER_SHAPE_EXTRACTOR_CACHE: Dict[Any, Any] = {}

@dataclass
class PatchData:
    patch_idx: int
    color_patch: np.ndarray
    gray_patch: np.ndarray
    coords: Optional[Any]
    barcode: Optional[str]
    base_filename: str

# ------------------------------------------------------------------------------
# extractor builders
# ------------------------------------------------------------------------------

def build_radiomics_extractor(
    classes=None,
    filters=None,
    label=EXTRACTOR_DEFAULT_LABEL,
    image_type_settings=None,
):
    """
    For intensity/texture extraction on image patch + ROI mask.
    """
    if classes is None:
        classes = EXTRACTOR_DEFAULT_CLASSES
    if filters is None:
        filters = EXTRACTOR_DEFAULT_FILTERS
    if image_type_settings is None:
        image_type_settings = EXTRACTOR_DEFAULT_IMAGE_TYPE_SETTINGS

    extractor = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_DEFAULT_SETTINGS)
    extractor.disableAllFeatures()

    for cls in classes:
        extractor.enableFeatureClassByName(cls)

    image_types = set(filters or [])
    image_types.add("Original")

    for filt in image_types:
        if filt == "LoG":
            log_cfg = image_type_settings.get("LoG", {})
            sigma = log_cfg.get("sigma", EXTRACTOR_DEFAULT_LOGFILTER_SIGMA)
            if not isinstance(sigma, (list, tuple)) or len(sigma) == 0:
                raise ValueError("image_type_settings['LoG']['sigma'] must be a non-empty list")
            extractor.enableImageTypeByName("LoG", customArgs={"sigma": list(sigma)})
        else:
            extractor.enableImageTypeByName(filt)

    return extractor


def build_shape2d_extractor(label=EXTRACTOR_DEFAULT_LABEL):
    """
    For per-cell morphology extraction on local binary mask.
    """
    settings = {
        "label": label,
        "force2D": True,
        "force2Ddimension": 0,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape2D")
    extractor.enableImageTypeByName("Original")
    return extractor


def _get_worker_radiomics_extractor(classes, filters, label, image_type_settings):
    key = (
        tuple(classes or []),
        tuple(filters or []),
        label,
        repr(image_type_settings or {}),
    )
    if key not in _WORKER_EXTRACTOR_CACHE:
        _WORKER_EXTRACTOR_CACHE[key] = build_radiomics_extractor(
            classes=classes,
            filters=filters,
            label=label,
            image_type_settings=image_type_settings,
        )
    return _WORKER_EXTRACTOR_CACHE[key]


def _get_worker_shape2d_extractor(label):
    key = ("shape2d", label)
    if key not in _WORKER_SHAPE_EXTRACTOR_CACHE:
        _WORKER_SHAPE_EXTRACTOR_CACHE[key] = build_shape2d_extractor(label=label)
    return _WORKER_SHAPE_EXTRACTOR_CACHE[key]


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
        class_name = normalize_class_name(r.get("class_name", "unknown"))
        try:
            feats = extract_single_cell_shape_features(
                gray_patch=gray_patch,
                geom=geom,
                shape_extractor=shape_extractor,
                label=label,
            )
            if not feats:
                continue
            feats["class_name"] = class_name
            per_cell_rows.append(feats)
        except Exception:
            continue

    if not per_cell_rows:
        return out

    cell_df = pd.DataFrame(per_cell_rows)
    morph_cols = [c for c in cell_df.columns if c != "class_name"]

    for col in morph_cols:
        vals = pd.to_numeric(cell_df[col], errors="coerce").dropna().tolist()
        agg = _execute_firstorder_aggregation(vals)
        base_name = strip_shape2d_prefix(col)
        for stat_name, stat_val in agg.items():
            out[f"morph_{base_name}_{stat_name.lower()}"] = stat_val

    for class_name, sub in cell_df.groupby("class_name"):
        safe_class = normalize_class_name(class_name)
        for col in morph_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().tolist()
            agg = _execute_firstorder_aggregation(vals)
            base_name = strip_shape2d_prefix(col)
            for stat_name, stat_val in agg.items():
                out[f"morph_{safe_class}_{base_name}_{stat_name.lower()}"] = stat_val

    return out


# ------------------------------------------------------------------------------
# cell-type distribution features
# ------------------------------------------------------------------------------

def extract_cell_type_distribution(
    patch_cellseg: gpd.GeoDataFrame,
) -> Dict[str, float]:
    out = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        out["dist_cell_count_total"] = 0.0
        for cls in KNOWN_CELL_CLASSES:
            out[f"dist_count_{cls}"] = 0.0
            out[f"dist_ratio_{cls}"] = 0.0
        return out

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg["class_name"] = (
        patch_cellseg["class_name"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    counts = patch_cellseg["class_name"].value_counts().to_dict()
    total = int(len(patch_cellseg))

    out["dist_cell_count_total"] = float(total)

    class_names = sorted(set(KNOWN_CELL_CLASSES).union(set(counts.keys())))
    for cls in class_names:
        cnt = int(counts.get(cls, 0))
        ratio = float(cnt / total) if total > 0 else 0.0
        out[f"dist_count_{cls}"] = float(cnt)
        out[f"dist_ratio_{cls}"] = ratio

    return out


# ------------------------------------------------------------------------------
# patch processors
# ------------------------------------------------------------------------------

def get_patch_cellseg(
    cellseg_df: gpd.GeoDataFrame,
    patch_idx: int,
) -> gpd.GeoDataFrame:
    patch_cellseg = cellseg_df[cellseg_df["patch_idx"] == patch_idx].copy()
    patch_cellseg = patch_cellseg[patch_cellseg.geometry.notnull()].copy()

    if len(patch_cellseg) > 0:
        patch_cellseg["class_name"] = patch_cellseg["class_name"].map(normalize_class_name)

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
    row["patch_mask_area"] = int(np.count_nonzero(patch_mask > 0))

    if save_patches:
        row["mask_path"] = save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=patch_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}__threshold",
        )

    if row["patch_mask_area"] < 50:
        row["status"] = "skipped_small_mask"
        return row

    safe_update_features(
        row,
        lambda: _add_prefix_to_keys(
            _execute_radiomics_on_mask(patch.gray_patch, patch_mask, extractor),
            "patch_",
        ),
        "error_patch_radiomics",
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
    row["n_cells_total"] = int(len(patch_cellseg))

    if len(patch_cellseg) == 0:
        row["status"] = "skipped_no_cellseg_mask"
        row.update(extract_cell_type_distribution(patch_cellseg))
        return row

    merged_mask = rasterize_geometries_to_mask(
        patch_cellseg.geometry.tolist(),
        image_shape=patch.gray_patch.shape,
        label=label,
    )
    row["cellseg_mask_area"] = int(np.count_nonzero(merged_mask > 0))

    if save_patches:
        row["mask_path"] = save_region_mask_images(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            mask_patch=merged_mask,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_filename=f"{patch.base_filename}__cellseg_all",
        )

    # if save_patches:
    #     for class_name, sub in patch_cellseg.groupby("class_name"):
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
            mask_filename=f"{patch.base_filename}__threshold",
        )

    safe_update_features(
        row,
        lambda: extract_patch_level_radiomics(patch.gray_patch, extractor, label=label),
        "error_patch_radiomics",
    )

    safe_update_features(
        row,
        lambda: extract_cellseg_level_radiomics(
            patch.gray_patch, patch_cellseg, extractor, label=label
        ),
        "error_cellseg_radiomics",
    )

    safe_update_features(
        row,
        lambda: extract_morphology_aggregates(
            patch.gray_patch,
            patch_cellseg,
            label=label,
            shape_extractor=shape_extractor,
        ),
        "error_morphology",
    )

    safe_update_features(
        row,
        lambda: extract_cell_type_distribution(patch_cellseg),
        "error_distribution",
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
    mask_source: str = "threshold",
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

    if mask_source == "threshold":
        return process_threshold_patch(
            patch=patch,
            row=row,
            extractor=extractor,
            output_dir=output_dir,
            sample_id=sample_id,
            label=label,
            save_patches=save_patches,
        )

    if mask_source == "cellseg":
        if cellseg_df is None:
            raise ValueError("mask_source='cellseg' requires cellseg_df")

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


# ------------------------------------------------------------------------------
# post-processing helpers
# ------------------------------------------------------------------------------

def is_processed_feature_column(col: str) -> bool:
    col_lower = col.lower()

    if col_lower.startswith("morph_"):
        return True

    if not (col_lower.startswith("patch_") or col_lower.startswith("cellseg_")):
        return False

    remainder = col_lower.split("_", 1)[1] if "_" in col_lower else ""
    return any(token in remainder for token in RADIOMICS_IMAGE_PREFIXES)


def get_radiomics_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if is_processed_feature_column(col)]


def clip_feature_series(
    s: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.Series, Dict[str, float]]:
    s_num = pd.to_numeric(s, errors="coerce")

    valid = s_num.dropna()
    if valid.empty:
        return s_num, {
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "min_after_clip": np.nan,
            "max_after_clip": np.nan,
        }

    lower_bound = valid.quantile(lower_q)
    upper_bound = valid.quantile(upper_q)

    clipped = s_num.clip(lower=lower_bound, upper=upper_bound)

    return clipped, {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "mean": float(clipped.mean()) if not pd.isna(clipped.mean()) else np.nan,
        "std": float(clipped.std(ddof=0)) if not pd.isna(clipped.std(ddof=0)) else np.nan,
        "min_after_clip": float(clipped.min()) if not pd.isna(clipped.min()) else np.nan,
        "max_after_clip": float(clipped.max()) if not pd.isna(clipped.max()) else np.nan,
    }


def z_normalize_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    mean = s_num.mean()
    std = s_num.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s_num)), index=s_num.index, dtype=float)

    return (s_num - mean) / std


def minmax_rescale_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    min_val = s_num.min()
    max_val = s_num.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.zeros(len(s_num)), index=s_num.index, dtype=float)

    return (s_num - min_val) / (max_val - min_val)


def build_processed_feature_df(
    df: pd.DataFrame,
    status_col: str = "status",
    ok_status: str = "ok",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_df = df.copy()
    feature_cols = get_radiomics_feature_columns(df)

    if not feature_cols:
        sample_cols = df.columns[:30].tolist()
        raise ValueError(
            f"No radiomics/morphology feature columns found to process. "
            f"Sample columns: {sample_cols}"
        )

    stats_rows = []
    ok_mask = processed_df[status_col] == ok_status

    for col in feature_cols:
        original_series = pd.to_numeric(processed_df.loc[ok_mask, col], errors="coerce")

        clipped, clip_stats = clip_feature_series(
            original_series,
            lower_q=lower_q,
            upper_q=upper_q,
        )

        z_norm = z_normalize_series(clipped)
        scaled = minmax_rescale_series(z_norm)

        processed_df.loc[ok_mask, col] = scaled.astype(float)
        processed_df.loc[~ok_mask, col] = np.nan

        stats_rows.append(
            {
                "feature": col,
                "lower_q": lower_q,
                "upper_q": upper_q,
                **clip_stats,
                "z_mean": float(z_norm.mean()) if len(z_norm.dropna()) else np.nan,
                "z_std": float(z_norm.std(ddof=0)) if len(z_norm.dropna()) else np.nan,
                "scaled_min": float(scaled.min()) if len(scaled.dropna()) else np.nan,
                "scaled_max": float(scaled.max()) if len(scaled.dropna()) else np.nan,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    return processed_df, stats_df
