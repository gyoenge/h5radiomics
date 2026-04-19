from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw
from radiomics import featureextractor
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from tqdm import tqdm

from h5radiomics.utils.h5 import (
    get_barcodes_key,
    get_coords_key,
    get_img_key,
    to_str_barcode,
)
from h5radiomics.utils.io import make_base_name
from h5radiomics.utils.paths import (
    get_patch_color_dir,
    get_patch_gray_dir,
    get_patch_mask_dir,
    get_patch_masked_color_dir,
    get_patch_masked_gray_dir,
)

logging.getLogger("radiomics").setLevel(logging.ERROR)


# ------------------------------------------------------------------------------
# extractor
# ------------------------------------------------------------------------------

def build_radiomics_extractor(
    classes=None,
    filters=None,
    label=255,
    image_type_settings=None,
):
    if classes is None:
        classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    if filters is None:
        filters = ["Original"]
    if image_type_settings is None:
        image_type_settings = {}

    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "verbose": False,
        "label": label,
        "force2D": True,
        "force2Ddimension": 0,
        "distances": [1],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    for cls in classes:
        extractor.enableFeatureClassByName(cls)

    image_types = set(filters or [])
    image_types.add("Original")

    for filt in image_types:
        if filt == "LoG":
            log_cfg = image_type_settings.get("LoG", {})
            sigma = log_cfg.get("sigma", [1.0, 2.0, 3.0])
            if not isinstance(sigma, (list, tuple)) or len(sigma) == 0:
                raise ValueError("image_type_settings['LoG']['sigma'] must be a non-empty list")
            extractor.enableImageTypeByName("LoG", customArgs={"sigma": list(sigma)})
        else:
            extractor.enableImageTypeByName(filt)

    return extractor


# ------------------------------------------------------------------------------
# geometry / mask helpers
# ------------------------------------------------------------------------------

def iter_polygons(geom):
    if geom is None:
        return
    if hasattr(geom, "is_empty") and geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield geom
        return

    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
        return

    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_polygons(g)
        return


def rasterize_geometries_to_mask(
    geometries,
    image_shape: Tuple[int, int],
    label: int = 255,
) -> np.ndarray:
    """
    Rasterize shapely polygons into uint8 mask with target label value.
    image_shape: (H, W)
    """
    h, w = image_shape
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    for geom in geometries:
        for poly in iter_polygons(geom):
            ext = [(float(x), float(y)) for x, y in poly.exterior.coords]
            if len(ext) >= 3:
                draw.polygon(ext, fill=label)

            for interior in poly.interiors:
                hole = [(float(x), float(y)) for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    return np.array(canvas, dtype=np.uint8)


def build_threshold_mask(gray_patch: np.ndarray, label: int = 255) -> np.ndarray:
    mask_patch = ((gray_patch > 30) & (gray_patch < 220)).astype(np.uint8)
    return (mask_patch * label).astype(np.uint8)


def load_cellseg_dataframe(cellseg_path: Optional[str]) -> Optional[gpd.GeoDataFrame]:
    if not cellseg_path:
        return None
    if not os.path.exists(cellseg_path):
        raise FileNotFoundError(f"cellseg parquet not found: {cellseg_path}")

    gdf = gpd.read_parquet(cellseg_path)

    if "patch_idx" not in gdf.columns:
        raise ValueError("cellseg parquet must contain 'patch_idx'")
    if "geometry" not in gdf.columns:
        raise ValueError("cellseg parquet must contain 'geometry'")

    if "class_name" not in gdf.columns:
        if "class_id" in gdf.columns:
            gdf["class_name"] = gdf["class_id"].astype(str)
        else:
            gdf["class_name"] = "unknown"

    gdf["class_name"] = gdf["class_name"].fillna("unknown").astype(str)
    return gdf


def build_masks_from_cellseg(
    patch_cellseg: gpd.GeoDataFrame,
    image_shape: Tuple[int, int],
    label: int = 255,
    celltype_mode: str = "merged",   # merged | per_class | single
    target_cell_type: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Returns:
      {
        region_name: {
          "mask": np.ndarray,
          "n_cells": int,
          "class_name": str
        }
      }
    """
    if patch_cellseg is None or len(patch_cellseg) == 0:
        return {}

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg = patch_cellseg[patch_cellseg.geometry.notnull()]
    if len(patch_cellseg) == 0:
        return {}

    if celltype_mode == "merged":
        mask = rasterize_geometries_to_mask(
            patch_cellseg.geometry.tolist(),
            image_shape=image_shape,
            label=label,
        )
        return {
            "all_cells": {
                "mask": mask,
                "n_cells": int(len(patch_cellseg)),
                "class_name": "all_cells",
            }
        }

    if celltype_mode == "single":
        if not target_cell_type:
            raise ValueError("target_cell_type is required when celltype_mode='single'")

        sub = patch_cellseg[patch_cellseg["class_name"] == target_cell_type]
        if len(sub) == 0:
            return {}

        mask = rasterize_geometries_to_mask(
            sub.geometry.tolist(),
            image_shape=image_shape,
            label=label,
        )
        return {
            target_cell_type: {
                "mask": mask,
                "n_cells": int(len(sub)),
                "class_name": str(target_cell_type),
            }
        }

    if celltype_mode == "per_class":
        out = {}
        for class_name, sub in patch_cellseg.groupby("class_name"):
            if len(sub) == 0:
                continue

            mask = rasterize_geometries_to_mask(
                sub.geometry.tolist(),
                image_shape=image_shape,
                label=label,
            )
            out[str(class_name)] = {
                "mask": mask,
                "n_cells": int(len(sub)),
                "class_name": str(class_name),
            }
        return out

    raise ValueError(f"Unsupported celltype_mode: {celltype_mode}")


# ------------------------------------------------------------------------------
# save helpers
# ------------------------------------------------------------------------------

def save_patch_images_once(
    color_patch: np.ndarray,
    gray_patch: np.ndarray,
    output_dir: str,
    sample_id: str,
    base_filename: str,
):
    color_dir = get_patch_color_dir(output_dir, sample_id)
    gray_dir = get_patch_gray_dir(output_dir, sample_id)

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(gray_dir, exist_ok=True)

    color_path = f"{color_dir}/{base_filename}.png"
    gray_path = f"{gray_dir}/{base_filename}.png"

    if not os.path.exists(color_path):
        Image.fromarray(color_patch).save(color_path)
    if not os.path.exists(gray_path):
        Image.fromarray(gray_patch).save(gray_path)

    return color_path, gray_path


def save_region_mask_images(
    color_patch: np.ndarray,
    gray_patch: np.ndarray,
    mask_patch: np.ndarray,
    output_dir: str,
    sample_id: str,
    mask_filename: str,
):
    mask_dir = get_patch_mask_dir(output_dir, sample_id)
    masked_color_dir = get_patch_masked_color_dir(output_dir, sample_id)
    masked_gray_dir = get_patch_masked_gray_dir(output_dir, sample_id)

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_color_dir, exist_ok=True)
    os.makedirs(masked_gray_dir, exist_ok=True)

    mask_path = f"{mask_dir}/{mask_filename}.png"
    masked_color_path = f"{masked_color_dir}/{mask_filename}.png"
    masked_gray_path = f"{masked_gray_dir}/{mask_filename}.png"

    Image.fromarray(mask_patch).save(mask_path)

    mask_binary = (mask_patch > 0).astype(np.uint8)
    masked_color = color_patch * mask_binary[..., None]
    masked_gray = gray_patch * mask_binary

    Image.fromarray(masked_color.astype(np.uint8)).save(masked_color_path)
    Image.fromarray(masked_gray.astype(np.uint8)).save(masked_gray_path)

    return mask_path


# ------------------------------------------------------------------------------
# per-patch extraction
# ------------------------------------------------------------------------------

def process_single_patch(
    f,
    img_key,
    coords_key,
    barcodes_key,
    i,
    output_dir: str,
    sample_id: str,
    extractor,
    label=255,
    save_patches=True,
    mask_source: str = "threshold",   # threshold | cellseg
    cellseg_df: Optional[gpd.GeoDataFrame] = None,
    celltype_mode: str = "merged",    # merged | per_class | single
    target_cell_type: Optional[str] = None,
):
    img = f[img_key][i]

    if img.ndim == 3 and img.shape[2] == 3:
        color_patch = img.astype(np.uint8)
    elif img.ndim == 3 and img.shape[0] == 3:
        color_patch = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape} for patch index {i}")

    gray_patch = np.array(Image.fromarray(color_patch).convert("L"))

    coords = f[coords_key][i] if coords_key else None
    barcode = f[barcodes_key][i] if barcodes_key else None
    barcode = to_str_barcode(barcode) if barcode is not None else None

    base_filename = make_base_name(i, barcode)

    color_path = ""
    gray_path = ""

    if save_patches:
        color_path, gray_path = save_patch_images_once(
            color_patch=color_patch,
            gray_patch=gray_patch,
            output_dir=output_dir,
            sample_id=sample_id,
            base_filename=base_filename,
        )

    region_masks: Dict[str, Dict]

    if mask_source == "threshold":
        region_masks = {
            "threshold": {
                "mask": build_threshold_mask(gray_patch, label=label),
                "n_cells": None,
                "class_name": "threshold",
            }
        }

    elif mask_source == "cellseg":
        if cellseg_df is None:
            raise ValueError("mask_source='cellseg' requires cellseg_df")

        patch_cellseg = cellseg_df[cellseg_df["patch_idx"] == i]
        region_masks = build_masks_from_cellseg(
            patch_cellseg=patch_cellseg,
            image_shape=gray_patch.shape,
            label=label,
            celltype_mode=celltype_mode,
            target_cell_type=target_cell_type,
        )

        if not region_masks:
            return [{
                "patch_idx": i,
                "barcode": barcode,
                "color_path": color_path,
                "gray_path": gray_path,
                "mask_path": "",
                "region_type": None,
                "cell_type": None,
                "n_cells": 0,
                "mask_area": 0,
                "x": coords[0] if coords is not None else None,
                "y": coords[1] if coords is not None else None,
                "status": "skipped_no_cellseg_mask",
            }]
    else:
        raise ValueError(f"Unsupported mask_source: {mask_source}")

    rows = []

    for region_name, info in region_masks.items():
        mask_patch = info["mask"]
        class_name = info["class_name"]
        n_cells = info["n_cells"]

        mask_area = int(np.count_nonzero(mask_patch > 0))
        if mask_area < 50:
            rows.append(
                {
                    "patch_idx": i,
                    "barcode": barcode,
                    "color_path": color_path,
                    "gray_path": gray_path,
                    "mask_path": "",
                    "region_type": region_name,
                    "cell_type": class_name,
                    "n_cells": n_cells,
                    "mask_area": mask_area,
                    "x": coords[0] if coords is not None else None,
                    "y": coords[1] if coords is not None else None,
                    "status": "skipped_small_mask",
                }
            )
            continue

        mask_path = ""
        if save_patches:
            region_filename = f"{base_filename}__{region_name}"
            mask_path = save_region_mask_images(
                color_patch=color_patch,
                gray_patch=gray_patch,
                mask_patch=mask_patch,
                output_dir=output_dir,
                sample_id=sample_id,
                mask_filename=region_filename,
            )

        try:
            image_sitk = sitk.GetImageFromArray(gray_patch)
            mask_sitk = sitk.GetImageFromArray(mask_patch)

            features = extractor.execute(image_sitk, mask_sitk)

            row = {
                "patch_idx": i,
                "barcode": barcode,
                "color_path": color_path,
                "gray_path": gray_path,
                "mask_path": mask_path,
                "region_type": region_name,
                "cell_type": class_name,
                "n_cells": n_cells,
                "mask_area": mask_area,
                "x": coords[0] if coords is not None else None,
                "y": coords[1] if coords is not None else None,
                "status": "ok",
            }
            row.update(features)
            rows.append(row)

        except Exception as e:
            rows.append(
                {
                    "patch_idx": i,
                    "barcode": barcode,
                    "color_path": color_path,
                    "gray_path": gray_path,
                    "mask_path": mask_path,
                    "region_type": region_name,
                    "cell_type": class_name,
                    "n_cells": n_cells,
                    "mask_area": mask_area,
                    "x": coords[0] if coords is not None else None,
                    "y": coords[1] if coords is not None else None,
                    "status": f"error: {repr(e)}",
                }
            )

    return rows


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
    celltype_mode: str = "merged",
    target_cell_type: Optional[str] = None,
):
    rows = []

    extractor = build_radiomics_extractor(
        classes=classes,
        filters=filters,
        label=label,
        image_type_settings=image_type_settings,
    )

    cellseg_df = load_cellseg_dataframe(cellseg_path) if mask_source == "cellseg" else None

    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        for i in patch_indices:
            try:
                patch_rows = process_single_patch(
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
                    celltype_mode=celltype_mode,
                    target_cell_type=target_cell_type,
                )
                rows.extend(patch_rows)
            except Exception as e:
                rows.append(
                    {
                        "patch_idx": i,
                        "barcode": None,
                        "color_path": "",
                        "gray_path": "",
                        "mask_path": "",
                        "region_type": None,
                        "cell_type": None,
                        "n_cells": None,
                        "mask_area": None,
                        "x": None,
                        "y": None,
                        "status": f"error: {str(e)}",
                    }
                )

    return rows


def split_indices(indices, num_chunks):
    chunk_size = math.ceil(len(indices) / num_chunks)
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def extract_radiomics(
    h5_path,
    output_dir: str,
    sample_id: str,
    extractor=None,
    label=255,
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
    mask_source:
      - threshold
      - cellseg

    celltype_mode:
      - merged
      - per_class
      - single
    """
    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        total_num_patches = len(f[img_key])

    patch_indices = list(range(total_num_patches))

    if mask_source == "cellseg" and not cellseg_path:
        raise ValueError("cellseg_path is required when mask_source='cellseg'")

    # single-process
    if num_workers is None or num_workers <= 1:
        rows = []

        if extractor is None:
            extractor = build_radiomics_extractor(
                classes=classes,
                filters=filters,
                label=label,
                image_type_settings=image_type_settings,
            )

        cellseg_df = load_cellseg_dataframe(cellseg_path) if mask_source == "cellseg" else None

        with h5py.File(h5_path, "r") as f:
            img_key = get_img_key(f)
            coords_key = get_coords_key(f)
            barcodes_key = get_barcodes_key(f)

            for i in tqdm(patch_indices, desc="[Processing patches]"):
                try:
                    patch_rows = process_single_patch(
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
                        celltype_mode=celltype_mode,
                        target_cell_type=target_cell_type,
                    )
                    rows.extend(patch_rows)
                except Exception as e:
                    rows.append(
                        {
                            "patch_idx": i,
                            "barcode": None,
                            "color_path": "",
                            "gray_path": "",
                            "mask_path": "",
                            "region_type": None,
                            "cell_type": None,
                            "n_cells": None,
                            "mask_area": None,
                            "x": None,
                            "y": None,
                            "status": f"error: {str(e)}",
                        }
                    )

        return {
            "total_num_patches": total_num_patches,
            "rows": rows,
        }

    # multi-process
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
                celltype_mode,
                target_cell_type,
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

    rows.sort(key=lambda x: (x["patch_idx"], str(x.get("region_type"))))
    return {
        "total_num_patches": total_num_patches,
        "rows": rows,
    }


def get_radiomics_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_prefixes = (
        "original_",
        "wavelet-",
        "log-sigma-",
        "square_",
        "squareroot_",
        "logarithm_",
        "exponential_",
    )
    return [col for col in df.columns if col.lower().startswith(feature_prefixes)]


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
            f"No radiomics feature columns found to process. "
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
