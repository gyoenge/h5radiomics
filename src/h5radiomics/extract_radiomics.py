# extract_radiomics.py
# Radiomics feature extraction from HDF5 files containing image patches and associated metadata. 
"""
Example usage:

(i) Using YAML config file:
cd /root/workspace/h5radiomics/src 
python -m h5radiomics.extract_radiomics \
  --config ../configs/default.yaml \
  --num_workers 8

(ii) Using command-line arguments to override defaults or YAML config: 
python -m h5radiomics.extract_radiomics \
  --sample_ids TENX99 TENX95 NCBI785 NCBI783 \
  --h5_dir /root/workspace/h5radiomics/h5 \
  --output_root /root/workspace/h5radiomics/outputs \
  --label 255 \
  --save_patches \
  --classes firstorder glcm glrlm glszm gldm ngtdm \
  --filters Original \
  --num_workers 8 
""" 
# --filters Original Wavelet LoG Square SquareRoot Logarithm Exponential 

import os 
import argparse
import yaml
import h5py 
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk

import math
from concurrent.futures import ProcessPoolExecutor, as_completed

from radiomics import featureextractor
from PIL import Image
from tqdm import tqdm

from h5radiomics.utils import (
    get_img_key,
    get_coords_key,
    get_barcodes_key,
    to_str_barcode,
    make_base_name,
)

from typing import List, Tuple, Dict

# to avoid too much logging from pyradiomics 
import logging 
logging.getLogger("radiomics").setLevel(logging.ERROR)

# =========================

def get_default_config():
    return {
        "sample_ids": ["TENX95", "NCBI785", "NCBI783", "TENX99"],
        "h5_dir": "/root/workspace/hest_data/eval/bench_data/IDC/patches",
        "output_root": "/root/workspace/h5radiomics/outputs",
        "label": 255,
        "save_patches": False,
        "classes": ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
        "filters": ["Original"],
        "num_workers": 0,
        "image_type_settings": {
            "LoG": {
                "sigma": [1.0, 2.0, 3.0]
            }
        },
        "processing": {
            "lower_q": 0.01,
            "upper_q": 0.99,
            "save_processed": True,
        },
    }


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return data


def merge_config(defaults, yaml_config, cli_args):
    config = defaults.copy()

    # YAML merge (deep merge for dict)
    if yaml_config:
        for k, v in yaml_config.items():
            if v is None:
                continue

            if isinstance(v, dict) and isinstance(config.get(k), dict):
                # nested dict merge
                config[k].update(v)
            else:
                config[k] = v

    # CLI override
    for k, v in vars(cli_args).items():
        if k in ("config", "save_patches", "no_save_patches"):
            continue
        if v is not None:
            config[k] = v

    return config


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

    settings = {}
    settings["binWidth"] = 25
    settings["resampledPixelSpacing"] = None
    settings["verbose"] = False
    settings["label"] = label
    settings["force2D"] = True
    settings["force2Ddimension"] = 0
    settings["distances"] = [1]

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    if classes is not None:
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

def process_single_patch(
    f,
    img_key,
    coords_key,
    barcodes_key,
    i,
    output_root,
    extractor,
    label=255,
    save_patches=True,
):
    img = f[img_key][i]

    if img.ndim == 3 and img.shape[2] == 3:
        color_patch = img.astype(np.uint8)
    elif img.ndim == 3 and img.shape[0] == 3:
        color_patch = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape} for patch index {i}")

    gray_patch = np.array(Image.fromarray(color_patch).convert("L"))

    mask_patch = ((gray_patch > 30) & (gray_patch < 220)).astype(np.uint8) 
    mask_patch = (mask_patch * label).astype(np.uint8)

    coords = f[coords_key][i] if coords_key else None
    barcode = f[barcodes_key][i] if barcodes_key else None
    barcode = to_str_barcode(barcode) if barcode is not None else None

    base_filename = make_base_name(i, barcode)

    color_path = ""
    gray_path = ""
    mask_path = ""

    if save_patches:
        color_dir = os.path.join(output_root, "patches_color")
        gray_dir = os.path.join(output_root, "patches_gray")
        mask_dir = os.path.join(output_root, "patches_mask")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(gray_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        color_path = os.path.join(color_dir, f"{base_filename}.png")
        gray_path = os.path.join(gray_dir, f"{base_filename}.png")
        mask_path = os.path.join(mask_dir, f"{base_filename}.png")

        Image.fromarray(color_patch).save(color_path)
        Image.fromarray(gray_patch).save(gray_path)
        Image.fromarray(mask_patch).save(mask_path)

        # also save masked color&gray 
        masked_color_dir = os.path.join(output_root, "masked_color")
        masked_gray_dir = os.path.join(output_root, "masked_gray")
        os.makedirs(masked_color_dir, exist_ok=True)
        os.makedirs(masked_gray_dir, exist_ok=True)
        masked_color = color_patch * (mask_patch > 0)[..., None]
        masked_gray = gray_patch * (mask_patch > 0)
        mask_binary = (mask_patch > 0).astype(np.uint8)
        masked_color = color_patch * mask_binary[..., None]
        masked_gray = gray_patch * mask_binary
        masked_color_path = os.path.join(masked_color_dir, f"{base_filename}.png")
        masked_gray_path = os.path.join(masked_gray_dir, f"{base_filename}.png")
        Image.fromarray(masked_color.astype(np.uint8)).save(masked_color_path)
        Image.fromarray(masked_gray.astype(np.uint8)).save(masked_gray_path)


    if np.count_nonzero(mask_patch) < 50:
        return {
            "patch_idx": i,
            "barcode": barcode,
            "color_path": color_path,
            "gray_path": gray_path,
            "mask_path": "",
            "x": coords[0] if coords is not None else None,
            "y": coords[1] if coords is not None else None,
            "status": "skipped_small_mask",
        }

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
            "x": coords[0] if coords is not None else None,
            "y": coords[1] if coords is not None else None,
            "status": "ok",
        }
        row.update(features)
        return row

    except Exception as e:
        return {
            "patch_idx": i,
            "barcode": barcode,
            "color_path": color_path,
            "gray_path": gray_path,
            "mask_path": mask_path,
            "x": coords[0] if coords is not None else None,
            "y": coords[1] if coords is not None else None,
            "status": f"error: {str(e)}",
        }


def process_patch_chunk(
    h5_path,
    patch_indices,
    output_root,
    classes,
    filters,
    label,
    save_patches,
    image_type_settings=None,
):
    rows = []

    extractor = build_radiomics_extractor(
        classes=classes,
        filters=filters,
        label=label,
        image_type_settings=image_type_settings,
    )

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
                    output_root=output_root,
                    extractor=extractor,
                    label=label,
                    save_patches=save_patches,
                )
                rows.append(row)
            except Exception as e:
                rows.append({
                    "patch_idx": i,
                    "barcode": None,
                    "color_path": "",
                    "gray_path": "",
                    "mask_path": "",
                    "x": None,
                    "y": None,
                    "status": f"error: {str(e)}",
                })

    return rows


def split_indices(indices, num_chunks):
    chunk_size = math.ceil(len(indices) / num_chunks)
    return [
        indices[i:i + chunk_size]
        for i in range(0, len(indices), chunk_size)
    ]


def extract_radiomics(
    h5_path,
    output_root,
    extractor=None,
    label=255,
    save_patches=True,
    num_workers=0,
    classes=None,
    filters=None,
    image_type_settings=None,
):
    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        total_num_patches = len(f[img_key])

    patch_indices = list(range(total_num_patches))

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
                        output_root=output_root,
                        extractor=extractor,
                        label=label,
                        save_patches=save_patches,
                    )
                    rows.append(row)
                except Exception as e:
                    rows.append({
                        "patch_idx": i,
                        "barcode": None,
                        "color_path": "",
                        "gray_path": "",
                        "mask_path": "",
                        "x": None,
                        "y": None,
                        "status": f"error: {str(e)}",
                    })

        return {
            "total_num_patches": total_num_patches,
            "rows": rows,
        }

    # multi-process
    num_workers = min(num_workers, os.cpu_count() or 1)
    
    chunks = split_indices(patch_indices, num_workers * 64) # chunk를 더 잘게 쪼개면 progress bar가 더 자연스럽게 움직임
    rows = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk_size = {}

        for chunk in chunks:
            future = executor.submit(
                process_patch_chunk,
                h5_path,
                chunk,
                output_root,
                classes,
                filters,
                label,
                save_patches,
                image_type_settings,
            )
            future_to_chunk_size[future] = len(chunk)

        with tqdm(total=len(chunks), desc="[Processing chunks]", position=0) as chunk_pbar, \
            tqdm(total=total_num_patches, desc="[Processing patches]", position=1) as patch_pbar:

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


def get_radiomics_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Metadata/status/path/coord 등은 제외하고
    실제 radiomics feature 컬럼만 골라낸다.
    
    일반적으로 pyradiomics 결과 컬럼은
    - diagnostics_
    - original_
    - wavelet-
    - log-sigma-
    - square_
    - squareroot_
    - logarithm_
    - exponential_
    등으로 시작한다.
    
    여기서는 diagnostics_ 는 제외하고,
    실제 학습/분석용 feature만 선택한다.
    """
    feature_prefixes = (
        "original_",
        "wavelet-",
        "log-sigma-",
        "square_",
        "squareroot_",
        "logarithm_",
        "exponential_",
    )

    return [
        col for col in df.columns
        if col.startswith(feature_prefixes)
    ]


def clip_feature_series(
    s: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    1% / 99% quantile 기반 clipping.
    
    추가적으로 min/max가 극단적인 경우에도 동일한 clipping 결과가 자연스럽게 반영된다.
    (즉, lower_bound보다 작은 값은 lower_bound로,
         upper_bound보다 큰 값은 upper_bound로 자름)

    NaN은 유지한다.
    """
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
    """
    z = (x - mean) / std
    std == 0 이거나 NaN이면 0으로 채운다.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    mean = s_num.mean()
    std = s_num.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s_num)), index=s_num.index, dtype=float)

    return (s_num - mean) / std


def minmax_rescale_series(s: pd.Series) -> pd.Series:
    """
    [0, 1] rescaling
    max == min 이면 0으로 채운다.
    """
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
    """
    raw DataFrame에서 radiomics feature 컬럼만 골라
    (i) clipping
    (ii) z-normalization
    (iii) [0,1] rescaling
    수행한 processed DataFrame 생성.
    
    - metadata 컬럼은 그대로 유지
    - feature 컬럼은 processed 값으로 대체
    - status != ok 인 행은 feature를 NaN 유지
    - 각 feature별 통계(summary)도 함께 반환
    """
    processed_df = df.copy()
    feature_cols = get_radiomics_feature_columns(df)

    if not feature_cols:
        raise ValueError("No radiomics feature columns found to process.")

    stats_rows = []

    # 정상 추출된 row만 기준으로 후처리
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

        # 정상 행만 processed 값 채우기
        processed_df.loc[ok_mask, col] = scaled.astype(float)

        # 비정상 행은 NaN 유지
        processed_df.loc[~ok_mask, col] = np.nan

        stats_rows.append({
            "feature": col,
            "lower_q": lower_q,
            "upper_q": upper_q,
            **clip_stats,
            "z_mean": float(z_norm.mean()) if len(z_norm.dropna()) else np.nan,
            "z_std": float(z_norm.std(ddof=0)) if len(z_norm.dropna()) else np.nan,
            "scaled_min": float(scaled.min()) if len(scaled.dropna()) else np.nan,
            "scaled_max": float(scaled.max()) if len(scaled.dropna()) else np.nan,
        })

    stats_df = pd.DataFrame(stats_rows)
    return processed_df, stats_df


def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def convert_value(v):
        if pd.isna(v) if not isinstance(v, (list, tuple, np.ndarray, dict)) else False:
            return np.nan

        # numpy scalar -> python scalar
        if isinstance(v, np.generic):
            return v.item()

        # 0-d ndarray -> scalar
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return v.item()
            # 1-d 이상 배열은 parquet scalar column에 바로 못 넣으므로 문자열/JSON으로 변환
            return json.dumps(v.tolist(), ensure_ascii=False)

        # list/tuple/dict 도 문자열로 저장
        if isinstance(v, (list, tuple, dict)):
            return json.dumps(v, ensure_ascii=False)

        return v

    for col in df.columns:
        df[col] = df[col].map(convert_value)

    return df


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract radiomics features from HDF5 files.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--h5_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--label", type=int, default=None)

    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--filters", nargs="+", type=str, default=None)

    parser.add_argument("--save_patches", action="store_true")
    parser.add_argument("--no_save_patches", action="store_true")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for multiprocessing. 0 or 1 means single-process.",
    )

    return parser.parse_args(args)
    


def main(args=None): 
    # Parse command-line arguments and merge with defaults and YAML config if provided.
    cli_args = parse_args(args)

    defaults = get_default_config()
    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}

    config = merge_config(defaults, yaml_config, cli_args)

    if cli_args.save_patches:
        config["save_patches"] = True
    if cli_args.no_save_patches:
        config["save_patches"] = False
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Process each sample ID 
    for sample in config["sample_ids"]:
        h5_path = os.path.join(config["h5_dir"], f"{sample}.h5")
        feature_output_root = os.path.join(config["output_root"], "features")
        output_root = os.path.join(feature_output_root, f"{sample}_features")
        os.makedirs(output_root, exist_ok=True)
        print(f"Processing sample {sample} with HDF5 file: {h5_path}")
        print(f"Output will be saved to: {output_root}")

        if not os.path.exists(h5_path):
            print(f"[error] HDF5 file not found: {h5_path}")
            continue

        # Initialize pyradiomics feature extractor with default settings (can be customized as needed)
        extractor = build_radiomics_extractor(
            classes=config["classes"],
            filters=config["filters"],
            label=config["label"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        # Extract radiomics features 
        result = extract_radiomics(
            h5_path=h5_path,
            output_root=output_root,
            extractor=extractor if config["num_workers"] <= 1 else None,
            label=config["label"],
            save_patches=config["save_patches"],
            num_workers=config["num_workers"],
            classes=config["classes"],
            filters=config["filters"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        df = pd.DataFrame(result["rows"])

        # Save results to a CSV file
        csv_path = os.path.join(output_root, f"{sample}_radiomics_features.csv")
        df.to_csv(csv_path, index=False)

        # Parquet 저장
        df_parquet = make_parquet_safe(df)
        parquet_path = os.path.join(output_root, f"{sample}_radiomics_features.parquet")
        df_parquet.to_parquet(parquet_path, index=False)

        print(f"Finished radiomcis extraction of sample {sample}. \
                Total patches: {result['total_num_patches']}.")   
        
        print(f"Saved raw CSV: {csv_path}")
        print(f"Saved raw Parquet: {parquet_path}")

        # Build processed features
        processing_cfg = config.get("processing", {})

        lower_q = processing_cfg.get("lower_q", 0.01)
        upper_q = processing_cfg.get("upper_q", 0.99)
        save_processed = processing_cfg.get("save_processed", True)

        if save_processed:
            processed_df, processed_stats_df = build_processed_feature_df(
                df,
                status_col="status",
                ok_status="ok",
                lower_q=lower_q,
                upper_q=upper_q,
            )

            processed_csv_path = os.path.join(
                output_root, f"{sample}_radiomics_features_processed.csv"
            )
            processed_df.to_csv(processed_csv_path, index=False)

            processed_parquet = make_parquet_safe(processed_df)
            processed_parquet_path = os.path.join(
                output_root, f"{sample}_radiomics_features_processed.parquet"
            )
            processed_parquet.to_parquet(processed_parquet_path, index=False)

            processed_stats_csv_path = os.path.join(
                output_root, f"{sample}_radiomics_features_processed_stats.csv"
            )
            processed_stats_df.to_csv(processed_stats_csv_path, index=False)

            print(f"Saved processed CSV: {processed_csv_path}")
            print(f"Saved processed Parquet: {processed_parquet_path}")
            print(f"Saved processed stats CSV: {processed_stats_csv_path}")
            print(f"Processed config: lower_q={lower_q}, upper_q={upper_q}")


# =========================

if __name__ == "__main__": 
    main() 

