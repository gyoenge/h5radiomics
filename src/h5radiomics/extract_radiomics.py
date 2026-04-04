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

    if yaml_config:
        for k, v in yaml_config.items():
            if v is not None:
                config[k] = v

    for k, v in vars(cli_args).items():
        if k in ("config", "save_patches", "no_save_patches"):
            continue
        if v is not None:
            config[k] = v

    return config


def build_radiomics_extractor(classes=None, filters=None, label=255):
    if classes is None:
        classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    if filters is None:
        filters = ["Original"]

    # Build and configure the pyradiomics feature extractor based on specified feature classes and filters.
    settings = {}

    settings['binWidth'] = 25 # Default bin width for intensity discretization 
    # small binWidth (e.g., 5) captures more detail but can be sensitive to noise, 
    # while large binWidth (e.g., 50) provides smoother features but may lose important information. 
    # A value of 25 is often a good starting point for stable features, but it may need to be adjusted based on the specific characteristics of the data.
    
    settings['resampledPixelSpacing'] = None # No resampling, use original pixel spacing
    # settings['interpolator'] = sitk.sitkBSpline # Interpolation method for resampling (if resampling is used)
    
    settings['verbose'] = False # Disable verbose (logging) output from pyradiomics 
    
    settings['label'] = label # Label value in the mask to consider as foreground for feature extraction (since we use binary mask with 0 and 1, set label to 1 or 255 depending on how mask is saved)
    
    settings['force2D'] = True # Force 2D feature extraction (since our patches are 2D)
    settings['force2Ddimension'] = 0 # Dimension to use for 2D feature extraction (0 for XY plane) 
    # When working with 2D image patches, it's important to set force2D=True to ensure that pyradiomics calculates features based on 2D analysis rather than treating the patch as a thin 3D volume. 
    # This prevents unintended consequences where texture and shape features might be calculated in a 3D context, which can lead to meaningless or unstable features when the image is essentially 2D. 
    
    settings['distances'] = [1] # Default distance for texture feature calculation (can be customized as needed)

    # Other settings can be added here as needed, such as normalization, outlier removal, or specific parameters for certain features. 
    # settings['normalize'] = True
    # settings['normalizeScale'] = 100 # Default normalization settings (can be customized as needed)
    # settings['removeOutliers'] = 3 # Default outlier removal settings (can be customized as needed)
    # settings['voxelArrayShift'] = 0 # Default voxel array shift (can be customized as needed)
    # settings['minimumROIDimensions'] = None # No minimum dimension requirement for the region of interest (ROI)
    # settings['minimumROISize'] = None # No minimum size requirement for the ROI (
    # resegmentRange = None # No resegmentation based on intensity range
    # resegmentMode = None # No resegmentation based on mode

    # Initialize pyradiomics feature extractor with settings 
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    # Enable specified feature classes
    if classes is not None:
        for cls in classes:
            extractor.enableFeatureClassByName(cls)

    # Enable specified filters if provided
    image_types = set(filters or [])
    image_types.add("Original")  # 항상 포함
    for filt in image_types:
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

    mask_patch = (gray_patch < 220).astype(np.uint8)
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
):
    rows = []

    extractor = build_radiomics_extractor(
        classes=classes,
        filters=filters,
        label=label,
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
        )

        df = pd.DataFrame(result["rows"])

        # Save results to a CSV file
        csv_path = os.path.join(output_root, f"{sample}_radiomics_features.csv")
        df.to_csv(csv_path, index=False)

        # Parquet 저장
        df_parquet = make_parquet_safe(df)
        parquet_path = os.path.join(output_root, f"{sample}_radiomics_features.parquet")
        df_parquet.to_parquet(parquet_path, index=False)

        print(f"Finished processing sample {sample}. \
                Total patches: {result['total_num_patches']}.")   
        
        print(f"Saved CSV: {csv_path}")
        print(f"Saved Parquet: {parquet_path}")


# =========================

if __name__ == "__main__": 
    main() 

