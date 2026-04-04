# extract_radiomics.py
# Radiomics feature extraction from HDF5 files containing image patches and associated metadata. 
"""
Example usage:

(i) Using YAML config file:
cd /root/workspace/h5radiomics/src 
python -m h5radiomics.extract_radiomics --config ../configs/default.yaml

(ii) Using command-line arguments to override defaults or YAML config: 
python -m h5radiomics.extract_radiomics \
  --sample_ids TENX95 NCBI785 NCBI783 \
  --h5_dir /root/workspace/hest_data/eval/bench_data/IDC/patches \
  --output_root /root/workspace/impl/h5radiomics/output_test \
  --label 255 \
  --save_patches \
  --classes firstorder glcm glrlm glszm gldm ngtdm \
  --filters Original Wavelet LoG Square SquareRoot Logarithm Exponential
""" 

import os 
import argparse
import yaml
import h5py 
import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor
from PIL import Image

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
        "output_root": "/root/workspace/impl/h5radiomics/output",
        "label": 255,
        "save_patches": False,
        "classes": ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
        "filters": ["Original"],
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


def extract_radiomics(
        h5_path, 
        output_root, 
        extractor, 
        label=255,
        save_patches=True, # Whether to save extracted patches as images. 
    ): 
    rows = []  # To store metadata and radiomics features for each patch

    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        total_num_patches = len(f[img_key])
        
        for i in range(total_num_patches):
            # Extract patch data 
            img = f[img_key][i]  

            if img.ndim == 3 and img.shape[2] == 3:
                # Already in (H, W, C) format 
                color_patch = img.astype(np.uint8)
            elif img.ndim == 3 and img.shape[0] == 3:
                # Transpose from (C, H, W) to (H, W, C)
                color_patch = np.transpose(img, (1, 2, 0)).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape} for patch index {i}")

            # Create grayscale patches
            gray_patch = np.array(Image.fromarray(color_patch).convert("L"))

            # Create mask patches
            # mask_patch = np.where(gray_patch > 0, 255, 0).astype(np.uint8)
            mask_patch = (gray_patch < 220).astype(np.uint8)  # return binary mask (0 or 1) instead of 0 or 255 
            mask_patch = (mask_patch * label).astype(np.uint8)  # Scale binary mask to the specified label value (e.g., 255)

            # Extract associated metadata if available
            coords = f[coords_key][i] if coords_key else None
            barcode = f[barcodes_key][i] if barcodes_key else None
            barcode = to_str_barcode(barcode) if barcode is not None else None

            # Make base filename for saving patches 
            base_filename = make_base_name(i, barcode)

            # Save patches as images 
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

            # Skip when mask is too small (e.g., less than 10% foreground) to avoid meaningless radiomics features
            if np.count_nonzero(mask_patch) < 50:
                print(f"[skip] patch {i}: foreground too small")
                row = {
                    "patch_idx": i,
                    "barcode": barcode,
                    "color_path": color_path,
                    "gray_path": gray_path,
                    "mask_path": "",
                    "x": coords[0] if coords is not None else None,
                    "y": coords[1] if coords is not None else None,
                    "status": "skipped_small_mask",
                }
                rows.append(row)
                continue

            # Extract radiomics features using pyradiomics
            try: 
                image_sitk = sitk.GetImageFromArray(gray_patch)
                mask_sitk = sitk.GetImageFromArray(mask_patch)

                features = extractor.execute(image_sitk, mask_sitk)
                # features_original = {k: v for k, v in features.items() if k.startswith("original")}  # Keep only original features

                row = {
                    "patch_idx": i,
                    "barcode": barcode,
                    "color_path": color_path,
                    "gray_path": gray_path,
                    "mask_path": mask_path,
                    "x": coords[0] if coords is not None else None,
                    "y": coords[1] if coords is not None else None,
                    "status": "ok",
                    # **features_original,  # Add radiomics features to the row
                }
                row.update(features)  # Add all features (including non-original) to the row
                rows.append(row)
            
            except Exception as e:
                print(f"[error] patch {i}: {e}")
                row = {
                    "patch_idx": i,
                    "barcode": barcode,
                    "color_path": color_path,
                    "gray_path": gray_path,
                    "mask_path": mask_path,
                    "x": coords[0] if coords is not None else None,
                    "y": coords[1] if coords is not None else None,
                    "status": f"error: {str(e)}",
                }
                rows.append(row)

        return {
            "total_num_patches": total_num_patches, 
            "rows": rows, 
        }


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
        output_root = os.path.join(config["output_root"], f"{sample}_outputs")
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
            extractor=extractor, 
            label=config["label"], 
            save_patches=config["save_patches"], 
        )

        # Save results to a CSV file
        df = pd.DataFrame(result["rows"])
        csv_path = os.path.join(output_root, f"{sample}_radiomics_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"Finished processing sample {sample}. \
                Total patches: {result['total_num_patches']}. \
                Results saved to {csv_path}.")   


# =========================

if __name__ == "__main__": 
    main() 

