# HEST-Radiomics (h5radiomics)

Radiomics extraction and analysis pipeline for HDF5-based WSI (Whole Slide Image) patches.

---

## Overview

This project provides an end-to-end pipeline for patch-level computational pathology:

- Radiomics feature extraction from HDF5-based WSI patches
  
  : extracts **Intensity, Texture, Cell-shape, Cell-type** features

- Feature preprocessing and normalization
- Feature statistics analysis and visualization
- Representative patch selection
- Cell segmentation using CellViT
- Scalable multi-processing for large datasets

The pipeline is designed with a **consistent sample-wise directory structure** to ensure reproducibility and modular processing.

---

## Input Data (H5)

Place your HDF5 patch files in:

    data/h5/


Each `.h5` file should contain:

- Image patches  
  - key: `img` / `imgs` / `images`
- Optional:
  - `coords` (patch coordinates)
  - `barcode` / `barcodes`

### Example

    data/h5/
    ├── NCBI783.h5
    ├── NCBI785.h5
    ├── TENX95.h5
    └── TENX99.h5

### Expected H5 Structure

    {
    "img": (N, H, W, 3) or (N, 3, H, W),
    "coords": (N, 2), # optional
    "barcodes": (N,) # optional
    }


### Notes

- Images must be RGB patches
- Patch size should be consistent (e.g., 224×224)
- Masks are automatically generated via grayscale thresholding

---

## Installation

Firstly, create and activate conda environment: 
```bash
conda create -n h5radiomics python=3.10  # cellvit requires >=3.10
conda activate h5radiomics 
```

Install `torch` and `torchvision`, matching with your cuda environment: 
```bash
# example 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0.0,>=1.24" # for numpy degrading 
```

Then: 
```bash
# inside the hest-radiomics/ directory 
pip install -e . --no-build-isolation
```

We need `cellvit` module installation for `segment` engine: 
```bash
conda install -c conda-forge openslide
pip install openslide-python
pip install cellvit
```

---

## Usage

### Run Full Pipeline (Recommended)

Run the entire pipeline (segment → extract → statistics):

```bash 
python -m h5radiomics.run \
  --config configs/fast_full_extract.yaml
```

Optional: 

```bash
# NOTE: segment is requirement of our extraction
# run only extraction (needs segment output files)
--skip_segment --skip_statistics

# run without statistics 
--skip_statistics
```

---

## Output Directory Structure

    All outputs are organized per sample:

    data/outputs/
    └── {sample_id}/
        ├── patches/   # only for save_patches: true
        │   ├── color/
        │   ├── gray/
        │   ├── mask/
        │   ├── masked_color/
        │   └── masked_gray/
        │
        ├── features/
        │   ├── raw/
        │   │   ├── features.csv
        │   │   └── features.parquet
        │   │
        │   ├── processed/
        │   │   ├── features.csv
        │   │   ├── features.parquet
        │   │   ├── processing_stats.csv
        │   │   └── processing_config.json
        │   │
        │   └── statistics/  
        │       ├── raw/
        │       │   ├── stats.csv
        │       │   ├── stats.parquet
        │       │   ├── representative/  # only for save_representatives: true 
        │       │   └── boxplots/        # only for save_boxplot: true
        │       │
        │       └── processed/
        │           ├── stats.csv
        │           ├── stats.parquet
        │           ├── representative/  # only for save_representatives: true 
        │           └── boxplots/        # only for save_boxplot: true
        │
        └── cellvitseg/
            ├── cellseg.geojson
            ├── cellseg.parquet
            ├── metadata.csv
            ├── summary.json
            └── overlay/        # only for save_png_overlay: true

