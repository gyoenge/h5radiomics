# h5radiomics

Radiomics extraction pipeline from H5-based WSI (Whole Slide Image)
patches.

------------------------------------------------------------------------

## Overview

This project provides a complete pipeline for:

-   Radiomics feature extraction from HDF5-stored WSI patches
-   Feature statistics analysis and visualization
-   Cell segmentation using CellViT
-   Efficient multi-processing for large-scale patch-level processing

------------------------------------------------------------------------

## Project Structure

    h5radiomics/
    ├── src/
    │   └── h5radiomics/
    │       ├── extract_radiomics.py
    │       ├── feature_statistics.py
    │       ├── segment_cellvit.py
    │       └── utils.py
    ├── configs/
    ├── h5/
    ├── outputs/
    ├── models/
    └── README.md


## Input Data (H5)

Place your HDF5 patch files in the `h5/` directory.

Each `.h5` file should contain: 

- Image patches (key: `img` / `imgs` / `images`)
- Optional:
  - `coords` (patch coordinates)
  - `barcode` / `barcodes`

### Example

    h5/
    ├── NCBI783.h5
    ├── NCBI785.h5
    ├── TENX95.h5
    └── TENX99.h5

### Expected H5 structure

    {
    "img": (N, H, W, 3) or (N, 3, H, W),
    "coords": (N, 2), # optional
    "barcodes": (N,) # optional
    }

### Notes

- Images must be RGB patches
- Patch size should be consistent (e.g., 224x224)
- Masks are automatically generated from grayscale thresholding

------------------------------------------------------------------------

## Installation

    pip install -r requirements.txt

or

    pip install -e .

------------------------------------------------------------------------

## Usage

### Radiomics Extraction

    python -m h5radiomics.extract_radiomics --config configs/default.yaml

### Feature Statistics

    python -m h5radiomics.feature_statistics --config configs/stats.yaml

### Cell Segmentation

    python -m h5radiomics.segment_cellvit --config configs/segment_cellvit.yaml

------------------------------------------------------------------------

## Output

-   CSV / Parquet feature files
-   Feature statistics + boxplots
-   Cell segmentation (GeoJSON / Parquet / overlay)

------------------------------------------------------------------------

## Features

-   PyRadiomics 기반 feature extraction
-   Multi-processing 최적화
-   Parquet 저장 지원
-   Feature normalization
-   Representative patch selection
-   CellViT segmentation

------------------------------------------------------------------------
