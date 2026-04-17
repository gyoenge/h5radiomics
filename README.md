# HEST-Radiomics (h5radiomics)

Radiomics extraction and analysis pipeline for HDF5-based WSI (Whole Slide Image) patches.

---

## Overview

This project provides an end-to-end pipeline for patch-level computational pathology:

- Radiomics feature extraction from HDF5-based WSI patches
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

```bash
pip install -r requirements.txt
# or 
pip install -e .
```

---

## Usage

### 1. Radiomics Feature Extraction

```bash
python -m h5radiomics.pipelines.run_extract \
  --config configs/extract.yaml \
  --num_workers 8
```

### 2. Feature Statistics

```bash 
python -m h5radiomics.pipelines.run_statistics \
  --config configs/statistics.yaml
```

### 3. Cell Segmentation (CellViT)

```bash
python -m h5radiomics.pipelines.run_segment \
  --config configs/segment.yaml
```

---

## Output Directory Structure

    All outputs are organized per sample:

    data/outputs/
    └── {sample_id}/
        ├── patches/
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
        │       │   ├── representative/
        │       │   └── boxplots/
        │       │
        │       └── processed/
        │           ├── stats.csv
        │           ├── stats.parquet
        │           ├── representative/
        │           └── boxplots/
        │
        └── cellvitseg/
            ├── cellseg.geojson
            ├── cellseg.parquet
            ├── metadata.csv
            ├── summary.json
            └── overlay/

---

## Key Features

- PyRadiomics-based feature extraction
- Patch-level processing from HDF5
- Multi-processing for scalability
- Structured feature preprocessing (clipping + normalization)
- Raw vs processed feature separation
- Detailed feature statistics and visualization
- Representative patch selection (quantile-based)
- Cell segmentation with CellViT
- Standardized and reproducible output structure

### Design Philosophy

This pipeline is designed with:

- Modularity: Separate engines for extract / statistics / segmentation
- Reproducibility: All processing steps are saved with configs and metadata
- Scalability: Multi-process support for large-scale WSI datasets
- Consistency: Unified directory structure across all samples

### Future Extensions

- Feature embedding (TransTab / PCA / UMAP)
- Clustering and phenotype discovery
- Feature selection and importance analysis
- Integration with downstream ML models

