# Radiomics Extractors

The `extractors/` module provides modular feature extraction pipelines for histopathology patches and cell segmentation outputs.

It supports:

- Patch-level radiomics
- Cell segmentation-based radiomics
- Cell morphology aggregation
- Cell-type distribution statistics
- Feature post-processing and normalization

The pipeline is designed for scalable WSI patch processing with multiprocessing-friendly extractor caching.

---

## Module Structure

```text
extractors/
├── builders.py
├── cell_distribution.py
├── constants.py
├── intensity_texture.py
├── patch_processor.py
├── postprocess.py
├── shape.py
└── README.md
```

---

## Feature Types

### 1. Patch-level Radiomics

Extract radiomics features from threshold-based patch masks.

Features include:

* First-order statistics
* GLCM
* GLRLM
* GLSZM
* GLDM
* NGTDM

Example prefix:

```text
patch_original_firstorder_mean
```


### 2. CellSeg-based Radiomics

Extract radiomics features using merged cell segmentation masks.

Example prefix:

```text
cellseg_all_original_glcm_contrast
```


### 3. Morphology Features

Extract per-cell `shape2D` features and aggregate them into patch-level statistics.

Per-cell features:

* Area
* Perimeter
* Elongation
* MajorAxisLength
* Solidity
* etc.

Aggregated statistics:

* mean
* median
* variance
* entropy
* skewness
* kurtosis
* percentile statistics

Example prefix:

```text
morph_area_mean
morph_perimeter_entropy
```


## 4. Cell-Type Distribution

Compute cell composition statistics from segmentation labels.

Supported cell classes:

* neoplastic
* inflammatory
* connective
* dead
* epithelial

Example features:

```text
dist_count_neoplastic
dist_ratio_epithelial
dist_cell_count_total
```

---


## Main Components

### builders.py

PyRadiomics extractor builders with worker-local caching.

Provides:

* Texture/intensity extractor
* Shape2D extractor
* Multiprocessing-safe extractor reuse

Key functions:

```python
build_radiomics_extractor()
build_shape2d_extractor()
```


### intensity_texture.py

Radiomics execution helpers and patch-level extraction.

Provides:

* Threshold-mask radiomics
* Cell segmentation radiomics
* Result cleaning/filtering
* Prefix handling


### shape.py

Cell morphology extraction and aggregation.

Pipeline:

```text
polygon → local mask → shape2D extraction → patch-level aggregation
```

Uses manual first-order aggregation for numerical stability.


### cell_distribution.py

Cell-type count and ratio statistics.

Normalizes upstream segmentation class labels into a unified schema.


### patch_processor.py

Main patch-level processing pipeline.

Supports:

* Threshold-based extraction
* CellSeg-based extraction
* Mask visualization saving
* Feature aggregation
* Error-safe execution

Main entrypoint:

```python
process_single_patch()
```


### postprocess.py

Feature normalization and preprocessing utilities.

Includes:

* Quantile clipping
* Z-normalization
* Min-max scaling
* Feature column filtering
* Statistics logging

Processing pipeline:

```text
raw feature
→ quantile clipping
→ z-normalization
→ min-max scaling
```

---

## Design Goals

* Modular feature extraction
* Scalable multiprocessing support
* Stable morphology aggregation
* Consistent feature naming
* Downstream ML-ready outputs
* Patch-level and cell-level integration

---

## Example Workflow

```text
WSI patch
    ↓
threshold mask / cell segmentation mask
    ↓
radiomics extraction
    ↓
cell morphology extraction
    ↓
distribution statistics
    ↓
feature aggregation
    ↓
post-processing normalization
```

---

## Notes

* PyRadiomics is used for texture/intensity and shape2D extraction.
* Morphology aggregation is manually implemented for robustness on small cell populations.
* Worker-local extractor caching avoids repeated PyRadiomics initialization overhead.
* Feature naming conventions are centralized in `constants.py`. 

