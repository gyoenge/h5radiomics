# HEST Radiomics Documentation

Welcome to the documentation for **HEST-Radiomics**, a radiomics extraction pipeline for spatial transcriptomics and whole-slide imaging (WSI) analysis.

This documentation describes the basic usage of the HEST radiomics pipeline.

The project provides:

- HEST dataset handling utilities
- Patch-level radiomics extraction
- Cell-aware segmentation processing
- Spatial transcriptomics integration
- HDF5 / H5AD-based workflows
- Multi-processing extraction pipelines


---

## Documentation Contents

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
dataset
radiomics 
```

```{toctree}
:maxdepth: 1
:caption: Output Description

outputs/overview.md
outputs/segment 
outputs/radiomics.md
```

<!-- 
```{toctree}
:maxdepth: 1
:caption: API Reference

api/extractors
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

github
``` -->

<!-- ---

## Project Structure

```text
hestradiomics/
├── extractors/
├── segment/
├── radiomics/
├── datasets/
├── utils/
└── io/
``` -->

---

## Main Features

### Patch-Level Radiomics

Extract handcrafted radiomics features from spatial transcriptomics patches using PyRadiomics.

### Cell-Aware Segmentation

Polygon-based cell segmentation storage with rasterized mask generation during runtime.

### HEST Dataset Integration

Built-in utilities for loading and processing HEST spatial transcriptomics datasets.

### Spatial Transcriptomics Support

Compatible with H5AD-based transcriptomics workflows and Visium-style datasets.

### Efficient Storage

Compact polygon-based HDF5 segmentation representation for large-scale WSI processing.


