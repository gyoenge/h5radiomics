# Run Step-by-step

This page describes how to run the HEST-Radiomics pipeline step by step.

The full pipeline can be executed with a single command, but running each step separately is useful for debugging, customization, and large-scale processing.

---

## Typical Workflow

The general pipeline workflow is:

```text
WSI
  ↓
Patch Extraction
  ↓
Segmentation
  ↓
Radiomics Extraction
  ↓
Spatial Transcriptomics Integration
  ↓
Visualization / Statistics Analysis
```

Generated outputs are progressively stored under the dataset-specific directory.

Example:

```text
data/
└── hest/
    └── IDC/
        ├── patches/
        ├── st/
        ├── segment/
        ├── segment_vis/
        ├── radiomics/
        └── radiomics_stats/
```

---

## Steps

### 1. Download HEST Base Dataset

Download the required HEST files for the selected cancer types.

This step prepares the base dataset, including patch files, spatial transcriptomics files, metadata, thumbnails, and optional visualization files.

Example output directories:

```text
data/hest/<ONCOTREE>/
├── patches/
├── st/
├── metadata/
├── thumbnails/
├── patches_vis/
└── spatial_plots/
```

Example command:

```bash
python -m hestradiomics.download
```

After this step, check that the patch and spatial transcriptomics files exist:

```text
data/hest/IDC/patches/
data/hest/IDC/st/
```

```{note}
The exact downloaded directories may depend on the dataset configuration.
```

---

### 2. Run Cell Segmentation before Extraction

Run cell segmentation on the extracted patch images.

This step generates cell-level segmentation outputs for each WSI sample.  
The segmentation results are stored as polygon-based HDF5 files.

Example output:

```text
data/hest/<ONCOTREE>/
├── segment/
│   ├── NCBI681.h5
│   ├── NCBI682.h5
│   └── ...
└── segment_vis/
    ├── NCBI681/
    ├── NCBI682/
    └── ...
```

Example command:

```bash
python -m hestradiomics.segment
```

The segmentation files contain:

- patch-level metadata
- cell-level metadata
- polygon-based cell boundaries
- cell type annotations
- patch-to-cell mappings

```{important}
Radiomics extraction can be performed without segmentation for patch-level features, but cell-aware radiomics features require valid segmentation outputs.
```

See also:

- {doc}`Segment Output Format <outputs/segment>`

---

### 3. Run Feature Extraction

Extract radiomics features from image patches.

This step computes handcrafted radiomics descriptors using patch images and, optionally, segmentation-derived masks.

Example output:

```text
data/hest/<ONCOTREE>/
└── radiomics/
    ├── NCBI681.h5ad
    ├── NCBI682.h5ad
    └── ...
```

Example command:

```bash
python -m hestradiomics.extract
```

The extracted features may include:

- first-order intensity features
- texture features
- shape features
- cell-aware radiomics features
- patch-level metadata

```{note}
Shape features require a valid binary mask. If PyRadiomics raises a label error, check that the mask foreground value matches the configured label.
```

See also:

- {doc}`Radiomics Output Format <outputs/radiomics>`

---

### 4. Run Result Visualization / Statistics Analysis

Generate visualization and statistics outputs for quality control and downstream analysis.

This step summarizes radiomics feature distributions and produces diagnostic plots.

Example output:

```text
data/hest/<ONCOTREE>/
└── radiomics_stats/
    ├── feature_summary.csv
    ├── feature_variance.csv
    ├── missing_value_summary.csv
    └── ...
```

Example command:

```bash
python -m hestradiomics.stats
```

Typical analyses include:

- missing value summary
- feature variance analysis
- feature distribution plots
- normalization diagnostics
- sample-level statistics
- radiomics feature quality control

---

## Full Pipeline Command

Alternatively, run the complete pipeline with:

```bash
python -m hestradiomics.run
```

This command executes the major pipeline steps automatically.

```{important}
For large datasets, step-by-step execution is recommended because segmentation and radiomics extraction can be computationally expensive.
```

---

## Recommended Execution Order

```text
1. Download HEST base dataset
2. Run cell segmentation
3. Run radiomics feature extraction
4. Run visualization/statistics analysis
```

---

## Troubleshooting

### Missing segmentation files

If cell-aware extraction fails, check whether segmentation files exist:

```text
data/hest/<ONCOTREE>/segment/
```

### PyRadiomics label error

If you encounter:

```text
ValueError: Label (1) not present in mask. Choose from [255]
```

convert the mask to a binary 0/1 mask:

```python
mask = (mask > 0).astype("uint8")
```

or configure PyRadiomics to use the correct label value.

### Missing output files

Check that the previous step completed successfully before running the next step.

For example, radiomics extraction requires patch files:

```text
data/hest/<ONCOTREE>/patches/
```

and cell-aware extraction additionally requires:

```text
data/hest/<ONCOTREE>/segment/
```

---

## See Also

- {doc}`Output Overview <outputs/overview>`
- {doc}`Segment Output Format <outputs/segment>`
- {doc}`Radiomics Output Format <outputs/radiomics>`
