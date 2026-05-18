# Radiomics Output Format 

Radiomics outputs store handcrafted image features extracted from spatial transcriptomics patches.

The HEST-Radiomics pipeline extracts radiomics features from patch images and, when segmentation outputs are available, can also support cell-aware feature extraction using segmentation-derived masks.

---

## Output Directory

Radiomics output files are stored under:

```text
data/
└── hest/
    └── <ONCOTREE>/
        └── radiomics/
            ├── NCBI681.h5ad
            ├── NCBI682.h5ad
            ├── NCBI683.h5ad
            └── ...
```

Each output file corresponds to a single WSI sample.

Example:

```text
data/hest/IDC/radiomics/NCBI681.h5ad
```

---

## File Format

Radiomics outputs are stored in AnnData-compatible H5AD format.

```text
<sample_id>.h5ad
```

The H5AD format is used because it can store:

- feature matrices
- patch-level metadata
- spatial transcriptomics identifiers
- preprocessing metadata
- downstream analysis annotations

---

## AnnData Structure

Each radiomics file is represented as an `AnnData` object.

```python
import scanpy as sc

adata = sc.read_h5ad("data/hest/IDC/radiomics/NCBI681.h5ad")

print(adata)
```

Typical structure:

```text
AnnData object with n_obs × n_vars
    obs: patch-level metadata
    var: radiomics feature metadata
    X: radiomics feature matrix
```

---

## `adata.X`

`adata.X` contains the radiomics feature matrix.

```text
shape: (N_patches, N_features)
```

- rows correspond to spatial transcriptomics patches
- columns correspond to extracted radiomics features

Example:

```python
X = adata.X
print(X.shape)
```

---

## `adata.obs`

`adata.obs` stores patch-level metadata.

Typical fields may include:

| Column | Description |
|---|---|
| `barcode` | Spatial transcriptomics barcode |
| `patch_idx` | Integer patch index |
| `coord_x` | Patch x-coordinate |
| `coord_y` | Patch y-coordinate |
| `sample_id` | Sample identifier |
| `oncotree` | Cancer type or dataset group |

Example:

```python
adata.obs.head()
```

To display all metadata columns:

```python
print(adata.obs.columns)
```

---

## `adata.var`

`adata.var` stores radiomics feature metadata.

Each row corresponds to one radiomics feature.

Typical fields may include:

| Column | Description |
|---|---|
| `feature_name` | Radiomics feature name |
| `feature_class` | Feature class, such as `firstorder`, `glcm`, or `shape` |
| `image_type` | Image type, such as `Original` |
| `source` | Feature extraction source |

Example:

```python
adata.var.head()
```

---

## Feature Naming Convention

Radiomics feature names typically follow the PyRadiomics naming style:

```text
<image_type>_<feature_class>_<feature_name>
```

Example:

```text
original_firstorder_Mean
original_firstorder_Entropy
original_glcm_Contrast
original_glrlm_RunLengthNonUniformity
original_gldm_DependenceEntropy
```

Common feature classes include:

| Feature Class | Description |
|---|---|
| `firstorder` | Intensity distribution statistics |
| `glcm` | Gray Level Co-occurrence Matrix texture features |
| `glrlm` | Gray Level Run Length Matrix texture features |
| `glszm` | Gray Level Size Zone Matrix texture features |
| `gldm` | Gray Level Dependence Matrix texture features |
| `ngtdm` | Neighboring Gray Tone Difference Matrix texture features |
| `shape` | Shape and morphology features, when masks are available |

---

## Loading Radiomics Features

```python
import scanpy as sc

adata = sc.read_h5ad("data/hest/IDC/radiomics/NCBI681.h5ad")

X = adata.X
obs = adata.obs
var = adata.var
```

Convert the feature matrix to a pandas DataFrame:

```python
import pandas as pd

features = pd.DataFrame(
    adata.X,
    index=adata.obs_names,
    columns=adata.var_names,
)

features.head()
```

---

## Linking with Spatial Transcriptomics Data

Radiomics outputs can be linked with spatial transcriptomics data using shared spot barcodes.

Example:

```python
rad = sc.read_h5ad("data/hest/IDC/radiomics/NCBI681.h5ad")
st = sc.read_h5ad("data/hest/IDC/st/NCBI681.h5ad")

common_barcodes = rad.obs_names.intersection(st.obs_names)

rad = rad[common_barcodes].copy()
st = st[common_barcodes].copy()
```

This enables joint analysis between:

- handcrafted image features
- spatial transcriptomics expression values
- patch-level metadata
- segmentation-derived cell features

---

## Quality Control

Before downstream modeling, it is recommended to check:

- missing values
- infinite values
- feature variance
- feature scale
- number of extracted patches
- barcode alignment with ST data

Example:

```python
import numpy as np

X = adata.X

print("Shape:", X.shape)
print("NaN count:", np.isnan(X).sum())
print("Inf count:", np.isinf(X).sum())
print("Feature variance:", np.nanvar(X, axis=0))
```

---

## Notes

- Each row represents one spatial transcriptomics patch.
- Each column represents one extracted radiomics feature.
- Feature availability may depend on the extraction configuration.
- Shape features require valid segmentation masks.
- Cell-aware radiomics features require segmentation outputs.
- Radiomics outputs can be regenerated from patch and segmentation files.

