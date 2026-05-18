# Segment File Format

The segmentation file stores cell-level segmentation metadata for each spatial transcriptomics patch.  
Instead of storing dense bitmap masks, the pipeline stores polygon-based vector representations of segmented cells, enabling compact storage and flexible downstream analysis.

The segmentation files are stored in HDF5 (`.h5`) format.

Example directory structure:

```text
data/
└── hest/
    └── IDC/
        └── segment/
            ├── NCBI681.h5
            ├── NCBI682.h5
            ├── NCBI683.h5
            └── ...
````

Each `.h5` file corresponds to a single WSI sample and contains:

* patch-level metadata
* cell-level segmentation metadata
* polygon-based cell boundaries
* cell type annotations

Example internal structure of a segmentation file:

```text
NCBI681.h5
├── cells/
│   ├── barcode
│   ├── cell_id_in_patch
│   ├── class_id
│   ├── class_name
│   ├── geometry_wkt
│   └── patch_idx
│
└── patches/
    ├── barcode
    ├── coord_raw_json
    ├── n_cells
    └── patch_idx
```

Each segmentation file contains two groups:

* `patches/` stores patch-level metadata
* `cells/` stores cell-level segmentation information

This structure enables efficient mapping between spatial transcriptomics patches and segmented cells.


---

## File Structure

#### `patches/`

Patch-level metadata.

###### `patches/barcode`

Visium spot barcode corresponding to each patch.

```python
shape: (N_patches,)
dtype: object
```

Example:

```text
AAACAAGTATCTCCCA-1
```

###### `patches/coord_raw_json`

Original WSI coordinates of each patch center.

```python
shape: (N_patches,)
dtype: object
```

Example:

```text
[18930, 7797]
```

Coordinates are stored as JSON strings.


###### `patches/n_cells`

Number of segmented cells contained in each patch.

```python
shape: (N_patches,)
dtype: int64
```

Example:

```python
[45, 32, 76, ...]
```

###### `patches/patch_idx`

Unique integer patch index.

```python
shape: (N_patches,)
dtype: int64
```

Example:

```python
[0, 1, 2, 3, ...]
```

---

#### `cells/`

Cell-level segmentation metadata.

Each row corresponds to a single segmented cell.

###### `cells/barcode`

Visium barcode of the parent patch containing the cell.

```python
shape: (N_cells,)
dtype: object
```

###### `cells/cell_id_in_patch`

Cell index within the patch.

```python
shape: (N_cells,)
dtype: int64
```

Example:

```python
[1, 2, 3, 4, ...]
```

###### `cells/class_id`

Cell type identifier predicted by the segmentation/classification model.

```python
shape: (N_cells,)
dtype: int64
```

Example:

```python
[-1, 1, 2, 3, 4, 5]
```

Typical classes may include:

| class_id | class_name   |
| -------- | ------------ |
| 1        | neoplastic   |
| 2        | inflammatory |
| 3        | connective   |
| 4        | dead         |
| 5        | epithelial   |

`-1` may indicate unknown or unclassified cells.

###### `cells/class_name`

Human-readable cell type label.

```python
shape: (N_cells,)
dtype: object
```

Example:

```text
connective
inflammatory
neoplastic
```

###### `cells/geometry_wkt`

Polygon representation of the segmented cell mask stored in WKT (Well-Known Text) format.

```python
shape: (N_cells,)
dtype: object
```

Example:

```text
POLYGON ((64 0, 63 1, 63 2, ...))
```

These polygons define the cell boundaries in patch-local coordinates.

###### `cells/patch_idx`

Patch index corresponding to the parent patch.

```python
shape: (N_cells,)
dtype: int64
```

This field enables efficient relational mapping between cells and patches.

Example:

| cell   | patch_idx |
| ------ | --------- |
| cell_0 | 0         |
| cell_1 | 0         |
| cell_2 | 1         |

---

## Why Polygon-Based Storage?

Instead of storing dense binary masks, the pipeline stores vector polygons for several reasons:

* significantly smaller storage size
* scalable for large WSI datasets
* preserves exact cell boundaries
* flexible for downstream geometry analysis
* enables efficient graph construction
* compatible with radiomics and morphometric pipelines

This design is particularly suitable for computational pathology and spatial transcriptomics workflows.

---

## Typical Downstream Workflow

The segmentation polygons are typically converted into rasterized masks during processing.

```text
WKT polygon
    ↓
Rasterization
    ↓
Binary mask
    ↓
Radiomics / morphology extraction
```

Example rasterization:

```python
mask = np.zeros((224, 224), dtype=np.uint8)
cv2.fillPoly(mask, [polygon_pts], 1)
```

---

## PyRadiomics Label Convention

PyRadiomics expects the ROI label to match the specified `label` argument.

Recommended binary mask convention:

```python
background = 0
foreground = 1
```

If masks are generated with value `255`, convert them before extraction:

```python
mask = (mask > 0).astype(np.uint8)
```

Otherwise, PyRadiomics may raise:

```python
ValueError: Label (1) not present in mask. Choose from [255]
```

---

## Potential Applications

This segmentation structure supports a wide range of downstream analyses:

* cell morphology analysis
* radiomics feature extraction
* nucleus shape statistics
* cell graph construction
* Delaunay triangulation graphs
* neighborhood analysis
* class-specific feature extraction
* tumor microenvironment modeling
* spatial transcriptomics integration

---

## Example

```python
import h5py

with h5py.File("NCBI681.h5", "r") as h5:
    print(h5.keys())

    barcodes = h5["patches/barcode"][:]
    polygons = h5["cells/geometry_wkt"][:]
```

