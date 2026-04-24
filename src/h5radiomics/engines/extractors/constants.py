"""
Cell class labels from upstream cell segmentation model.

Note: 
- Must match exactly with the segmentation model output.
- Used for distribution / cell-type-aware feature extraction.
"""

KNOWN_CELL_CLASSES = [
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
    # "background",   # excluded by design
    # "unknown",      # excluded by design
]

# For unknown cell class 
UNKNOWN_CELL_CLASS = "unknown"

# Column name used in cell segmentation dataframe
CELL_CLASS_COLUMN = "class_name"


"""
Radiomics image prefixes used to filter valid PyRadiomics outputs.

PyRadiomics generates features with prefixes depending on:
- Image type (Original, Wavelet, LoG, etc.)

We use these prefixes to:
- Identify valid features
- Filter out non-radiomics metadata columns
"""

RADIOMICS_IMAGE_PREFIXES = (
    "original_",
    "wavelet-",
    "log-sigma-",
    "square_",
    "squareroot_",
    "logarithm_",
    "exponential_",
)


"""
Default configuration for PyRadiomics FeatureExtractor.

These values define:
- Feature classes
- Image filters
- Extraction behavior

Note: 
- Can be overridden via YAML config
"""

# Feature classes to extract
EXTRACTOR_DEFAULT_CLASSES = [
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
]

# Image types (filters)
EXTRACTOR_DEFAULT_FILTERS = [
    "Original",
]

# Additional image type settings (e.g., LoG sigma)
EXTRACTOR_DEFAULT_IMAGE_TYPE_SETTINGS = {}

# Core extractor parameters
EXTRACTOR_DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "verbose": False,
    "force2D": True,
    "force2Ddimension": 0,
    "distances": [1],
}

# Mask label value used for ROI extraction
EXTRACTOR_DEFAULT_LABEL = 255

# LoG filter sigma values (used when LoG enabled)
EXTRACTOR_DEFAULT_LOGFILTER_SIGMA = [1.0, 2.0, 3.0]

# Minimum ROI area to consider valid for radiomics
EXTRACTOR_DEFAULT_MASK_ROI_AREA_THRESHOLD = 50

"""
Processing status flags.

Used in output rows to track:
- Success
- Skipped cases
- Error types
"""

# 정상 처리
STATUS_OK = "ok"

# Skip cases
STATUS_SKIPPED_SMALL_MASK = "skipped_small_mask"
STATUS_SKIPPED_NO_CELLSEG = "skipped_no_cellseg_mask"

# Error categories
ERROR_PATCH_RADIOMICS = "error_patch_radiomics"
ERROR_CELLSEG_RADIOMICS = "error_cellseg_radiomics"
ERROR_MORPHOLOGY = "error_morphology"
ERROR_DISTRIBUTION = "error_distribution"


"""
Feature naming rules for consistent downstream usage.

Prefix strategy:
- patch_      : patch-level radiomics
- cellseg_*   : cell segmentation-based features
- morph_      : morphology features
- dist_*      : cell distribution features
"""

PATCH_FEATURE_PREFIX = "patch_"

CELLSEG_FEATURE_PREFIX = "cellseg"
CELLSEG_ALL_SUFFIX = "all"

MORPH_FEATURE_PREFIX = "morph_"

# Distribution features
DIST_TOTAL_COUNT_KEY = "dist_cell_count_total"
DIST_COUNT_PREFIX = "dist_count_"
DIST_RATIO_PREFIX = "dist_ratio_"


"""
Mask source identifiers.

Defines which mask was used for feature extraction:
- threshold : simple threshold-based mask
- cellseg   : cell segmentation mask
"""

MASK_SOURCE_THRESHOLD = "threshold"
MASK_SOURCE_CELLSEG = "cellseg"


"""
Internal constants related to PyRadiomics naming.
"""

RADIOMICS_IMAGE_TYPE_ORIGINAL = "Original"
RADIOMICS_IMAGE_TYPE_LOG = "LoG"
RADIOMICS_IMAGE_TYPE_LOG_SIGMA = "sigma"

# Shape feature class name (PyRadiomics)
RADIOMICS_FEATURE_CLASS_SHAPE2D = "shape2D"


"""
Low-level thresholds for numerical stability and noise filtering.
"""

# Patch-level mask minimum area
PATCH_MASK_AREA_MIN_THRESHOLD = EXTRACTOR_DEFAULT_MASK_ROI_AREA_THRESHOLD

# Local mask minimum pixels (very small region filtering)
LOCAL_MASK_MIN_PIXELS = 3

# Histogram bin constraints (first-order features)
FIRSTORDER_HIST_MAX_BINS = 16
FIRSTORDER_HIST_MIN_BINS = 2

# Polygon mask margin (padding)
LOCAL_POLYGON_MASK_MARGIN = 1


"""
Default quantile clipping values for feature normalization.
"""

DEFAULT_CLIP_LOWER_Q = 0.01
DEFAULT_CLIP_UPPER_Q = 0.99


"""
Statistical aggregation keys for morphology feature vectors.

Used in manual aggregation (instead of PyRadiomics firstorder)
to ensure numerical stability.
"""

MORPH_AGG_STAT_KEYS = [
    "mean",
    "median",
    "minimum",
    "maximum",
    "range",
    "variance",
    "standarddeviation",
    "p10",
    "p25",
    "p75",
    "p90",
    "iqr",
    "meanabsolutedeviation",
    "skewness",
    "kurtosis",
    "entropy",
    "energy",
    "rootmeansquared",
]


"""
Cell-type aggregation mode.

merged:
    - all cell types aggregated together

future options:
    - per_class (class-wise features)
"""

DEFAULT_CELLTYPE_MODE = "merged"


"""
Metadata / Output Schema Constants
: Column names and auxiliary values used in patch-level processing outputs.
: Avoid hard-coded dataframe keys across processors.
"""

# Common metadata columns
PATCH_IDX_COLUMN = "patch_idx"
STATUS_COLUMN = "status"
MASK_PATH_COLUMN = "mask_path"

# Area / count statistics columns
PATCH_MASK_AREA_COLUMN = "patch_mask_area"
CELLSEG_MASK_AREA_COLUMN = "cellseg_mask_area"
N_CELLS_TOTAL_COLUMN = "n_cells_total"

# Saved mask suffix names 
THRESHOLD_MASK_SUFFIX = "__threshold"
CELLSEG_ALL_MASK_SUFFIX = "__cellseg_all"

# Stats columns 
FEATURE_COLUMN = "feature"
LOWER_Q_COLUMN = "lower_q"
UPPER_Q_COLUMN = "upper_q"
LOWER_BOUND_COLUMN = "lower_bound" # for clipped 
UPPER_BOUND_COLUMN = "upper_bound" # for clipped 
MEAN_COLUMN = "mean" # for clipped 
STD_COLUMN = "std" # for clipped 
MIN_AFTER_CLIP_COLUMN = "min_after_clip" # for clipped 
MAX_AFTER_CLIP_COLUMN = "max_after_clip" # for clipped 
Z_MEAN_COLUMN = "z_mean"
Z_STD_COLUMN = "z_std"
SCALED_MIN_COLUMN = "scaled_min" 
SCALED_MAX_COLUMN = "scaled_max"

