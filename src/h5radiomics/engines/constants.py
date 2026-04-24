"""
Cell class labels (Cell segmentation output)
: Must be consistent with upstream cell segmentation model outputs.
"""

KNOWN_CELL_CLASSES = [
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
    "background",
    "unknown",
]

"""
Radiomics feature prefixes
: Used to identify valid radiomics features from PyRadiomics outputs.
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
Extractor Settings 
: 
"""

EXTRACTOR_DEFAULT_CLASSES = [
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
]

EXTRACTOR_DEFAULT_FILTERS = [
    "Original",
]

EXTRACTOR_DEFAULT_IMAGE_TYPE_SETTINGS = {}

EXTRACTOR_DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "verbose": False,
    "force2D": True,
    "force2Ddimension": 0,
    "distances": [1],
}

EXTRACTOR_DEFAULT_LABEL = 255

EXTRACTOR_DEFAULT_LOGFILTER_SIGMA = [1.0, 2.0, 3.0]

EXTRACTOR_DEFAULT_MASK_ROI_AREA_THRESHOLD = 50 


"""

"""

STATUS_OK = "ok"
STATUS_SKIPPED_SMALL_MASK = "skipped_small_mask"
STATUS_SKIPPED_NO_CELLSEG = "skipped_no_cellseg_mask"

ERROR_PATCH_RADIOMICS = "error_patch_radiomics"
ERROR_CELLSEG_RADIOMICS = "error_cellseg_radiomics"
ERROR_MORPHOLOGY = "error_morphology"
ERROR_DISTRIBUTION = "error_distribution"


"""

"""

PATCH_FEATURE_PREFIX = "patch_"
CELLSEG_FEATURE_PREFIX = "cellseg"
CELLSEG_ALL_SUFFIX = "all"
MORPH_FEATURE_PREFIX = "morph_"

DIST_TOTAL_COUNT_KEY = "dist_cell_count_total"
DIST_COUNT_PREFIX = "dist_count_"
DIST_RATIO_PREFIX = "dist_ratio_"

MASK_SOURCE_THRESHOLD = "threshold"
MASK_SOURCE_CELLSEG = "cellseg"

RADIOMICS_IMAGE_TYPE_ORIGINAL = "Original"
RADIOMICS_IMAGE_TYPE_LOG = "LoG"
RADIOMICS_FEATURE_CLASS_SHAPE2D = "shape2D"

"""

"""

PATCH_MASK_AREA_MIN_THRESHOLD = 50
LOCAL_MASK_MIN_PIXELS = 3
FIRSTORDER_HIST_MAX_BINS = 16
FIRSTORDER_HIST_MIN_BINS = 2
LOCAL_POLYGON_MASK_MARGIN = 1

DEFAULT_CLIP_LOWER_Q = 0.01
DEFAULT_CLIP_UPPER_Q = 0.99

"""

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

"""

DEFAULT_CELLTYPE_MODE = "merged"

"""

"""

# mask source
MASK_SOURCE_THRESHOLD = "threshold"
MASK_SOURCE_CELLSEG = "cellseg"

# image types / feature classes
IMAGE_TYPE_ORIGINAL = "Original"
IMAGE_TYPE_LOG = "LoG"
FEATURE_CLASS_SHAPE2D = "shape2D"

# prefixes / naming
PATCH_FEATURE_PREFIX = "patch_"
MORPH_FEATURE_PREFIX = "morph_"
DIST_TOTAL_COUNT_KEY = "dist_cell_count_total"
DIST_COUNT_PREFIX = "dist_count_"
DIST_RATIO_PREFIX = "dist_ratio_"

# status / error labels
STATUS_OK = "ok"
STATUS_SKIPPED_SMALL_MASK = "skipped_small_mask"
STATUS_SKIPPED_NO_CELLSEG_MASK = "skipped_no_cellseg_mask"

ERROR_PATCH_RADIOMICS = "error_patch_radiomics"
ERROR_CELLSEG_RADIOMICS = "error_cellseg_radiomics"
ERROR_MORPHOLOGY = "error_morphology"
ERROR_DISTRIBUTION = "error_distribution"

# thresholds
PATCH_MASK_AREA_MIN_THRESHOLD = 50
LOCAL_MASK_MIN_ROI_PIXELS = 3
LOCAL_POLYGON_MASK_MARGIN = 1

# histogram / clipping
FIRSTORDER_HIST_BINS_MIN = 2
FIRSTORDER_HIST_BINS_MAX = 16
DEFAULT_CLIP_LOWER_Q = 0.01
DEFAULT_CLIP_UPPER_Q = 0.99

