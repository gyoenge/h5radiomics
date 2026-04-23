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

