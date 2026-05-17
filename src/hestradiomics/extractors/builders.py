from __future__ import annotations

from typing import Any, Dict

from radiomics import featureextractor  # pyright: ignore[reportMissingImports]
from hestradiomics.extractors.constants import *

# ------------------------------------------------------------------------------
# Per-worker extractor caches
# ------------------------------------------------------------------------------
# PyRadiomics extractor initialization is relatively expensive.
# In multiprocessing settings, each worker process maintains its own cache
# so that the extractor is created only once and reused for subsequent samples.
#
# Key:
#   Tuple describing extractor configuration
#
# Value:
#   Initialized RadiomicsFeatureExtractor instance
# ------------------------------------------------------------------------------
_WORKER_EXTRACTOR_CACHE: Dict[Any, Any] = {}

# Separate cache for Shape2D extractors
# because morphology extraction uses a different configuration.
_WORKER_SHAPE_EXTRACTOR_CACHE: Dict[Any, Any] = {}


# ------------------------------------------------------------------------------
# Extractor builders
# ------------------------------------------------------------------------------

def build_radiomics_extractor(
    classes=None,
    filters=None,
    label=EXTRACTOR_DEFAULT_LABEL,
    image_type_settings=None,
):
    """
    Build a PyRadiomics extractor for intensity and texture feature extraction.

    This extractor is intended for:
        - Image patch analysis
        - ROI mask-based feature extraction

    Enabled feature classes may include:
        - firstorder
        - glcm
        - glrlm
        - glszm
        - gldm
        - ngtdm

    Enabled image filters may include:
        - Original
        - LoG
        - Wavelet
        - Square
        - etc.

    Args:
        classes:
            List of radiomics feature classes to enable.
            If None, default classes are used.

        filters:
            List of image filters to enable.
            If None, default filters are used.

        label:
            ROI label value inside the segmentation mask.

        image_type_settings:
            Optional configuration dictionary for image filters.
            Example:
                {
                    "LoG": {
                        "sigma": [1.0, 2.0, 3.0]
                    }
                }

    Returns:
        Initialized RadiomicsFeatureExtractor instance.
    """

    # Use default feature classes if not provided
    if classes is None:
        classes = EXTRACTOR_DEFAULT_CLASSES

    # Use default image filters if not provided
    if filters is None:
        filters = EXTRACTOR_DEFAULT_FILTERS

    # Use default filter settings if not provided
    if image_type_settings is None:
        image_type_settings = EXTRACTOR_DEFAULT_IMAGE_TYPE_SETTINGS

    # Create extractor with default PyRadiomics settings
    extractor = featureextractor.RadiomicsFeatureExtractor(
        **EXTRACTOR_DEFAULT_SETTINGS
    )

    # Disable all features first for explicit control
    extractor.disableAllFeatures()

    # Enable requested feature classes
    for cls in classes:
        extractor.enableFeatureClassByName(cls)

    # Ensure Original image type is always enabled
    image_types = set(filters or [])
    image_types.add(RADIOMICS_IMAGE_TYPE_ORIGINAL)

    # Enable requested image filters
    for filt in image_types:

        # Special handling for LoG filter
        # because sigma values must be explicitly configured
        if filt == RADIOMICS_IMAGE_TYPE_LOG:

            log_cfg = image_type_settings.get(
                RADIOMICS_IMAGE_TYPE_LOG,
                {},
            )

            sigma = log_cfg.get(
                RADIOMICS_IMAGE_TYPE_LOG_SIGMA,
                EXTRACTOR_DEFAULT_LOGFILTER_SIGMA,
            )

            # Validate sigma configuration
            if not isinstance(sigma, (list, tuple)) or len(sigma) == 0:
                raise ValueError(
                    f"image_type_settings[{RADIOMICS_IMAGE_TYPE_LOG}]"
                    f"[{RADIOMICS_IMAGE_TYPE_LOG_SIGMA}] "
                    f"must be a non-empty list"
                )

            # Enable LoG filter with custom sigma values
            extractor.enableImageTypeByName(
                RADIOMICS_IMAGE_TYPE_LOG,
                customArgs={
                    RADIOMICS_IMAGE_TYPE_LOG_SIGMA: list(sigma)
                },
            )

        else:
            # Enable other image types normally
            extractor.enableImageTypeByName(filt)

    return extractor


def build_shape2d_extractor(label=EXTRACTOR_DEFAULT_LABEL):
    """
    Build a Shape2D-only extractor for cell morphology analysis.

    This extractor is intended for:
        - Per-cell shape feature extraction
        - Binary nucleus/cell masks
        - Morphological measurements

    Enabled feature class:
        - shape2D

    Example features:
        - area
        - perimeter
        - elongation
        - major axis length
        - minor axis length
        - etc.

    Args:
        label:
            ROI label value inside the mask.

    Returns:
        Initialized Shape2D RadiomicsFeatureExtractor instance.
    """

    # Shape extraction only requires minimal settings
    settings = {
        "label": label,
        "force2D": EXTRACTOR_DEFAULT_SETTINGS["force2D"],
        "force2Ddimension": EXTRACTOR_DEFAULT_SETTINGS["force2Ddimension"],
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Disable all features first
    extractor.disableAllFeatures()

    # Enable only Shape2D features
    extractor.enableFeatureClassByName(
        RADIOMICS_FEATURE_CLASS_SHAPE2D
    )

    # Shape features are extracted from original masks only
    extractor.enableImageTypeByName(
        RADIOMICS_IMAGE_TYPE_ORIGINAL
    )

    return extractor


def _get_worker_radiomics_extractor(
    classes,
    filters,
    label,
    image_type_settings,
):
    """
    Retrieve or create a cached radiomics extractor for the current worker.

    A unique cache key is generated from:
        - feature classes
        - image filters
        - label
        - image filter settings

    This avoids repeatedly rebuilding identical extractors,
    which improves multiprocessing performance.

    Returns:
        Cached RadiomicsFeatureExtractor instance.
    """

    key = (
        tuple(classes or []),
        tuple(filters or []),
        label,
        repr(image_type_settings or {}),
    )

    # Build extractor only if not already cached
    if key not in _WORKER_EXTRACTOR_CACHE:
        _WORKER_EXTRACTOR_CACHE[key] = build_radiomics_extractor(
            classes=classes,
            filters=filters,
            label=label,
            image_type_settings=image_type_settings,
        )

    return _WORKER_EXTRACTOR_CACHE[key]


def _get_worker_shape2d_extractor(label):
    """
    Retrieve or create a cached Shape2D extractor for the current worker.

    Shape extractors are cached separately because their configuration
    differs from standard intensity/texture extractors.

    Args:
        label:
            ROI label value inside the mask.

    Returns:
        Cached Shape2D RadiomicsFeatureExtractor instance.
    """

    key = (
        RADIOMICS_FEATURE_CLASS_SHAPE2D,
        label,
    )

    # Build Shape2D extractor only once per worker
    if key not in _WORKER_SHAPE_EXTRACTOR_CACHE:
        _WORKER_SHAPE_EXTRACTOR_CACHE[key] = (
            build_shape2d_extractor(label=label)
        )

    return _WORKER_SHAPE_EXTRACTOR_CACHE[key]
