from __future__ import annotations

from typing import Any, Dict

from radiomics import featureextractor  # pyright: ignore[reportMissingImports] 
from h5radiomics.engines.extractors.constants import *

_WORKER_EXTRACTOR_CACHE: Dict[Any, Any] = {}
_WORKER_SHAPE_EXTRACTOR_CACHE: Dict[Any, Any] = {}

# ------------------------------------------------------------------------------
# extractor builders
# ------------------------------------------------------------------------------

def build_radiomics_extractor(
    classes=None,
    filters=None,
    label=EXTRACTOR_DEFAULT_LABEL,
    image_type_settings=None,
):
    """
    For intensity/texture extraction on image patch + ROI mask.
    """
    if classes is None:
        classes = EXTRACTOR_DEFAULT_CLASSES
    if filters is None:
        filters = EXTRACTOR_DEFAULT_FILTERS
    if image_type_settings is None:
        image_type_settings = EXTRACTOR_DEFAULT_IMAGE_TYPE_SETTINGS

    extractor = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_DEFAULT_SETTINGS)
    extractor.disableAllFeatures()

    for cls in classes:
        extractor.enableFeatureClassByName(cls)

    image_types = set(filters or [])
    image_types.add(RADIOMICS_IMAGE_TYPE_ORIGINAL)

    for filt in image_types:
        if filt == RADIOMICS_IMAGE_TYPE_LOG:
            log_cfg = image_type_settings.get(RADIOMICS_IMAGE_TYPE_LOG, {})
            sigma = log_cfg.get(RADIOMICS_IMAGE_TYPE_LOG_SIGMA, EXTRACTOR_DEFAULT_LOGFILTER_SIGMA)
            if not isinstance(sigma, (list, tuple)) or len(sigma) == 0:
                raise ValueError(f"image_type_settings[{RADIOMICS_IMAGE_TYPE_LOG}][{RADIOMICS_IMAGE_TYPE_LOG_SIGMA}] must be a non-empty list")
            extractor.enableImageTypeByName(RADIOMICS_IMAGE_TYPE_LOG, customArgs={RADIOMICS_IMAGE_TYPE_LOG_SIGMA: list(sigma)})
        else:
            extractor.enableImageTypeByName(filt)

    return extractor


def build_shape2d_extractor(label=EXTRACTOR_DEFAULT_LABEL):
    """
    For per-cell morphology extraction on local binary mask.
    """
    settings = {
        "label": label,
        "force2D": EXTRACTOR_DEFAULT_SETTINGS["force2D"],
        "force2Ddimension": EXTRACTOR_DEFAULT_SETTINGS["force2Ddimension"],
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(RADIOMICS_FEATURE_CLASS_SHAPE2D)
    extractor.enableImageTypeByName(RADIOMICS_IMAGE_TYPE_ORIGINAL)
    return extractor


def _get_worker_radiomics_extractor(classes, filters, label, image_type_settings):
    key = (
        tuple(classes or []),
        tuple(filters or []),
        label,
        repr(image_type_settings or {}),
    )
    if key not in _WORKER_EXTRACTOR_CACHE:
        _WORKER_EXTRACTOR_CACHE[key] = build_radiomics_extractor(
            classes=classes,
            filters=filters,
            label=label,
            image_type_settings=image_type_settings,
        )
    return _WORKER_EXTRACTOR_CACHE[key]


def _get_worker_shape2d_extractor(label):
    key = (RADIOMICS_FEATURE_CLASS_SHAPE2D, label)
    if key not in _WORKER_SHAPE_EXTRACTOR_CACHE:
        _WORKER_SHAPE_EXTRACTOR_CACHE[key] = build_shape2d_extractor(label=label)
    return _WORKER_SHAPE_EXTRACTOR_CACHE[key]

