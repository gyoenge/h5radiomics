from __future__ import annotations

from typing import Any, Dict

from radiomics import featureextractor  # pyright: ignore[reportMissingImports] 
from h5radiomics.engines.constants import *

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
    image_types.add("Original")

    for filt in image_types:
        if filt == "LoG":
            log_cfg = image_type_settings.get("LoG", {})
            sigma = log_cfg.get("sigma", EXTRACTOR_DEFAULT_LOGFILTER_SIGMA)
            if not isinstance(sigma, (list, tuple)) or len(sigma) == 0:
                raise ValueError("image_type_settings['LoG']['sigma'] must be a non-empty list")
            extractor.enableImageTypeByName("LoG", customArgs={"sigma": list(sigma)})
        else:
            extractor.enableImageTypeByName(filt)

    return extractor


def build_shape2d_extractor(label=EXTRACTOR_DEFAULT_LABEL):
    """
    For per-cell morphology extraction on local binary mask.
    """
    settings = {
        "label": label,
        "force2D": True,
        "force2Ddimension": 0,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape2D")
    extractor.enableImageTypeByName("Original")
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
    key = ("shape2d", label)
    if key not in _WORKER_SHAPE_EXTRACTOR_CACHE:
        _WORKER_SHAPE_EXTRACTOR_CACHE[key] = build_shape2d_extractor(label=label)
    return _WORKER_SHAPE_EXTRACTOR_CACHE[key]

