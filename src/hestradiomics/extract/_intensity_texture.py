from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import SimpleITK as sitk

from radiomics import featureextractor

from hestradiomics.extract.constants import *
from hestradiomics.utils import (
    build_threshold_mask,
    rasterize_geometries_to_mask,
    make_feature_prefix,
)

# ------------------------------------------------------------------------------
# Types
# ------------------------------------------------------------------------------

FeatureType = Literal["intensity", "texture"]

FilterName = Literal[
    "Original",
    "Wavelet",
    "LoG",
    "Square",
    "SquareRoot",
    "Logarithm",
    "Exponential",
]

FilterType = List[FilterName]

# ------------------------------------------------------------------------------
# Feature mapping
# ------------------------------------------------------------------------------

FEATURE_CLASS_MAP: Dict[FeatureType, List[str]] = {
    "intensity": ["firstorder"],
    "texture": ["glcm", "glrlm", "glszm", "gldm", "ngtdm"],
}

# ------------------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------------------

def _is_radiomics_feature_key(
    k: str,
) -> bool:

    k = k.lower()

    return any(
        k.startswith(prefix)
        for prefix in RADIOMICS_IMAGE_PREFIXES
    )


def _clean_radiomics_result(
    feature_dict: Dict[str, Any],
) -> Dict[str, float]:

    out = {}

    for k, v in feature_dict.items():

        if not _is_radiomics_feature_key(k):
            continue

        try:
            out[k] = float(v)

        except Exception:
            continue

    return out


def _add_prefix_to_keys(
    d: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:

    return {
        f"{prefix}{k}": v
        for k, v in d.items()
    }


def _execute_radiomics_on_mask(
    gray_patch: np.ndarray,
    mask_patch: np.ndarray,
    extractor,
) -> Dict[str, float]:

    if (
        mask_patch is None
        or np.count_nonzero(mask_patch > 0)
        < EXTRACTOR_DEFAULT_MASK_ROI_AREA_THRESHOLD
    ):
        return {}

    image_sitk = sitk.GetImageFromArray(gray_patch)
    mask_sitk = sitk.GetImageFromArray(mask_patch)

    features = extractor.execute(
        image_sitk,
        mask_sitk,
    )

    return _clean_radiomics_result(features)

# ------------------------------------------------------------------------------
# Radiomics extraction functions
# ------------------------------------------------------------------------------

def extract_patch_level_radiomics(
    gray_patch: np.ndarray,
    extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:

    patch_mask = build_threshold_mask(
        gray_patch,
        label=label,
    )

    features = _execute_radiomics_on_mask(
        gray_patch,
        patch_mask,
        extractor,
    )

    return _add_prefix_to_keys(
        features,
        PATCH_FEATURE_PREFIX,
    )


def extract_cellseg_level_radiomics(
    gray_patch: np.ndarray,
    patch_cellseg: gpd.GeoDataFrame,
    extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:

    out = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        return out

    patch_cellseg = patch_cellseg[
        patch_cellseg.geometry.notnull()
    ].copy()

    if len(patch_cellseg) == 0:
        return out

    mask_all = rasterize_geometries_to_mask(
        patch_cellseg.geometry.tolist(),
        image_shape=gray_patch.shape,
        label=label,
    )

    all_feats = _execute_radiomics_on_mask(
        gray_patch,
        mask_all,
        extractor,
    )

    out.update(
        _add_prefix_to_keys(
            all_feats,
            make_feature_prefix(
                CELLSEG_FEATURE_PREFIX,
                CELLSEG_ALL_SUFFIX,
            ),
        )
    )

    return out

# ------------------------------------------------------------------------------
# Extractor class
# ------------------------------------------------------------------------------

class RadiomicsFeatureExtractor:

    _PROCESS_LOCAL_CACHE: Dict[Any, Any] = {}

    def __init__(
        self,
        feature_type: FeatureType,
        filters: Optional[FilterType] = None,
        label: int = EXTRACTOR_DEFAULT_LABEL,
        settings: Optional[Dict[str, Any]] = None,
        log_sigma: Optional[List[float]] = None,
    ):

        self.feature_type = feature_type
        self.filters = filters or ["Original"]
        self.label = label

        self.settings = {
            **EXTRACTOR_DEFAULT_SETTINGS,
            "label": self.label,
        }

        if settings:
            self.settings.update(settings)

        self.log_settings = {
            RADIOMICS_IMAGE_TYPE_LOG_SIGMA:
                log_sigma
                or EXTRACTOR_DEFAULT_LOGFILTER_SIGMA
        }

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build_extractor(self):

        extractor = featureextractor.RadiomicsFeatureExtractor(
            **self.settings
        )

        extractor.disableAllFeatures()

        for image_type in self._get_image_types():

            if image_type == RADIOMICS_IMAGE_TYPE_LOG:

                extractor.enableImageTypeByName(
                    RADIOMICS_IMAGE_TYPE_LOG,
                    customArgs=self.log_settings,
                )

            else:

                extractor.enableImageTypeByName(
                    image_type
                )

        for cls in FEATURE_CLASS_MAP[self.feature_type]:

            extractor.enableFeatureClassByName(cls)

        return extractor

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    def get_cached_extractor(self):

        key = self._make_cache_key()

        if key not in self._PROCESS_LOCAL_CACHE:

            self._PROCESS_LOCAL_CACHE[key] = (
                self.build_extractor()
            )

        return self._PROCESS_LOCAL_CACHE[key]

    # -------------------------------------------------------------------------
    # Public extract APIs
    # -------------------------------------------------------------------------

    def extract_from_patch(
        self,
        gray_patch: np.ndarray,
        use_cache: bool = True,
    ) -> Dict[str, float]:

        extractor = (
            self.get_cached_extractor()
            if use_cache
            else self.build_extractor()
        )

        return extract_patch_level_radiomics(
            gray_patch=gray_patch,
            extractor=extractor,
            label=self.label,
        )

    def extract_from_cellseg(
        self,
        gray_patch: np.ndarray,
        patch_cellseg: gpd.GeoDataFrame,
        use_cache: bool = True,
    ) -> Dict[str, float]:

        extractor = (
            self.get_cached_extractor()
            if use_cache
            else self.build_extractor()
        )

        return extract_cellseg_level_radiomics(
            gray_patch=gray_patch,
            patch_cellseg=patch_cellseg,
            extractor=extractor,
            label=self.label,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_image_types(self) -> List[str]:

        image_types = set(self.filters or [])

        image_types.add(
            RADIOMICS_IMAGE_TYPE_ORIGINAL
        )

        return sorted(image_types)

    def _make_cache_key(
        self,
    ) -> Tuple[Any, ...]:

        return (
            self.feature_type,
            tuple(sorted(self.filters)),
            self.label,
            repr(self.settings),
            repr(self.log_settings),
        )
