from __future__ import annotations

from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np

from hestradiomics.extract._cellcomposition import DistributionExtractor
from hestradiomics.extract._cellshape import MorphologyExtractor
from hestradiomics.extract._intensity_texture import RadiomicsFeatureExtractor
from hestradiomics.extract.constants import *
from hestradiomics.utils import (
    PatchData,
    build_patch_row_base,
    build_threshold_mask,
    load_patch_data,
    normalize_class_name,
    rasterize_geometries_to_mask,
    safe_update_features,
)


class PatchProcessor:
    def __init__(
        self,
        intensity_extractor: Optional[RadiomicsFeatureExtractor] = None,
        texture_extractor: Optional[RadiomicsFeatureExtractor] = None,
        morphology_extractor: Optional[MorphologyExtractor] = None,
        distribution_extractor: Optional[DistributionExtractor] = None,
        label: int = EXTRACTOR_DEFAULT_LABEL,
    ):
        self.intensity_extractor = intensity_extractor
        self.texture_extractor = texture_extractor
        self.morphology_extractor = morphology_extractor
        self.distribution_extractor = distribution_extractor
        self.label = label

    def get_patch_cellseg(
        self,
        cellseg_df: gpd.GeoDataFrame,
        patch_idx: int,
    ) -> gpd.GeoDataFrame:
        patch_cellseg = cellseg_df[
            cellseg_df[PATCH_IDX_COLUMN] == patch_idx
        ].copy()

        patch_cellseg = patch_cellseg[
            patch_cellseg.geometry.notnull()
        ].copy()

        if len(patch_cellseg) > 0:
            patch_cellseg[CELL_CLASS_COLUMN] = (
                patch_cellseg[CELL_CLASS_COLUMN]
                .map(normalize_class_name)
            )

        return patch_cellseg

    def process_single_patch(
        self,
        f,
        img_key: str,
        coords_key: str,
        barcodes_key: str,
        i: int,
        output_dir: str,
        sample_id: str,
        mask_source: str = MASK_SOURCE_THRESHOLD,
        cellseg_df: Optional[gpd.GeoDataFrame] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        patch: PatchData = load_patch_data(
            f=f,
            img_key=img_key,
            coords_key=coords_key,
            barcodes_key=barcodes_key,
            patch_idx=i,
        )

        row = build_patch_row_base(
            patch=patch,
            output_dir=output_dir,
            sample_id=sample_id,
        )

        row[STATUS_COLUMN] = STATUS_OK

        if mask_source == MASK_SOURCE_THRESHOLD:
            return self._process_threshold_patch(
                patch=patch,
                row=row,
                use_cache=use_cache,
            )

        if mask_source == MASK_SOURCE_CELLSEG:
            if cellseg_df is None:
                raise ValueError(
                    f"mask_source='{MASK_SOURCE_CELLSEG}' requires cellseg_df"
                )

            return self._process_cellseg_patch(
                patch=patch,
                row=row,
                cellseg_df=cellseg_df,
                use_cache=use_cache,
            )

        raise ValueError(f"Unsupported mask_source: {mask_source}")

    def _process_threshold_patch(
        self,
        patch: PatchData,
        row: Dict[str, Any],
        use_cache: bool,
    ) -> Dict[str, Any]:
        mask = build_threshold_mask(
            patch.gray_patch,
            label=self.label,
        )

        row[PATCH_MASK_AREA_COLUMN] = int(
            np.count_nonzero(mask > 0)
        )

        if row[PATCH_MASK_AREA_COLUMN] < PATCH_MASK_AREA_MIN_THRESHOLD:
            row[STATUS_COLUMN] = STATUS_SKIPPED_SMALL_MASK
            return row

        if self.intensity_extractor:
            safe_update_features(
                row,
                lambda: self.intensity_extractor.extract_from_mask(
                    gray_patch=patch.gray_patch,
                    mask=mask,
                    prefix="patch_intensity_",
                    use_cache=use_cache,
                ),
                ERROR_PATCH_RADIOMICS,
            )

        if self.texture_extractor:
            safe_update_features(
                row,
                lambda: self.texture_extractor.extract_from_mask(
                    gray_patch=patch.gray_patch,
                    mask=mask,
                    prefix="patch_texture_",
                    use_cache=use_cache,
                ),
                ERROR_PATCH_RADIOMICS,
            )

        return row

    def _process_cellseg_patch(
        self,
        patch: PatchData,
        row: Dict[str, Any],
        cellseg_df: gpd.GeoDataFrame,
        use_cache: bool,
    ) -> Dict[str, Any]:
        patch_cellseg = self.get_patch_cellseg(
            cellseg_df=cellseg_df,
            patch_idx=patch.patch_idx,
        )

        row[N_CELLS_TOTAL_COLUMN] = int(
            len(patch_cellseg)
        )

        if len(patch_cellseg) == 0:
            row[STATUS_COLUMN] = STATUS_SKIPPED_NO_CELLSEG

            if self.distribution_extractor:
                row.update(
                    self.distribution_extractor.extract(
                        patch_cellseg
                    )
                )

            return row

        merged_mask = rasterize_geometries_to_mask(
            patch_cellseg.geometry.tolist(),
            image_shape=patch.gray_patch.shape,
            label=self.label,
        )

        row[CELLSEG_MASK_AREA_COLUMN] = int(
            np.count_nonzero(merged_mask > 0)
        )

        if self.intensity_extractor:
            safe_update_features(
                row,
                lambda: self.intensity_extractor.extract_from_cellseg(
                    gray_patch=patch.gray_patch,
                    patch_cellseg=patch_cellseg,
                    prefix="cellseg_intensity_",
                    use_cache=use_cache,
                ),
                ERROR_CELLSEG_RADIOMICS,
            )

        if self.texture_extractor:
            safe_update_features(
                row,
                lambda: self.texture_extractor.extract_from_cellseg(
                    gray_patch=patch.gray_patch,
                    patch_cellseg=patch_cellseg,
                    prefix="cellseg_texture_",
                    use_cache=use_cache,
                ),
                ERROR_CELLSEG_RADIOMICS,
            )

        if self.morphology_extractor:
            safe_update_features(
                row,
                lambda: self.morphology_extractor.extract(
                    gray_patch=patch.gray_patch,
                    patch_cellseg=patch_cellseg,
                    use_cache=use_cache,
                ),
                ERROR_MORPHOLOGY,
            )

        if self.distribution_extractor:
            safe_update_features(
                row,
                lambda: self.distribution_extractor.extract(
                    patch_cellseg
                ),
                ERROR_DISTRIBUTION,
            )

        return row


def build_default_patch_processor(
    filters=None,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> PatchProcessor:
    filters = filters or EXTRACTOR_DEFAULT_FILTERS

    return PatchProcessor(
        intensity_extractor=RadiomicsFeatureExtractor(
            feature_type="intensity",
            filters=filters,
            label=label,
        ),
        texture_extractor=RadiomicsFeatureExtractor(
            feature_type="texture",
            filters=filters,
            label=label,
        ),
        morphology_extractor=MorphologyExtractor(
            label=label,
        ),
        distribution_extractor=DistributionExtractor(),
        label=label,
    )


def process_single_patch(
    f,
    img_key: str,
    coords_key: str,
    barcodes_key: str,
    i: int,
    output_dir: str,
    sample_id: str,
    mask_source: str = MASK_SOURCE_THRESHOLD,
    cellseg_df: Optional[gpd.GeoDataFrame] = None,
    filters=None,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    use_cache: bool = True,
) -> Dict[str, Any]:
    processor = build_default_patch_processor(
        filters=filters,
        label=label,
    )

    return processor.process_single_patch(
        f=f,
        img_key=img_key,
        coords_key=coords_key,
        barcodes_key=barcodes_key,
        i=i,
        output_dir=output_dir,
        sample_id=sample_id,
        mask_source=mask_source,
        cellseg_df=cellseg_df,
        use_cache=use_cache,
    )
