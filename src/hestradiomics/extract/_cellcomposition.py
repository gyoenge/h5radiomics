from __future__ import annotations

from typing import Dict

import geopandas as gpd

from hestradiomics.extract.constants import *


def _empty_distribution() -> Dict[str, float]:
    out: Dict[str, float] = {
        DIST_TOTAL_COUNT_KEY: 0.0,
    }

    for cls in KNOWN_CELL_CLASSES:
        out[f"{DIST_COUNT_PREFIX}{cls}"] = 0.0
        out[f"{DIST_RATIO_PREFIX}{cls}"] = 0.0

    return out


def _normalize_cell_classes(
    patch_cellseg: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    patch_cellseg = patch_cellseg.copy()

    if CELL_CLASS_COLUMN not in patch_cellseg.columns:
        patch_cellseg[CELL_CLASS_COLUMN] = UNKNOWN_CELL_CLASS

    patch_cellseg[CELL_CLASS_COLUMN] = (
        patch_cellseg[CELL_CLASS_COLUMN]
        .fillna(UNKNOWN_CELL_CLASS)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return patch_cellseg


def extract_cell_type_distribution(
    patch_cellseg: gpd.GeoDataFrame,
) -> Dict[str, float]:
    if patch_cellseg is None or len(patch_cellseg) == 0:
        return _empty_distribution()

    patch_cellseg = _normalize_cell_classes(patch_cellseg)

    counts = patch_cellseg[CELL_CLASS_COLUMN].value_counts().to_dict()
    total = int(len(patch_cellseg))

    out: Dict[str, float] = {
        DIST_TOTAL_COUNT_KEY: float(total),
    }

    for cls in KNOWN_CELL_CLASSES:
        count = int(counts.get(cls, 0))
        ratio = float(count / total) if total > 0 else 0.0

        out[f"{DIST_COUNT_PREFIX}{cls}"] = float(count)
        out[f"{DIST_RATIO_PREFIX}{cls}"] = ratio

    return out


class DistributionExtractor:
    def extract(
        self,
        patch_cellseg: gpd.GeoDataFrame,
    ) -> Dict[str, float]:
        return extract_cell_type_distribution(patch_cellseg)

