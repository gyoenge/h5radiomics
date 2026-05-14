from __future__ import annotations

from typing import Dict
import geopandas as gpd
from h5radiomics.engines.extractors.constants import *


# ------------------------------------------------------------------------------
# cell-type distribution features
# ------------------------------------------------------------------------------

def extract_cell_type_distribution(
    patch_cellseg: gpd.GeoDataFrame,
) -> Dict[str, float]:
    out = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        out[DIST_TOTAL_COUNT_KEY] = 0.0
        for cls in KNOWN_CELL_CLASSES:
            out[f"{DIST_COUNT_PREFIX}{cls}"] = 0.0
            out[f"{DIST_RATIO_PREFIX}{cls}"] = 0.0
        return out

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg[CELL_CLASS_COLUMN] = (
        patch_cellseg[CELL_CLASS_COLUMN]
        .fillna(UNKNOWN_CELL_CLASS)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    counts = patch_cellseg[CELL_CLASS_COLUMN].value_counts().to_dict()
    total = int(len(patch_cellseg))

    out[DIST_TOTAL_COUNT_KEY] = float(total)

    # class_names = sorted(set(KNOWN_CELL_CLASSES).union(set(counts.keys())))
    class_names = list(KNOWN_CELL_CLASSES)
    for cls in class_names:
        cnt = int(counts.get(cls, 0))
        ratio = float(cnt / total) if total > 0 else 0.0
        out[f"{DIST_COUNT_PREFIX}{cls}"] = float(cnt)
        out[f"{DIST_RATIO_PREFIX}{cls}"] = ratio

    return out

