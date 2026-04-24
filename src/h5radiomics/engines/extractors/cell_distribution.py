from __future__ import annotations

from typing import Dict
import geopandas as gpd
from h5radiomics.engines.constants import *


# ------------------------------------------------------------------------------
# cell-type distribution features
# ------------------------------------------------------------------------------

def extract_cell_type_distribution(
    patch_cellseg: gpd.GeoDataFrame,
) -> Dict[str, float]:
    out = {}

    if patch_cellseg is None or len(patch_cellseg) == 0:
        out["dist_cell_count_total"] = 0.0
        for cls in KNOWN_CELL_CLASSES:
            out[f"dist_count_{cls}"] = 0.0
            out[f"dist_ratio_{cls}"] = 0.0
        return out

    patch_cellseg = patch_cellseg.copy()
    patch_cellseg["class_name"] = (
        patch_cellseg["class_name"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    counts = patch_cellseg["class_name"].value_counts().to_dict()
    total = int(len(patch_cellseg))

    out["dist_cell_count_total"] = float(total)

    class_names = sorted(set(KNOWN_CELL_CLASSES).union(set(counts.keys())))
    for cls in class_names:
        cnt = int(counts.get(cls, 0))
        ratio = float(cnt / total) if total > 0 else 0.0
        out[f"dist_count_{cls}"] = float(cnt)
        out[f"dist_ratio_{cls}"] = ratio

    return out

