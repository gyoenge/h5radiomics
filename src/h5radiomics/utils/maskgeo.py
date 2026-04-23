import math
import os 
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import geopandas as gpd
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

# ------------------------------------------------------------------------------
# geometry / mask helpers
# ------------------------------------------------------------------------------

def iter_polygons(geom):
    if geom is None:
        return
    if hasattr(geom, "is_empty") and geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield geom
        return

    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
        return

    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_polygons(g)
        return


def rasterize_geometries_to_mask(
    geometries,
    image_shape: Tuple[int, int],
    label: int = 255,
) -> np.ndarray:
    """
    Rasterize shapely polygons into uint8 mask with target label value.
    image_shape: (H, W)
    """
    h, w = image_shape
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    for geom in geometries:
        for poly in iter_polygons(geom):
            ext = [(float(x), float(y)) for x, y in poly.exterior.coords]
            if len(ext) >= 3:
                draw.polygon(ext, fill=label)

            for interior in poly.interiors:
                hole = [(float(x), float(y)) for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    return np.array(canvas, dtype=np.uint8)


def build_threshold_mask(gray_patch: np.ndarray, label: int = 255) -> np.ndarray:
    mask_patch = ((gray_patch > 30) & (gray_patch < 220)).astype(np.uint8)
    return (mask_patch * label).astype(np.uint8)


def build_full_patch_mask(gray_patch: np.ndarray, label: int = 255) -> np.ndarray:
    return np.full(gray_patch.shape, label, dtype=np.uint8)


def load_cellseg_dataframe(cellseg_path: Optional[str]) -> Optional[gpd.GeoDataFrame]:
    if not cellseg_path:
        return None
    if not os.path.exists(cellseg_path):
        raise FileNotFoundError(f"cellseg parquet not found: {cellseg_path}")

    gdf = gpd.read_parquet(cellseg_path)

    if "patch_idx" not in gdf.columns:
        raise ValueError("cellseg parquet must contain 'patch_idx'")
    if "geometry" not in gdf.columns:
        raise ValueError("cellseg parquet must contain 'geometry'")

    if "class_name" not in gdf.columns:
        if "class_id" in gdf.columns:
            gdf["class_name"] = gdf["class_id"].astype(str)
        else:
            gdf["class_name"] = "unknown"

    gdf["class_name"] = gdf["class_name"].fillna("unknown").astype(str)
    return gdf


def build_local_polygon_mask(
    geom,
    label: int = 255,
    margin: int = 1,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Build local binary mask using polygon bounding box.
    Returns:
      mask_local: uint8 array
      bbox_xyxy: (min_x, min_y, max_x, max_y)
    """
    polys = list(iter_polygons(geom))
    if len(polys) == 0:
        raise ValueError("No valid polygon found")

    min_x, min_y, max_x, max_y = geom.bounds
    min_x = int(math.floor(min_x)) - margin
    min_y = int(math.floor(min_y)) - margin
    max_x = int(math.ceil(max_x)) + margin
    max_y = int(math.ceil(max_y)) + margin

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)

    shifted_geoms = []
    for poly in polys:
        shifted_geoms.append(affinity.translate(poly, xoff=-min_x, yoff=-min_y))

    mask_local = rasterize_geometries_to_mask(
        shifted_geoms,
        image_shape=(height, width),
        label=label,
    )
    return mask_local, (min_x, min_y, max_x, max_y)


def crop_patch_by_bbox(gray_patch: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    min_x, min_y, max_x, max_y = bbox_xyxy
    h, w = gray_patch.shape

    x0 = max(0, min_x)
    y0 = max(0, min_y)
    x1 = min(w - 1, max_x)
    y1 = min(h - 1, max_y)

    return gray_patch[y0:y1 + 1, x0:x1 + 1]


def align_local_mask_to_crop(
    mask_local: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    gray_patch_shape: Tuple[int, int],
) -> np.ndarray:
    """
    When bbox extends outside patch, crop local mask to match actual cropped patch shape.
    """
    min_x, min_y, max_x, max_y = bbox_xyxy
    h, w = gray_patch_shape

    x_start = 0 if min_x >= 0 else -min_x
    y_start = 0 if min_y >= 0 else -min_y

    x_end = mask_local.shape[1] if max_x < w else mask_local.shape[1] - (max_x - (w - 1))
    y_end = mask_local.shape[0] if max_y < h else mask_local.shape[0] - (max_y - (h - 1))

    return mask_local[y_start:y_end, x_start:x_end]

