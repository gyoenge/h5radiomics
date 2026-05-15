from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from shapely import wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hestradiomics.utils.h5 import ensure_uint8_rgb


# ============================================================
# Paths / Constants
# ============================================================

HEST_ROOT = "/root/workspace/hest-radiomics/data/hest"

DOWNLOAD_ONCOTREE = [
    "IDC",
    "SKCM",
    "LUAD",
    "PAAD",
    "COAD",
]

MODEL_NAME = "CellViT-SAM-H-x20.pth"
MODEL_PATH = f"/root/workspace/hest-radiomics/models/{MODEL_NAME}"

DEVICE = "cuda:0"
BATCH_SIZE = 8
NUM_WORKERS = 0
USE_CLASS_COLOR = True

RUN_SEGMENT = True
RUN_OVERLAY = True

OVERWRITE_SEGMENT = False
OVERWRITE_OVERLAY = False


MODEL_SRC_MAP = {
    "CellViT-256-x20.pth": "1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2",
    "CellViT-256-x40.pth": "1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q",
    "CellViT-SAM-H-x40.pth": "1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV",
    "CellViT-SAM-H-x20.pth": "1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30",
}


# ============================================================
# Utils
# ============================================================

def infer_cellvit_model_type(model_name: str) -> str:
    name = model_name.upper()

    if "SAM" in name:
        return "SAM"
    if "HIPT" in name:
        return "HIPT"
    if "CELLVIT" in name or "256" in name:
        return "CellViT"

    raise ValueError(f"Cannot infer CellViT model type from model_name={model_name}")


def debug_print(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _decode_scalar(v):
    if isinstance(v, np.ndarray) and v.shape:
        v = v[0]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def polygon_from_mask(mask: np.ndarray) -> List[Polygon]:
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: List[Polygon] = []

    for cnt in contours:
        if len(cnt) < 3:
            continue

        coords = cnt[:, 0, :]
        poly = Polygon(coords)

        if poly.is_valid and poly.area > 0:
            polygons.append(poly)

    return polygons


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


def verify_or_download_model(model_path: str, model_name: str):
    if os.path.exists(model_path):
        print(f"[INFO] Found model at {model_path}")
        return

    if model_name not in MODEL_SRC_MAP:
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Known downloadable models: {list(MODEL_SRC_MAP.keys())}"
        )

    try:
        import gdown
    except Exception as e:
        raise ImportError("gdown is required to auto-download model weights.") from e

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"[INFO] Downloading {model_name} -> {model_path}")
    gdown.download(
        id=MODEL_SRC_MAP[model_name],
        output=model_path,
        quiet=False,
    )


# ============================================================
# Dataset
# ============================================================

class H5PatchDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        patch_indices: Optional[List[int]] = None,
        img_key_candidates: Tuple[str, ...] = ("img", "imgs", "images"),
        barcode_key_candidates: Tuple[str, ...] = ("barcode", "barcodes"),
    ):
        self.h5_path = h5_path
        self.img_key_candidates = img_key_candidates
        self.barcode_key_candidates = barcode_key_candidates

        with h5py.File(self.h5_path, "r") as f:
            self.img_key = self._find_key(f, self.img_key_candidates)
            self.barcode_key = self._find_key(f, self.barcode_key_candidates)
            self.has_coords = "coords" in f

            n = f[self.img_key].shape[0]
            self.patch_indices = list(range(n)) if patch_indices is None else patch_indices

    @staticmethod
    def _find_key(f: h5py.File, candidates: Iterable[str]) -> str:
        for k in candidates:
            if k in f:
                return k
        raise KeyError(f"None of keys found: {candidates}")

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        patch_idx = self.patch_indices[i]

        with h5py.File(self.h5_path, "r") as f:
            img = f[self.img_key][patch_idx]
            barcode_raw = f[self.barcode_key][patch_idx]
            coord = f["coords"][patch_idx] if self.has_coords else None

        return {
            "patch_idx": int(patch_idx),
            "barcode": _decode_scalar(barcode_raw),
            "coord": None if coord is None else np.asarray(coord),
            "image": ensure_uint8_rgb(img),
        }


def collate_patches(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [b["image"] for b in batch],
        "patch_idx": [b["patch_idx"] for b in batch],
        "barcodes": [b["barcode"] for b in batch],
        "coords": [b["coord"] for b in batch],
    }


# ============================================================
# CellViT Adapter
# ============================================================

class CellViTInferenceAdapter:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        output_dir: str,
        runtime_dir: Optional[str] = None,
        device: str = "cuda:0",
        verbose: bool = False,
    ):
        self._patch_ray_init()

        from cellvit.detect_cells import CellViTInference, SystemConfiguration

        self.model_path = model_path
        self.model_name = infer_cellvit_model_type(model_name)
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose

        self._tmp_root = runtime_dir or os.path.join(output_dir, "_cellvit_runtime")
        os.makedirs(self._tmp_root, exist_ok=True)

        self.runner = self._build_runner(CellViTInference, SystemConfiguration)

        debug_print(
            self.verbose,
            "[DEBUG] runner callables:",
            [
                name
                for name in dir(self.runner)
                if not name.startswith("_") and callable(getattr(self.runner, name))
            ],
        )

    def _patch_ray_init(self):
        import ray

        if getattr(ray, "_patched_by_h5radiomics", False):
            return

        original_init = ray.init

        def patched_init(*args, **kwargs):
            kwargs.setdefault("ignore_reinit_error", True)
            kwargs.setdefault("include_dashboard", False)
            kwargs.setdefault("_memory", 32 * 1024**3)
            kwargs.setdefault("object_store_memory", 8 * 1024**3)

            print("[INFO] patched ray.init called")
            return original_init(*args, **kwargs)

        ray.init = patched_init
        ray._patched_by_h5radiomics = True

    def _build_system_configuration(self, SystemConfiguration):
        sig = inspect.signature(SystemConfiguration)
        kwargs = {}

        for name, param in sig.parameters.items():
            if name == "device":
                kwargs[name] = self.device

            elif name in ("gpu", "gpu_id", "gpu_ids"):
                if "cuda:" in self.device:
                    gpu_id = int(self.device.split(":")[-1])
                else:
                    gpu_id = 0
                kwargs[name] = [gpu_id] if name == "gpu_ids" else gpu_id

            elif name in ("mixed_precision", "amp"):
                kwargs[name] = False

            elif name == "enforce_mixed_precision":
                kwargs[name] = False

            elif name == "batch_size":
                kwargs[name] = 1

            elif name == "num_workers":
                kwargs[name] = 0

            elif name == "seed":
                kwargs[name] = 42

            elif name == "verbose":
                kwargs[name] = False

            elif param.default is not inspect._empty:
                pass

        try:
            return SystemConfiguration(**kwargs)
        except Exception:
            return SystemConfiguration()

    def _build_runner(self, CellViTInference, SystemConfiguration):
        system_configuration = self._build_system_configuration(SystemConfiguration)
        sig = inspect.signature(CellViTInference)

        kwargs = {}

        for name, param in sig.parameters.items():
            if name == "model_path":
                kwargs[name] = self.model_path

            elif name == "model_name":
                kwargs[name] = self.model_name

            elif name == "outdir":
                kwargs[name] = self._tmp_root

            elif name == "system_configuration":
                kwargs[name] = system_configuration

            elif name == "device":
                kwargs[name] = self.device

            elif param.default is not inspect._empty:
                pass

        return CellViTInference(**kwargs)

    def _try_call_method(self, method_names: List[str], *args, **kwargs):
        last_err = None

        for name in method_names:
            fn = getattr(self.runner, name, None)
            if fn is None:
                continue

            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise last_err

        return None

    def _predict_single_raw(self, image: np.ndarray):
        import cv2

        pil = Image.fromarray(image)
        x = self.runner.inference_transforms(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.runner.model(x)

        out = self.runner.apply_softmax_reorder(out)

        debug_print(
            self.verbose,
            "[DEBUG] model output keys:",
            out.keys() if isinstance(out, dict) else type(out),
        )

        type_map = self._extract_type_map_from_output(out)

        if isinstance(out, dict) and "instance_map" in out:
            inst_map = out["instance_map"][0]

            if torch.is_tensor(inst_map):
                inst_map = inst_map.detach().cpu().numpy()

            inst_map = np.asarray(inst_map, dtype=np.int32)

            return {
                "instance_map": inst_map,
                "type_map": type_map,
            }

        if isinstance(out, dict) and "nuclei_binary_map" in out:
            bin_map = out["nuclei_binary_map"][0]

            if torch.is_tensor(bin_map):
                arr = bin_map.detach().cpu().numpy()
            else:
                arr = np.asarray(bin_map)

            arr = np.asarray(arr)

            if arr.ndim == 3:
                if arr.shape[0] in (2, 3, 4):
                    pred = np.argmax(arr, axis=0)
                elif arr.shape[-1] in (2, 3, 4):
                    pred = np.argmax(arr, axis=-1)
                else:
                    raise RuntimeError(f"Unexpected nuclei_binary_map shape: {arr.shape}")

            elif arr.ndim == 2:
                pred = arr

            else:
                raise RuntimeError(f"Unexpected nuclei_binary_map ndim: {arr.ndim}")

            pred = np.asarray(pred)
            pred = np.squeeze(pred)

            if pred.ndim != 2:
                raise RuntimeError(f"pred must be 2D, got shape={pred.shape}")

            fg = (pred > 0).astype(np.uint8)
            fg = np.ascontiguousarray(fg)

            num_labels, labels = cv2.connectedComponents(fg)
            debug_print(self.verbose, "[DEBUG] connected components:", num_labels - 1)

            labels = np.asarray(labels, dtype=np.int32)

            return {
                "instance_map": labels,
                "type_map": type_map,
            }

        raise RuntimeError(
            f"Unknown output format. keys={list(out.keys()) if isinstance(out, dict) else type(out)}"
        )

    def _predict_batch_raw(self, images: List[np.ndarray]):
        result = self._try_call_method(
            ["predict_batch", "process_batch", "infer_batch", "run_batch"],
            images,
        )

        if result is not None:
            return result

        pil_images = [Image.fromarray(img) for img in images]

        result = self._try_call_method(
            ["predict_batch", "process_batch", "infer_batch", "run_batch"],
            pil_images,
        )

        if result is not None:
            return result

        return None

    def _extract_gdf_from_any(self, raw: Any) -> Optional[gpd.GeoDataFrame]:
        if raw is None:
            return None

        if isinstance(raw, gpd.GeoDataFrame):
            return raw

        if isinstance(raw, str) and raw.endswith((".geojson", ".json")) and os.path.exists(raw):
            return gpd.read_file(raw)

        if isinstance(raw, dict):
            for k in ["gdf", "geojson_gdf", "cells_gdf"]:
                if k in raw and isinstance(raw[k], gpd.GeoDataFrame):
                    return raw[k]

            for k in ["geojson_path", "cells_geojson", "cells_path", "output_geojson"]:
                if k in raw and isinstance(raw[k], str) and os.path.exists(raw[k]):
                    return gpd.read_file(raw[k])

            for k in ["geometry", "geometries"]:
                if k in raw and isinstance(raw[k], list):
                    geoms = raw[k]
                    return gpd.GeoDataFrame(
                        {"cell_id_in_patch": list(range(len(geoms)))},
                        geometry=geoms,
                        crs=None,
                    )

        return None

    def _extract_instance_map_from_any(self, raw: Any) -> Optional[np.ndarray]:
        if raw is None:
            return None

        if isinstance(raw, np.ndarray):
            raw = np.asarray(raw)

            if raw.ndim == 2:
                return raw.astype(np.int32)

            if raw.ndim == 3 and 1 in raw.shape:
                raw = np.squeeze(raw)
                if raw.ndim == 2:
                    return raw.astype(np.int32)

        if torch.is_tensor(raw):
            raw = raw.detach().cpu().numpy()

            if raw.ndim == 2:
                return raw.astype(np.int32)

            if raw.ndim == 3 and 1 in raw.shape:
                raw = np.squeeze(raw)
                if raw.ndim == 2:
                    return raw.astype(np.int32)

        if isinstance(raw, dict):
            for k in [
                "instance_map",
                "instance_maps",
                "inst_map",
                "inst_maps",
                "nuclei_instance_map",
                "nuclei_map",
            ]:
                if k in raw:
                    v = raw[k]

                    if torch.is_tensor(v):
                        v = v.detach().cpu().numpy()

                    v = np.asarray(v)

                    if v.ndim == 2:
                        return v.astype(np.int32)

                    if v.ndim == 3 and v.shape[0] == 1:
                        return v[0].astype(np.int32)

                    if v.ndim == 3 and v.shape[-1] == 1:
                        return v[..., 0].astype(np.int32)

                    if v.ndim == 3:
                        return v.astype(np.int32)

        return None

    def _extract_type_map_from_output(self, out: Any) -> Optional[np.ndarray]:
        if not isinstance(out, dict) or "nuclei_type_map" not in out:
            return None

        type_map = out["nuclei_type_map"][0]

        if torch.is_tensor(type_map):
            type_map = type_map.detach().cpu().numpy()
        else:
            type_map = np.asarray(type_map)

        if type_map.ndim == 3:
            if type_map.shape[0] <= 10:
                type_map = np.argmax(type_map, axis=0)
            elif type_map.shape[-1] <= 10:
                type_map = np.argmax(type_map, axis=-1)
            else:
                raise RuntimeError(f"Unexpected nuclei_type_map shape: {type_map.shape}")

        elif type_map.ndim != 2:
            raise RuntimeError(f"Unexpected nuclei_type_map ndim: {type_map.ndim}")

        return type_map.astype(np.int32)

    def _instance_majority_type(
        self,
        inst_mask: np.ndarray,
        type_map: Optional[np.ndarray],
    ) -> int:
        if type_map is None:
            return -1

        vals = type_map[inst_mask > 0]
        vals = vals[vals > 0]

        if vals.size == 0:
            return -1

        counts = np.bincount(vals.astype(np.int32))
        return int(np.argmax(counts))

    def _raw_to_gdf(self, raw: Any) -> gpd.GeoDataFrame:
        gdf = self._extract_gdf_from_any(raw)

        if gdf is not None:
            if "geometry" not in gdf.columns:
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=None)

            return gdf

        inst_map = None
        type_map = None

        if isinstance(raw, dict):
            inst_map = self._extract_instance_map_from_any(raw)
            type_map = raw.get("type_map", None)
        else:
            inst_map = self._extract_instance_map_from_any(raw)

        if inst_map is not None:
            rows = []

            instance_ids = np.unique(inst_map)
            instance_ids = instance_ids[instance_ids > 0]

            class_name_map = {
                0: "background",
                1: "neoplastic",
                2: "inflammatory",
                3: "connective",
                4: "dead",
                5: "epithelial",
            }

            for inst_id in instance_ids:
                mask = inst_map == inst_id
                class_id = self._instance_majority_type(mask.astype(np.uint8), type_map)
                class_name = class_name_map.get(class_id, f"class_{class_id}")

                polys = polygon_from_mask(mask)

                for poly in polys:
                    rows.append(
                        {
                            "cell_id_in_patch": int(inst_id),
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "geometry": poly,
                        }
                    )

            if rows:
                return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)

            return gpd.GeoDataFrame(
                {"cell_id_in_patch": [], "class_id": [], "class_name": []},
                geometry=[],
                crs=None,
            )

        raise RuntimeError(
            f"Unsupported CellViT output type: {type(raw)}. "
            "Need GeoDataFrame, geojson path, or instance map-like output."
        )

    def predict_batch_to_gdfs(self, images: List[np.ndarray]) -> List[gpd.GeoDataFrame]:
        raw_batch = self._predict_batch_raw(images)

        if raw_batch is not None:
            if isinstance(raw_batch, list):
                return [self._raw_to_gdf(x) for x in raw_batch]

            if isinstance(raw_batch, tuple):
                return [self._raw_to_gdf(x) for x in list(raw_batch)]

            if isinstance(raw_batch, dict):
                inst = self._extract_instance_map_from_any(raw_batch)

                if inst is not None and inst.ndim == 3:
                    return [self._raw_to_gdf(m) for m in inst]

        return [self._raw_to_gdf(self._predict_single_raw(img)) for img in images]


# ============================================================
# H5 save/load for segmentation results
# ============================================================

def save_cellseg_h5(
    rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    save_path: str,
) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(save_path, "w") as f:
        cells_grp = f.create_group("cells")
        patches_grp = f.create_group("patches")

        if rows:
            cells_grp.create_dataset(
                "patch_idx",
                data=np.asarray([r["patch_idx"] for r in rows], dtype=np.int64),
                compression="gzip",
            )
            cells_grp.create_dataset(
                "barcode",
                data=np.asarray([r["barcode"] for r in rows], dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )
            cells_grp.create_dataset(
                "cell_id_in_patch",
                data=np.asarray([r["cell_id_in_patch"] for r in rows], dtype=np.int64),
                compression="gzip",
            )
            cells_grp.create_dataset(
                "class_id",
                data=np.asarray([r["class_id"] for r in rows], dtype=np.int64),
                compression="gzip",
            )
            cells_grp.create_dataset(
                "class_name",
                data=np.asarray([r["class_name"] for r in rows], dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )
            cells_grp.create_dataset(
                "geometry_wkt",
                data=np.asarray([r["geometry"].wkt for r in rows], dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )
        else:
            cells_grp.create_dataset("patch_idx", data=np.asarray([], dtype=np.int64))
            cells_grp.create_dataset("barcode", data=np.asarray([], dtype=object), dtype=str_dtype)
            cells_grp.create_dataset("cell_id_in_patch", data=np.asarray([], dtype=np.int64))
            cells_grp.create_dataset("class_id", data=np.asarray([], dtype=np.int64))
            cells_grp.create_dataset("class_name", data=np.asarray([], dtype=object), dtype=str_dtype)
            cells_grp.create_dataset("geometry_wkt", data=np.asarray([], dtype=object), dtype=str_dtype)

        patches_grp.create_dataset(
            "patch_idx",
            data=np.asarray([r["patch_idx"] for r in summary_rows], dtype=np.int64),
            compression="gzip",
        )
        patches_grp.create_dataset(
            "barcode",
            data=np.asarray([r["barcode"] for r in summary_rows], dtype=object),
            dtype=str_dtype,
            compression="gzip",
        )
        patches_grp.create_dataset(
            "n_cells",
            data=np.asarray([r["n_cells"] for r in summary_rows], dtype=np.int64),
            compression="gzip",
        )

        if summary_rows and "coord_raw" in summary_rows[0]:
            coord_json = [
                json.dumps(r["coord_raw"], ensure_ascii=False)
                for r in summary_rows
            ]
            patches_grp.create_dataset(
                "coord_raw_json",
                data=np.asarray(coord_json, dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )

    return save_path


def load_cellseg_h5(seg_h5_path: str) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    with h5py.File(seg_h5_path, "r") as f:
        cells = f["cells"]
        patches = f["patches"]

        cell_rows = []

        n_cells = len(cells["patch_idx"])

        for i in range(n_cells):
            cell_rows.append(
                {
                    "patch_idx": int(cells["patch_idx"][i]),
                    "barcode": _decode_scalar(cells["barcode"][i]),
                    "cell_id_in_patch": int(cells["cell_id_in_patch"][i]),
                    "class_id": int(cells["class_id"][i]),
                    "class_name": _decode_scalar(cells["class_name"][i]),
                    "geometry": wkt.loads(_decode_scalar(cells["geometry_wkt"][i])),
                }
            )

        patch_rows = []

        n_patches = len(patches["patch_idx"])

        for i in range(n_patches):
            row = {
                "patch_idx": int(patches["patch_idx"][i]),
                "barcode": _decode_scalar(patches["barcode"][i]),
                "n_cells": int(patches["n_cells"][i]),
            }

            if "coord_raw_json" in patches:
                coord_raw = _decode_scalar(patches["coord_raw_json"][i])
                row["coord_raw"] = json.loads(coord_raw)

            patch_rows.append(row)

    if cell_rows:
        seg_gdf = gpd.GeoDataFrame(cell_rows, geometry="geometry", crs=None)
    else:
        seg_gdf = gpd.GeoDataFrame(
            {
                "patch_idx": [],
                "barcode": [],
                "cell_id_in_patch": [],
                "class_id": [],
                "class_name": [],
            },
            geometry=[],
            crs=None,
        )

    patch_df = pd.DataFrame(patch_rows)

    return seg_gdf, patch_df


# ============================================================
# Segmentation only
# ============================================================

def _gdf_to_cell_rows(
    gdf: gpd.GeoDataFrame,
    patch_idx: int,
    barcode: str,
) -> List[Dict[str, Any]]:
    rows = []

    if len(gdf) == 0:
        return rows

    gdf = gdf.copy()

    if "cell_id_in_patch" not in gdf.columns:
        gdf["cell_id_in_patch"] = list(range(1, len(gdf) + 1))

    for _, r in gdf.iterrows():
        rows.append(
            {
                "patch_idx": int(patch_idx),
                "barcode": barcode,
                "cell_id_in_patch": int(r["cell_id_in_patch"]),
                "class_id": (
                    int(r["class_id"])
                    if "class_id" in gdf.columns and pd.notna(r["class_id"])
                    else -1
                ),
                "class_name": (
                    str(r["class_name"])
                    if "class_name" in gdf.columns and pd.notna(r["class_name"])
                    else "unknown"
                ),
                "geometry": r.geometry,
            }
        )

    return rows


def segment_h5_patches_with_cellvit(
    h5_path: str,
    seg_h5_path: str,
    model_path: str,
    runtime_dir: str,
    summary_json_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    patch_indices: Optional[List[int]] = None,
    device: str = "cuda:0",
    predictor: Optional[CellViTInferenceAdapter] = None,
) -> str:
    """
    Segment H5 patches with CellViT and save only H5 result.

    Output:
      segment/
        ├── {sample_id}.h5
        ├── {sample_id}.summary.json
        └── _cellvit_runtime/
    """
    os.makedirs(os.path.dirname(seg_h5_path), exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)

    dataset = H5PatchDataset(
        h5_path=h5_path,
        patch_indices=patch_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patches,
        pin_memory=torch.cuda.is_available(),
    )

    if predictor is None:
        predictor = CellViTInferenceAdapter(
            model_path=model_path,
            model_name=os.path.basename(model_path),
            output_dir=os.path.dirname(seg_h5_path),
            runtime_dir=runtime_dir,
            device=device,
        )

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(dataset), desc="Segmenting patches")

    for batch in loader:
        images = batch["images"]
        patch_idxs = batch["patch_idx"]
        barcodes = batch["barcodes"]
        coords = batch["coords"]

        gdfs = predictor.predict_batch_to_gdfs(images)

        for gdf, patch_idx, barcode, coord in zip(gdfs, patch_idxs, barcodes, coords):
            rows = _gdf_to_cell_rows(
                gdf=gdf,
                patch_idx=patch_idx,
                barcode=barcode,
            )

            all_rows.extend(rows)

            summary_rows.append(
                {
                    "patch_idx": int(patch_idx),
                    "barcode": barcode,
                    "coord_raw": None if coord is None else np.asarray(coord).tolist(),
                    "n_cells": int(len(gdf)),
                }
            )

            pbar.update(1)

    pbar.close()

    save_cellseg_h5(
        rows=all_rows,
        summary_rows=summary_rows,
        save_path=seg_h5_path,
    )

    summary = {
        "h5_path": h5_path,
        "segment_dir": os.path.dirname(seg_h5_path),
        "runtime_dir": runtime_dir,
        "num_patches": len(dataset),
        "num_polygons": int(len(all_rows)),
        "seg_h5_path": seg_h5_path,
    }

    if summary_json_path is not None:
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[INFO] segmentation done")
    print(json.dumps(summary, indent=2))

    return seg_h5_path


# ============================================================
# Overlay only
# ============================================================

def save_overlay_png(
    img: np.ndarray,
    gdf: gpd.GeoDataFrame,
    save_path: str,
    title: Optional[str] = None,
    use_class_color: bool = True,
):
    img = ensure_uint8_rgb(img)

    class_color_map = {
        "neoplastic": "#ff4d4d",
        "inflammatory": "#4da6ff",
        "connective": "#00c853",
        "dead": "#ffd54f",
        "epithelial": "#ab47bc",
        "background": "#9e9e9e",
        "unknown": "#00ffff",
    }
    default_color = "#00ffff"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if use_class_color:
        if len(gdf) > 0:
            for cls_name, sub_gdf in gdf.groupby("class_name"):
                color = class_color_map.get(str(cls_name).lower(), default_color)

                patches = []

                for geom in sub_gdf.geometry:
                    for poly in iter_polygons(geom):
                        xy = np.asarray(poly.exterior.coords)

                        if len(xy) >= 3:
                            patches.append(MplPolygon(xy, closed=True))

                if patches:
                    collection = PatchCollection(
                        patches,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.0,
                    )
                    ax.add_collection(collection)

    else:
        patches = []

        for geom in gdf.geometry:
            for poly in iter_polygons(geom):
                xy = np.asarray(poly.exterior.coords)

                if len(xy) >= 3:
                    patches.append(MplPolygon(xy, closed=True))

        if patches:
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor="none",
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )

    if use_class_color:
        handles = [
            Line2D([0], [0], color=color, lw=2, label=cls)
            for cls, color in class_color_map.items()
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=8)

    if title:
        ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)


def save_overlays_from_cellseg_h5(
    source_h5_path: str,
    seg_h5_path: str,
    overlay_dir: str,
    use_class_color: bool = True,
    patch_indices: Optional[List[int]] = None,
):
    """
    Load segmentation result from segment/{sample_id}.h5 and save overlay PNGs.

    This function does not run CellViT again.
    """
    os.makedirs(overlay_dir, exist_ok=True)

    seg_gdf, patch_df = load_cellseg_h5(seg_h5_path)

    if patch_indices is not None:
        patch_df = patch_df[patch_df["patch_idx"].isin(patch_indices)].copy()
        seg_gdf = seg_gdf[seg_gdf["patch_idx"].isin(patch_indices)].copy()

    dataset = H5PatchDataset(
        h5_path=source_h5_path,
        patch_indices=patch_df["patch_idx"].astype(int).tolist(),
    )

    gdf_by_patch = {
        int(patch_idx): sub_gdf.copy()
        for patch_idx, sub_gdf in seg_gdf.groupby("patch_idx")
    }

    for i in tqdm(range(len(dataset)), desc="Saving overlays"):
        sample = dataset[i]

        patch_idx = int(sample["patch_idx"])
        barcode = str(sample["barcode"])
        img = sample["image"]

        gdf = gdf_by_patch.get(
            patch_idx,
            gpd.GeoDataFrame(
                {"cell_id_in_patch": [], "class_id": [], "class_name": []},
                geometry=[],
                crs=None,
            ),
        )

        save_path = os.path.join(
            overlay_dir,
            f"{barcode}_idx{patch_idx}.png",
        )

        save_overlay_png(
            img=img,
            gdf=gdf,
            save_path=save_path,
            title=f"idx={patch_idx}, bc={barcode}, cells={len(gdf)}",
            use_class_color=use_class_color,
        )

    print(f"[INFO] overlays saved to {overlay_dir}")


# ============================================================
# Multi-oncotree wrappers
# ============================================================

def list_sample_ids_from_patches(oncotree_root: str) -> List[str]:
    patches_dir = os.path.join(oncotree_root, "patches")

    if not os.path.isdir(patches_dir):
        print(f"[WARN] patches dir not found: {patches_dir}")
        return []

    sample_ids = []

    for filename in sorted(os.listdir(patches_dir)):
        if filename.endswith(".h5"):
            sample_ids.append(os.path.splitext(filename)[0])

    return sample_ids


def build_sample_paths(
    hest_root: str,
    oncotree: str,
    sample_id: str,
) -> Dict[str, str]:
    data_root = os.path.join(hest_root, oncotree)

    segment_dir = os.path.join(data_root, "segment")
    segment_vis_dir = os.path.join(data_root, "segment_vis")

    return {
        "data_root": data_root,
        "patch_h5_path": os.path.join(data_root, "patches", f"{sample_id}.h5"),
        "segment_dir": segment_dir,
        "segment_vis_dir": segment_vis_dir,
        "seg_h5_path": os.path.join(segment_dir, f"{sample_id}.h5"),
        "summary_json_path": os.path.join(segment_dir, f"{sample_id}.summary.json"),
        "runtime_dir": os.path.join(segment_dir, "_cellvit_runtime"),
        "overlay_dir": os.path.join(segment_vis_dir, sample_id),
    }


def segment_one_sample(
    hest_root: str,
    oncotree: str,
    sample_id: str,
    model_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Optional[str]:
    paths = build_sample_paths(
        hest_root=hest_root,
        oncotree=oncotree,
        sample_id=sample_id,
    )

    patch_h5_path = paths["patch_h5_path"]
    seg_h5_path = paths["seg_h5_path"]

    if not os.path.exists(patch_h5_path):
        print(f"[WARN] patch h5 not found: {patch_h5_path}")
        return None

    if os.path.exists(seg_h5_path) and not overwrite:
        print(f"[SKIP] segment exists: {seg_h5_path}")
        return seg_h5_path

    print("=" * 80)
    print(f"[SEGMENT] oncotree={oncotree}, sample_id={sample_id}")
    print(f"[INPUT]   {patch_h5_path}")
    print(f"[OUTPUT]  {seg_h5_path}")
    print("=" * 80)

    return segment_h5_patches_with_cellvit(
        h5_path=patch_h5_path,
        seg_h5_path=seg_h5_path,
        model_path=model_path,
        runtime_dir=paths["runtime_dir"],
        summary_json_path=paths["summary_json_path"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def save_overlay_one_sample(
    hest_root: str,
    oncotree: str,
    sample_id: str,
    use_class_color: bool = True,
    overwrite: bool = False,
) -> Optional[str]:
    paths = build_sample_paths(
        hest_root=hest_root,
        oncotree=oncotree,
        sample_id=sample_id,
    )

    patch_h5_path = paths["patch_h5_path"]
    seg_h5_path = paths["seg_h5_path"]
    overlay_dir = paths["overlay_dir"]

    if not os.path.exists(patch_h5_path):
        print(f"[WARN] patch h5 not found: {patch_h5_path}")
        return None

    if not os.path.exists(seg_h5_path):
        print(f"[WARN] segment h5 not found: {seg_h5_path}")
        return None

    if os.path.isdir(overlay_dir) and len(os.listdir(overlay_dir)) > 0 and not overwrite:
        print(f"[SKIP] overlay exists: {overlay_dir}")
        return overlay_dir

    print("=" * 80)
    print(f"[OVERLAY] oncotree={oncotree}, sample_id={sample_id}")
    print(f"[SEG]     {seg_h5_path}")
    print(f"[OUTPUT]  {overlay_dir}")
    print("=" * 80)

    save_overlays_from_cellseg_h5(
        source_h5_path=patch_h5_path,
        seg_h5_path=seg_h5_path,
        overlay_dir=overlay_dir,
        use_class_color=use_class_color,
    )

    return overlay_dir


def segment_all_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    model_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> List[str]:
    output_paths = []

    for oncotree in oncotrees:
        oncotree_root = os.path.join(hest_root, oncotree)
        sample_ids = list_sample_ids_from_patches(oncotree_root)

        print(f"[INFO] {oncotree}: {len(sample_ids)} samples")

        for sample_id in sample_ids:
            seg_h5_path = segment_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                model_path=model_path,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                overwrite=overwrite,
            )

            if seg_h5_path is not None:
                output_paths.append(seg_h5_path)

    return output_paths


def save_overlays_all_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    use_class_color: bool = True,
    overwrite: bool = False,
) -> List[str]:
    output_dirs = []

    for oncotree in oncotrees:
        oncotree_root = os.path.join(hest_root, oncotree)
        sample_ids = list_sample_ids_from_patches(oncotree_root)

        print(f"[INFO] {oncotree}: {len(sample_ids)} samples")

        for sample_id in sample_ids:
            overlay_dir = save_overlay_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                use_class_color=use_class_color,
                overwrite=overwrite,
            )

            if overlay_dir is not None:
                output_dirs.append(overlay_dir)

    return output_dirs


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    verify_or_download_model(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
    )

    segment_paths: List[str] = []
    overlay_dirs: List[str] = []

    if RUN_SEGMENT:
        segment_paths = segment_all_oncotrees(
            hest_root=HEST_ROOT,
            oncotrees=DOWNLOAD_ONCOTREE,
            model_path=MODEL_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=DEVICE,
            overwrite=OVERWRITE_SEGMENT,
        )

    if RUN_OVERLAY:
        overlay_dirs = save_overlays_all_oncotrees(
            hest_root=HEST_ROOT,
            oncotrees=DOWNLOAD_ONCOTREE,
            use_class_color=USE_CLASS_COLOR,
            overwrite=OVERWRITE_OVERLAY,
        )

    print("[INFO] all done")
    print(f"[INFO] segmented samples: {len(segment_paths)}")
    print(f"[INFO] overlay dirs: {len(overlay_dirs)}")

