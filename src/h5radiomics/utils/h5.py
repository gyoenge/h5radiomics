from __future__ import annotations

import h5py
import numpy as np


def get_img_key(f: h5py.File) -> str:
    for key in ("img", "imgs", "images"):
        if key in f:
            return key
    raise KeyError("img/imgs/images key를 찾지 못했습니다.")


def get_coords_key(f: h5py.File) -> str | None:
    return "coords" if "coords" in f else None


def get_barcodes_key(f: h5py.File) -> str | None:
    for key in ("barcodes", "barcode"):
        if key in f:
            return key
    return None


def to_str_barcode(barcode) -> str:
    if barcode is None:
        return ""
    if isinstance(barcode, np.ndarray) and barcode.shape:
        barcode = barcode[0]
    if isinstance(barcode, bytes):
        return barcode.decode("utf-8")
    return str(barcode)


def ensure_hwc(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img

    if img.ndim != 3:
        raise ValueError(f"Unsupported img ndim: {img.ndim}")

    if img.shape[-1] in (1, 3, 4):
        return img

    if img.shape[0] in (1, 3, 4):
        return np.transpose(img, (1, 2, 0))

    raise ValueError(f"Cannot infer image layout from shape: {img.shape}")


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    img = ensure_hwc(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


def load_h5_patches(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        imgs = f[img_key][:]
        coords = f[coords_key][:] if coords_key else None
        barcodes = [to_str_barcode(b) for b in f[barcodes_key][:]] if barcodes_key else None

    return imgs, coords, barcodes


# def inspect_h5_file(h5_path: str | Path) -> dict:
#     h5_path = Path(h5_path)

#     if not h5_path.exists():
#         raise FileNotFoundError(f"H5 file not found: {h5_path}")

#     summary = {
#         "datasets": {},
#         "groups": [],
#     }

#     with h5py.File(h5_path, "r") as f:
#         def visitor(name, obj):
#             if isinstance(obj, h5py.Group):
#                 summary["groups"].append(name)
#             elif isinstance(obj, h5py.Dataset):
#                 summary["datasets"][name] = {
#                     "shape": obj.shape,
#                     "dtype": str(obj.dtype),
#                 }

#         f.visititems(visitor)

#     return summary
