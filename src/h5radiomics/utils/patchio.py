from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os 
import numpy as np
from PIL import Image
from h5radiomics.utils.io import make_base_name
from h5radiomics.utils.h5 import (
    to_str_barcode,
)
from h5radiomics.utils.paths import (
    get_patch_color_dir,
    get_patch_gray_dir,
    get_patch_mask_dir,
    get_patch_masked_color_dir,
    get_patch_masked_gray_dir,
)


@dataclass
class PatchData:
    patch_idx: int
    color_patch: np.ndarray
    gray_patch: np.ndarray
    coords: Optional[Any]
    barcode: Optional[str]
    base_filename: str


# ------------------------------------------------------------------------------
# save helpers (I/O)
# ------------------------------------------------------------------------------

def save_patch_images_once(
    color_patch: np.ndarray,
    gray_patch: np.ndarray,
    output_dir: str,
    sample_id: str,
    base_filename: str,
):
    color_dir = get_patch_color_dir(output_dir, sample_id)
    gray_dir = get_patch_gray_dir(output_dir, sample_id)

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(gray_dir, exist_ok=True)

    color_path = f"{color_dir}/{base_filename}.png"
    gray_path = f"{gray_dir}/{base_filename}.png"

    if not os.path.exists(color_path):
        Image.fromarray(color_patch).save(color_path)
    if not os.path.exists(gray_path):
        Image.fromarray(gray_patch).save(gray_path)

    return color_path, gray_path


def save_region_mask_images(
    color_patch: np.ndarray,
    gray_patch: np.ndarray,
    mask_patch: np.ndarray,
    output_dir: str,
    sample_id: str,
    mask_filename: str,
):
    mask_dir = get_patch_mask_dir(output_dir, sample_id)
    masked_color_dir = get_patch_masked_color_dir(output_dir, sample_id)
    masked_gray_dir = get_patch_masked_gray_dir(output_dir, sample_id)

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_color_dir, exist_ok=True)
    os.makedirs(masked_gray_dir, exist_ok=True)

    mask_path = f"{mask_dir}/{mask_filename}.png"
    masked_color_path = f"{masked_color_dir}/{mask_filename}.png"
    masked_gray_path = f"{masked_gray_dir}/{mask_filename}.png"

    Image.fromarray(mask_patch).save(mask_path)

    mask_binary = (mask_patch > 0).astype(np.uint8)
    masked_color = color_patch * mask_binary[..., None]
    masked_gray = gray_patch * mask_binary

    Image.fromarray(masked_color.astype(np.uint8)).save(masked_color_path)
    Image.fromarray(masked_gray.astype(np.uint8)).save(masked_gray_path)

    return mask_path


# ------------------------------------------------------------------------------
# patch loading / row builders
# ------------------------------------------------------------------------------

def load_patch_data(
    f,
    img_key,
    coords_key,
    barcodes_key,
    patch_idx: int,
) -> PatchData:
    img = f[img_key][patch_idx]

    if img.ndim == 3 and img.shape[2] == 3:
        color_patch = img.astype(np.uint8)
    elif img.ndim == 3 and img.shape[0] == 3:
        color_patch = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape} for patch index {patch_idx}")

    gray_patch = np.array(Image.fromarray(color_patch).convert("L"))
    coords = f[coords_key][patch_idx] if coords_key else None
    barcode = f[barcodes_key][patch_idx] if barcodes_key else None
    barcode = to_str_barcode(barcode) if barcode is not None else None
    base_filename = make_base_name(patch_idx, barcode)

    return PatchData(
        patch_idx=patch_idx,
        color_patch=color_patch,
        gray_patch=gray_patch,
        coords=coords,
        barcode=barcode,
        base_filename=base_filename,
    )


def build_patch_row_base(
    patch: PatchData,
    output_dir: str,
    sample_id: str,
    save_patches: bool,
) -> Dict[str, Any]:
    color_path = ""
    gray_path = ""

    if save_patches:
        color_path, gray_path = save_patch_images_once(
            color_patch=patch.color_patch,
            gray_patch=patch.gray_patch,
            output_dir=output_dir,
            sample_id=sample_id,
            base_filename=patch.base_filename,
        )

    return {
        "patch_idx": patch.patch_idx,
        "barcode": patch.barcode,
        "color_path": color_path,
        "gray_path": gray_path,
        "mask_path": "",
        "x": patch.coords[0] if patch.coords is not None else None,
        "y": patch.coords[1] if patch.coords is not None else None,
        "status": "ok",
    }


