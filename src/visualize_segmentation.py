# visualize_segmentation.py
# Visualization of segmentation results for image patches stored in HDF5 files

# pip install h5py pandas pyarrow shapely matplotlib

import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely import wkb
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.affinity import translate
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# from utils import (
#     load_h5_patches,
#     show_patch,
# )

################################# 


# SEG_ROOT = "/root/workspace/h5radiomics/segmentation"
# PATCH_ROOT = "/root/workspace/h5radiomics/h5"
# SAMPLES = ["NCBI783", "NCBI785", "TENX95", "TENX99"]
OUTPUT_DIR = "/root/workspace/h5radiomics/output_test/vis_seg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

### 

# =========================
# 1. H5 로드
# =========================
def load_h5_metadata(h5_path):
    """
    h5 파일에서 coords, barcode, img shape 정보를 읽는다.
    img 전체를 한 번에 메모리에 올리지 않도록 dataset handle은 반환하지 않음.
    """
    with h5py.File(h5_path, "r") as f:
        if "img" not in f:
            raise KeyError("'img' key not found in h5 file")
        if "coords" not in f:
            raise KeyError("'coords' key not found in h5 file")
        if "barcode" not in f:
            raise KeyError("'barcode' key not found in h5 file")

        img_shape = f["img"].shape
        coords = f["coords"][:]
        raw_barcodes = f["barcode"][:]

    barcodes = []
    for b in raw_barcodes:
        if isinstance(b, np.ndarray):
            b = b[0]
        if isinstance(b, bytes):
            b = b.decode("utf-8")
        else:
            b = str(b)
        barcodes.append(b)

    return coords, barcodes, img_shape


def load_h5_patch(h5_path, patch_idx):
    """
    특정 patch image, coord, barcode를 읽는다.
    """
    with h5py.File(h5_path, "r") as f:
        img = f["img"][patch_idx]
        coord = f["coords"][patch_idx]
        barcode_raw = f["barcode"][patch_idx]

    if isinstance(barcode_raw, np.ndarray):
        barcode_raw = barcode_raw[0]
    barcode = barcode_raw.decode("utf-8") if isinstance(barcode_raw, bytes) else str(barcode_raw)

    return img, coord, barcode


# =========================
# 2. 이미지 shape 정리
# =========================
def ensure_hwc(img):
    """
    img를 HWC 형태로 맞춘다.
    지원:
    - HWC
    - CHW (C,H,W), C=1 or 3 or 4
    """
    if img.ndim == 2:
        return img

    if img.ndim != 3:
        raise ValueError(f"Unsupported img ndim: {img.ndim}")

    # 이미 HWC인 경우
    if img.shape[-1] in (1, 3, 4):
        return img

    # CHW인 경우
    if img.shape[0] in (1, 3, 4):
        return np.transpose(img, (1, 2, 0))

    raise ValueError(f"Cannot infer image format from shape: {img.shape}")


def get_patch_hw(img_hwc):
    if img_hwc.ndim == 2:
        return img_hwc.shape[0], img_hwc.shape[1]
    return img_hwc.shape[0], img_hwc.shape[1]


# =========================
# 3. parquet 로드
# =========================
def load_segmentation_parquet(parquet_path):
    """
    parquet의 geometry(WKB bytes)를 shapely geometry로 변환한다.
    """
    df = pd.read_parquet(parquet_path).copy()

    if "geometry" not in df.columns:
        raise KeyError("'geometry' column not found in parquet")

    df["geometry"] = df["geometry"].apply(wkb.loads)
    return df


# =========================
# 4. patch와 겹치는 cell 추출
# =========================
def get_cells_for_patch(seg_df, patch_coord, patch_w, patch_h):
    """
    patch_coord = (x0, y0) in WSI coordinates
    patch bbox와 intersect하는 geometry만 추출하고,
    patch local 좌표계로 이동한 geometry_local 컬럼을 추가한다.
    """
    x0, y0 = float(patch_coord[0]), float(patch_coord[1])
    patch_bbox = box(x0, y0, x0 + patch_w, y0 + patch_h)

    # 1) patch와 겹치는 후보만 추출
    intersects_mask = seg_df["geometry"].apply(lambda g: g.intersects(patch_bbox))
    sub = seg_df.loc[intersects_mask].copy()

    if len(sub) == 0:
        sub["geometry_local"] = pd.Series(dtype=object)
        return sub

    # 2) 실제 patch bbox로 clip
    clipped = sub["geometry"].apply(lambda g: g.intersection(patch_bbox))

    # 3) empty 제거
    valid_mask = clipped.apply(lambda g: not g.is_empty)
    sub = sub.loc[valid_mask].copy()
    clipped = clipped.loc[valid_mask]

    if len(sub) == 0:
        sub["geometry_local"] = pd.Series(dtype=object)
        return sub

    # 4) patch local 좌표계로 이동
    sub["geometry_local"] = clipped.apply(
        lambda g: translate(g, xoff=-x0, yoff=-y0)
    ).values

    print(f"[DEBUG] total cells: {len(seg_df)}")
    print(f"[DEBUG] intersected cells: {intersects_mask.sum()}")

    return sub


# =========================
# 5. shapely geometry -> matplotlib patch
# =========================
def geometry_to_mpl_patches(geom):
    """
    shapely Polygon / MultiPolygon 을 matplotlib patch 리스트로 변환
    interior hole은 일단 외곽선만 그림
    """
    patches = []

    if geom.is_empty:
        return patches

    if isinstance(geom, Polygon):
        exterior = np.asarray(geom.exterior.coords)
        if len(exterior) >= 3:
            patches.append(MplPolygon(exterior, closed=True))

    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            exterior = np.asarray(poly.exterior.coords)
            if len(exterior) >= 3:
                patches.append(MplPolygon(exterior, closed=True))

    return patches


# =========================
# 6. overlay 시각화
# =========================
def visualize_patch_overlay(
    img,
    matched_df,
    class_col="class",
    geometry_col="geometry_local",
    figsize=(8, 8),
    title=None,
    show_boundary=True,
    boundary_color="lime",
    boundary_linewidth=0.8,
    show_fill=False,
    fill_alpha=0.25,
):
    """
    img 위에 segmentation polygon overlay
    """
    img_hwc = ensure_hwc(img)

    fig, ax = plt.subplots(figsize=figsize)

    if img_hwc.ndim == 2:
        ax.imshow(img_hwc, cmap="gray")
    else:
        # float이면 0~1 또는 적당히 clip
        if np.issubdtype(img_hwc.dtype, np.floating):
            show_img = np.clip(img_hwc, 0, 1)
        else:
            show_img = img_hwc
        ax.imshow(show_img)

    all_patches = []
    for geom in matched_df[geometry_col]:
        all_patches.extend(geometry_to_mpl_patches(geom))

    if len(all_patches) > 0:
        if show_fill:
            fill_collection = PatchCollection(
                all_patches,
                facecolor="red",
                edgecolor="none",
                alpha=fill_alpha,
                zorder=2,
            )
            ax.add_collection(fill_collection)

        if show_boundary:
            edge_collection = PatchCollection(
                all_patches,
                facecolor="none",
                edgecolor=boundary_color,
                linewidth=boundary_linewidth,
                zorder=3,
            )
            ax.add_collection(edge_collection)

    if title is not None:
        ax.set_title(title)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def visualize_patch_only(img, title=None):
    img_hwc = ensure_hwc(img)

    plt.figure(figsize=(6, 6))

    if img_hwc.ndim == 2:
        plt.imshow(img_hwc, cmap="gray")
    else:
        if np.issubdtype(img_hwc.dtype, np.floating):
            img_show = np.clip(img_hwc, 0, 1)
        else:
            img_show = img_hwc
        plt.imshow(img_show)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

def save_patch_overlay(
    img,
    matched_df,
    save_path,
    class_col="class",
    geometry_col="geometry_local",
    figsize=(8, 8),
    title=None,
    show_boundary=True,
    boundary_color="lime",
    boundary_linewidth=0.8,
    show_fill=False,
    fill_alpha=0.25,
):
    img_hwc = ensure_hwc(img)

    fig, ax = plt.subplots(figsize=figsize)

    if img_hwc.ndim == 2:
        ax.imshow(img_hwc, cmap="gray")
    else:
        if np.issubdtype(img_hwc.dtype, np.floating):
            show_img = np.clip(img_hwc, 0, 1)
        else:
            show_img = img_hwc
        ax.imshow(show_img)

    all_patches = []
    for geom in matched_df[geometry_col]:
        all_patches.extend(geometry_to_mpl_patches(geom))

    if len(all_patches) > 0:
        if show_fill:
            fill_collection = PatchCollection(
                all_patches,
                facecolor="red",
                edgecolor="none",
                alpha=fill_alpha,
                zorder=2,
            )
            ax.add_collection(fill_collection)

        if show_boundary:
            edge_collection = PatchCollection(
                all_patches,
                facecolor="none",
                edgecolor=boundary_color,
                linewidth=boundary_linewidth,
                zorder=3,
            )
            ax.add_collection(edge_collection)

    if title:
        ax.set_title(title)

    ax.set_axis_off()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close(fig)   # 중요 (메모리 누수 방지)

def save_patch_only(img, save_path, title=None):
    img_hwc = ensure_hwc(img)

    fig = plt.figure(figsize=(6, 6))

    if img_hwc.ndim == 2:
        plt.imshow(img_hwc, cmap="gray")
    else:
        if np.issubdtype(img_hwc.dtype, np.floating):
            img_show = np.clip(img_hwc, 0, 1)
        else:
            img_show = img_hwc
        plt.imshow(img_show)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================
# 7. barcode 또는 index로 patch 찾기
# =========================
def find_patch_index_by_barcode(barcodes, target_barcode):
    for i, bc in enumerate(barcodes):
        if bc == target_barcode:
            return i
    raise ValueError(f"barcode not found: {target_barcode}")


# =========================
# 8. 한 번에 실행하는 함수
# =========================
def visualize_h5_patch_with_segmentation(
    h5_path,
    parquet_path,
    patch_idx=None,
    target_barcode=None,
    figsize=(8, 8),
    max_cells_print=10,
):
    """
    patch_idx 또는 target_barcode 중 하나로 patch 선택
    """
    if patch_idx is None and target_barcode is None:
        raise ValueError("Either patch_idx or target_barcode must be provided")

    coords, barcodes, _ = load_h5_metadata(h5_path)

    if target_barcode is not None:
        patch_idx = find_patch_index_by_barcode(barcodes, target_barcode)

    img, coord, barcode = load_h5_patch(h5_path, patch_idx)
    img_hwc = ensure_hwc(img)
    patch_h, patch_w = get_patch_hw(img_hwc)

    print(f"[INFO] patch_idx : {patch_idx}")
    print(f"[INFO] barcode   : {barcode}")
    print(f"[INFO] coord     : {coord}")
    print(f"[INFO] img shape : {img_hwc.shape}")

    seg_df = load_segmentation_parquet(parquet_path)
    matched_df = get_cells_for_patch(seg_df, coord, patch_w, patch_h)

    print(f"[INFO] matched cells: {len(matched_df)}")
    if len(matched_df) > 0:
        cols_to_show = [c for c in ["class", "cell_id"] if c in matched_df.columns]
        print(matched_df[cols_to_show].head(max_cells_print))

    title = f"patch_idx={patch_idx}, barcode={barcode}, matched_cells={len(matched_df)}"

    filename = f"{barcode}_idx{patch_idx}_cells{len(matched_df)}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)

    if len(matched_df) == 0:
        print("[WARN] No matched cells → saving patch only")
        save_patch_only(
            img_hwc,
            save_path,
            title=f"{barcode} (NO CELLS)"
        )
    else:
        save_patch_overlay(
            img=img_hwc,
            matched_df=matched_df,
            save_path=save_path,
            title=title,
            show_boundary=True,
            boundary_color="lime",
            boundary_linewidth=0.7,
            show_fill=False,
        )

    print(f"[INFO] saved to: {save_path}")

    return {
        "patch_idx": patch_idx,
        "barcode": barcode,
        "coord": coord,
        "img": img_hwc,
        "matched_df": matched_df,
    }


# =========================
# 9. 사용 예시
# =========================
if __name__ == "__main__":
    h5_path = "/root/workspace/h5radiomics/h5/TENX99.h5"
    parquet_path = "/root/workspace/h5radiomics/segmentation/TENX99_cellvit_seg.parquet"

    # 방법 1) patch index로 보기
    result = visualize_h5_patch_with_segmentation(
        h5_path=h5_path,
        parquet_path=parquet_path,
        patch_idx=400,  # trials: 0 
        figsize=(8, 8),
    )

    # 방법 2) barcode로 보기
    # result = visualize_h5_patch_with_segmentation(
    #     h5_path=h5_path,
    #     parquet_path=parquet_path,
    #     target_barcode="000x097",
    #     figsize=(8, 8),
    # )

