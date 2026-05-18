from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import h5py
from tqdm import tqdm

from hestradiomics.extract.constants import (
    CELL_CLASS_COLUMN,
    CELLSEG_ALL_MASK_SUFFIX,
    EXTRACTOR_DEFAULT_LABEL,
    MASK_PATH_COLUMN,
    MASK_SOURCE_CELLSEG,
    MASK_SOURCE_THRESHOLD,
    PATCH_IDX_COLUMN,
    THRESHOLD_MASK_SUFFIX,
)
from hestradiomics.extract.pipeline import get_patch_cellseg
from hestradiomics.segment.pipeline import (
    load_cellseg_h5,
    save_overlay_png,
)
from hestradiomics.utils import (
    build_threshold_mask,
    filter_sample_ids,
    get_barcodes_key,
    get_coords_key,
    get_img_key,
    load_cellseg_dataframe,
    load_patch_data,
    rasterize_geometries_to_mask,
    save_region_mask_images,
)


def save_patch_visualizations(
    patch,
    output_dir: str | Path,
    sample_id: str,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    mask_source: str = MASK_SOURCE_THRESHOLD,
    cellseg_df: Optional[gpd.GeoDataFrame] = None,
    save_threshold_mask: bool = True,
    save_cellseg_all_mask: bool = True,
    save_cellseg_class_masks: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    output_dir = Path(output_dir)

    if mask_source == MASK_SOURCE_THRESHOLD:
        if save_threshold_mask:
            threshold_mask = build_threshold_mask(
                patch.gray_patch,
                label=label,
            )

            out[MASK_PATH_COLUMN] = save_region_mask_images(
                color_patch=patch.color_patch,
                gray_patch=patch.gray_patch,
                mask_patch=threshold_mask,
                output_dir=str(output_dir),
                sample_id=sample_id,
                mask_filename=f"{patch.base_filename}{THRESHOLD_MASK_SUFFIX}",
            )

        return out

    if mask_source == MASK_SOURCE_CELLSEG:
        if cellseg_df is None:
            raise ValueError("cellseg_df is required for cellseg visualization")

        patch_cellseg = get_patch_cellseg(
            cellseg_df=cellseg_df,
            patch_idx=patch.patch_idx,
        )

        if len(patch_cellseg) == 0:
            return out

        if save_cellseg_all_mask:
            merged_mask = rasterize_geometries_to_mask(
                patch_cellseg.geometry.tolist(),
                image_shape=patch.gray_patch.shape,
                label=label,
            )

            out[MASK_PATH_COLUMN] = save_region_mask_images(
                color_patch=patch.color_patch,
                gray_patch=patch.gray_patch,
                mask_patch=merged_mask,
                output_dir=str(output_dir),
                sample_id=sample_id,
                mask_filename=f"{patch.base_filename}{CELLSEG_ALL_MASK_SUFFIX}",
            )

        if save_cellseg_class_masks:
            for class_name, sub in patch_cellseg.groupby(CELL_CLASS_COLUMN):
                if len(sub) == 0:
                    continue

                mask_cls = rasterize_geometries_to_mask(
                    sub.geometry.tolist(),
                    image_shape=patch.gray_patch.shape,
                    label=label,
                )

                save_region_mask_images(
                    color_patch=patch.color_patch,
                    gray_patch=patch.gray_patch,
                    mask_patch=mask_cls,
                    output_dir=str(output_dir),
                    sample_id=sample_id,
                    mask_filename=f"{patch.base_filename}__cellseg_{class_name}",
                )

        if save_threshold_mask:
            threshold_mask = build_threshold_mask(
                patch.gray_patch,
                label=label,
            )

            save_region_mask_images(
                color_patch=patch.color_patch,
                gray_patch=patch.gray_patch,
                mask_patch=threshold_mask,
                output_dir=str(output_dir),
                sample_id=sample_id,
                mask_filename=f"{patch.base_filename}{THRESHOLD_MASK_SUFFIX}",
            )

        return out

    raise ValueError(f"Unsupported mask_source: {mask_source}")


def save_patch_visualizations_from_h5(
    h5_path: str | Path,
    output_dir: str | Path,
    sample_id: Optional[str] = None,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    mask_source: str = MASK_SOURCE_THRESHOLD,
    cellseg_path: Optional[str | Path] = None,
    patch_indices: Optional[List[int]] = None,
    save_threshold_mask: bool = True,
    save_cellseg_all_mask: bool = True,
    save_cellseg_class_masks: bool = True,
) -> List[Dict[str, Any]]:
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    sample_id = sample_id or h5_path.stem

    cellseg_df = None

    if mask_source == MASK_SOURCE_CELLSEG:
        if cellseg_path is None:
            raise ValueError("cellseg_path is required when mask_source='cellseg'")

        cellseg_df = load_cellseg_dataframe(str(cellseg_path))

    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        total_num_patches = len(f[img_key])
        indices = patch_indices or list(range(total_num_patches))

        for i in tqdm(indices, desc=f"[Visualizing patches] {sample_id}"):
            patch = load_patch_data(
                f=f,
                img_key=img_key,
                coords_key=coords_key,
                barcodes_key=barcodes_key,
                patch_idx=i,
            )

            row = save_patch_visualizations(
                patch=patch,
                output_dir=output_dir,
                sample_id=sample_id,
                label=label,
                mask_source=mask_source,
                cellseg_df=cellseg_df,
                save_threshold_mask=save_threshold_mask,
                save_cellseg_all_mask=save_cellseg_all_mask,
                save_cellseg_class_masks=save_cellseg_class_masks,
            )

            row["patch_idx"] = int(i)
            row["barcode"] = patch.barcode
            rows.append(row)

    return rows


def save_segment_overlays_from_h5(
    patch_h5_path: str | Path,
    cellseg_h5_path: str | Path,
    overlay_dir: str | Path,
    use_class_color: bool = True,
    patch_indices: Optional[List[int]] = None,
) -> str:
    patch_h5_path = Path(patch_h5_path)
    cellseg_h5_path = Path(cellseg_h5_path)
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    seg_gdf, patch_df = load_cellseg_h5(str(cellseg_h5_path))

    if patch_indices is not None:
        patch_df = patch_df[
            patch_df[PATCH_IDX_COLUMN].isin(patch_indices)
        ].copy()

        seg_gdf = seg_gdf[
            seg_gdf[PATCH_IDX_COLUMN].isin(patch_indices)
        ].copy()

    gdf_by_patch = {
        int(patch_idx): sub_gdf.copy()
        for patch_idx, sub_gdf in seg_gdf.groupby(PATCH_IDX_COLUMN)
    }

    with h5py.File(patch_h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        indices = patch_df[PATCH_IDX_COLUMN].astype(int).tolist()

        for patch_idx in tqdm(indices, desc="Saving segment overlays"):
            patch = load_patch_data(
                f=f,
                img_key=img_key,
                coords_key=coords_key,
                barcodes_key=barcodes_key,
                patch_idx=patch_idx,
            )

            gdf = gdf_by_patch.get(
                patch_idx,
                gpd.GeoDataFrame(
                    {
                        "cell_id_in_patch": [],
                        "class_id": [],
                        CELL_CLASS_COLUMN: [],
                    },
                    geometry=[],
                    crs=None,
                ),
            )

            save_path = overlay_dir / f"{patch.barcode}_idx{patch_idx}.png"

            save_overlay_png(
                img=patch.color_patch,
                gdf=gdf,
                save_path=str(save_path),
                title=f"idx={patch_idx}, bc={patch.barcode}, cells={len(gdf)}",
                use_class_color=use_class_color,
            )

    return str(overlay_dir)


def run_visualization_for_oncotree(
    oncotree: str,
    data_root: str | Path,
    sample_ids: Optional[tuple[str, ...]] = None,
    patch_dirname: str = "patches",
    cellseg_dirname: str = "cellseg",
    patch_vis_dirname: str = "patch_vis",
    overlay_dirname: str = "segment_vis",
    label: int = EXTRACTOR_DEFAULT_LABEL,
    mask_source: str = MASK_SOURCE_CELLSEG,
    save_patch_masks: bool = True,
    save_segment_overlays: bool = True,
    overwrite: bool = False,
    use_class_color: bool = True,
) -> None:
    data_root = Path(data_root)
    oncotree_root = data_root / oncotree

    patch_dir = oncotree_root / patch_dirname
    cellseg_dir = oncotree_root / cellseg_dirname
    patch_vis_root = oncotree_root / patch_vis_dirname
    overlay_root = oncotree_root / overlay_dirname

    if not patch_dir.exists():
        print(f"[SKIP] Patch dir not found: {patch_dir}")
        return

    h5_paths = sorted(patch_dir.glob("*.h5"))

    if not h5_paths:
        print(f"[SKIP] No h5 files found: {patch_dir}")
        return

    all_sample_ids = [path.stem for path in h5_paths]

    target_sample_ids = filter_sample_ids(
        all_sample_ids=all_sample_ids,
        selected_sample_ids=sample_ids,
    )

    if not target_sample_ids:
        print(f"[SKIP] No selected samples found for {oncotree}")
        return

    target_sample_id_set = set(target_sample_ids)

    h5_paths = [
        path for path in h5_paths
        if path.stem in target_sample_id_set
    ]

    for h5_path in h5_paths:
        sample_id = h5_path.stem
        cellseg_path = cellseg_dir / f"{sample_id}.h5"

        patch_vis_dir = patch_vis_root / sample_id
        overlay_dir = overlay_root / sample_id

        if save_patch_masks:
            if patch_vis_dir.exists() and any(patch_vis_dir.iterdir()) and not overwrite:
                print(f"[SKIP] patch visualizations exist: {patch_vis_dir}")

            elif mask_source == MASK_SOURCE_CELLSEG and not cellseg_path.exists():
                print(f"[SKIP] cellseg h5 not found: {cellseg_path}")

            else:
                save_patch_visualizations_from_h5(
                    h5_path=h5_path,
                    output_dir=patch_vis_dir,
                    sample_id=sample_id,
                    label=label,
                    mask_source=mask_source,
                    cellseg_path=(
                        cellseg_path
                        if mask_source == MASK_SOURCE_CELLSEG
                        else None
                    ),
                )

        if save_segment_overlays:
            if not cellseg_path.exists():
                print(f"[SKIP] cellseg h5 not found: {cellseg_path}")
                continue

            if overlay_dir.exists() and any(overlay_dir.iterdir()) and not overwrite:
                print(f"[SKIP] segment overlays exist: {overlay_dir}")
            else:
                save_segment_overlays_from_h5(
                    patch_h5_path=h5_path,
                    cellseg_h5_path=cellseg_path,
                    overlay_dir=overlay_dir,
                    use_class_color=use_class_color,
                )


def run_visualization_from_config(config) -> None:
    for oncotree in config.download.oncotrees:
        run_visualization_for_oncotree(
            oncotree=oncotree,
            data_root=config.download.download_dir,
            sample_ids=config.sample_ids,
            patch_dirname=config.radiomics.patch_dirname,
            cellseg_dirname=config.radiomics.segment_dirname,
            mask_source=config.radiomics.mask_source,
            overwrite=getattr(config.radiomics, "overwrite_visualization", False),
            use_class_color=config.cellseg.use_class_color,
        )


