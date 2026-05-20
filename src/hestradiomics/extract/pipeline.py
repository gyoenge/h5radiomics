from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import anndata as ad
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from hestradiomics.extract._cellcomposition import DistributionExtractor
from hestradiomics.extract._cellshape import MorphologyExtractor
from hestradiomics.extract._intensity_texture import RadiomicsFeatureExtractor
from hestradiomics.extract.constants import *
from hestradiomics.extract.postprocess import (
    build_processed_feature_df,
    get_radiomics_feature_columns,
)
from hestradiomics.utils import (
    PatchData,
    build_patch_row_base,
    build_threshold_mask,
    load_patch_data,
    normalize_class_name,
    rasterize_geometries_to_mask,
    safe_update_features,
)


import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)


# =============================================================================
# H5 utils
# =============================================================================

def find_h5_key(f: h5py.File, candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in f:
            return key

    available = list(f.keys())
    raise KeyError(
        f"None of the candidate keys were found. "
        f"candidates={list(candidates)}, available={available}"
    )


def find_h5_keys(f: h5py.File) -> tuple[str, str, str]:
    img_key = find_h5_key(
        f,
        candidates=("img", "imgs", "images", "patches"),
    )
    coords_key = find_h5_key(
        f,
        candidates=("coords", "coord", "coordinates", "spatial"),
    )
    barcodes_key = find_h5_key(
        f,
        candidates=("barcodes", "barcode", "spot_id", "spot_ids"),
    )

    return img_key, coords_key, barcodes_key


# =============================================================================
# Cell segmentation loader
# =============================================================================

def load_cellseg_parquet(cellseg_path: str | Path) -> gpd.GeoDataFrame:
    cellseg_path = Path(cellseg_path)

    if not cellseg_path.exists():
        raise FileNotFoundError(f"Cell segmentation file not found: {cellseg_path}")

    cellseg_df = gpd.read_parquet(cellseg_path)

    if "geometry" not in cellseg_df.columns:
        raise ValueError(f"'geometry' column not found in {cellseg_path}")

    if not isinstance(cellseg_df, gpd.GeoDataFrame):
        cellseg_df = gpd.GeoDataFrame(cellseg_df, geometry="geometry")

    return cellseg_df


# =============================================================================
# Radiomics output saver
# =============================================================================

def _make_h5ad_obs(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    obs = df.drop(columns=list(feature_cols), errors="ignore").copy()

    if obs.index.name is None:
        obs.index.name = "obs_id"

    for col in obs.columns:
        if obs[col].dtype == "object":
            obs[col] = obs[col].where(obs[col].notna(), "")
            obs[col] = obs[col].astype(str)

    return obs


def save_processed_radiomics_outputs(
    processed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_parquet_path: str | Path,
    output_h5ad_path: str | Path,
) -> None:
    output_parquet_path = Path(output_parquet_path)
    output_h5ad_path = Path(output_h5ad_path)

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    output_h5ad_path.parent.mkdir(parents=True, exist_ok=True)

    processed_df.to_parquet(output_parquet_path, index=False)

    feature_cols = get_radiomics_feature_columns(processed_df)

    feature_list_txt_path = (
        output_parquet_path.parent
        / f"{output_parquet_path.stem}_feature_list.txt"
    )
    with open(feature_list_txt_path, "w") as f:
        for feature_name in feature_cols:
            f.write(f"{feature_name}\n")


    X = (
        processed_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    obs = _make_h5ad_obs(
        df=processed_df,
        feature_cols=feature_cols,
    )

    var = pd.DataFrame(index=pd.Index(feature_cols, name="feature"))

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
    )

    adata.uns["radiomics_feature_columns"] = list(feature_cols)
    adata.uns["postprocess_stats"] = stats_df.to_dict(orient="list")

    adata.write_h5ad(output_h5ad_path)

    print(f"[SAVE] parquet: {output_parquet_path}")
    print(f"[SAVE] h5ad: {output_h5ad_path}")


# =============================================================================
# Patch processor
# =============================================================================

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

        if len(patch_cellseg) > 0 and CELL_CLASS_COLUMN in patch_cellseg.columns:
            patch_cellseg[CELL_CLASS_COLUMN] = (
                patch_cellseg[CELL_CLASS_COLUMN]
                .map(normalize_class_name)
            )

        return patch_cellseg

    def process_single_patch(
        self,
        f: h5py.File,
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

        row[PATCH_MASK_AREA_COLUMN] = int(np.count_nonzero(mask > 0))

        if row[PATCH_MASK_AREA_COLUMN] < PATCH_MASK_AREA_MIN_THRESHOLD:
            row[STATUS_COLUMN] = STATUS_SKIPPED_SMALL_MASK
            return row

        if self.intensity_extractor:
            safe_update_features(
                row,
                lambda: self.intensity_extractor.extract_from_patch(
                    gray_patch=patch.gray_patch,
                    use_cache=use_cache,
                ),
                ERROR_PATCH_RADIOMICS,
            )

        if self.texture_extractor:
            safe_update_features(
                row,
                lambda: self.texture_extractor.extract_from_patch(
                    gray_patch=patch.gray_patch,
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

        row[N_CELLS_TOTAL_COLUMN] = int(len(patch_cellseg))

        if len(patch_cellseg) == 0:
            row[STATUS_COLUMN] = STATUS_SKIPPED_NO_CELLSEG

            if self.distribution_extractor:
                row.update(self.distribution_extractor.extract(patch_cellseg))

            return row

        merged_mask = rasterize_geometries_to_mask(
            patch_cellseg.geometry.tolist(),
            image_shape=patch.gray_patch.shape,
            label=self.label,
        )

        row[CELLSEG_MASK_AREA_COLUMN] = int(np.count_nonzero(merged_mask > 0))

        if self.intensity_extractor:
            safe_update_features(
                row,
                lambda: self.intensity_extractor.extract_from_cellseg(
                    gray_patch=patch.gray_patch,
                    patch_cellseg=patch_cellseg,
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
                lambda: self.distribution_extractor.extract(patch_cellseg),
                ERROR_DISTRIBUTION,
            )

        return row


# =============================================================================
# Processor factory
# =============================================================================

def build_default_patch_processor(
    filters: Optional[Sequence[str]] = None,
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


# =============================================================================
# Multiprocessing helpers
# =============================================================================

def chunk_indices(
    n_items: int,
    chunk_size: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got: {chunk_size}")

    return [
        list(range(start, min(start + chunk_size, n_items)))
        for start in range(0, n_items, chunk_size)
    ]


def process_patch_chunk_worker(
    patch_path: str,
    indices: list[int],
    output_dir: str,
    sample_id: str,
    mask_source: str,
    cellseg_path: Optional[str],
    filters: Optional[Sequence[str]],
    label: int,
    use_cache: bool,
) -> tuple[list[Dict[str, Any]], int]:
    rows: list[Dict[str, Any]] = []

    cellseg_df = None
    if mask_source == MASK_SOURCE_CELLSEG:
        if cellseg_path is None:
            raise ValueError(
                f"cellseg_path is required when mask_source='{MASK_SOURCE_CELLSEG}'"
            )
        cellseg_df = load_cellseg_parquet(cellseg_path)

    processor = build_default_patch_processor(
        filters=filters,
        label=label,
    )

    with h5py.File(patch_path, "r") as f:
        img_key, coords_key, barcodes_key = find_h5_keys(f)

        for i in indices:
            try:
                row = processor.process_single_patch(
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
            except Exception as e:
                row = {
                    SAMPLE_ID_COLUMN: sample_id,
                    PATCH_IDX_COLUMN: i,
                    STATUS_COLUMN: STATUS_ERROR,
                    ERROR_COLUMN: str(e),
                }

            rows.append(row)

    return rows, len(indices)


def _extract_rows_single_process(
    patch_path: Path,
    output_dir: Path,
    sample_id: str,
    mask_source: str,
    cellseg_path: Optional[str | Path],
    filters: Optional[Sequence[str]],
    label: int,
    use_cache: bool,
    n_patches: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []

    cellseg_df = None
    if mask_source == MASK_SOURCE_CELLSEG:
        if cellseg_path is None:
            raise ValueError(
                f"cellseg_path is required when mask_source='{MASK_SOURCE_CELLSEG}'"
            )
        cellseg_df = load_cellseg_parquet(cellseg_path)

    processor = build_default_patch_processor(
        filters=filters,
        label=label,
    )

    with h5py.File(patch_path, "r") as f:
        img_key, coords_key, barcodes_key = find_h5_keys(f)

        for i in tqdm(range(n_patches), desc=f"[EXTRACT] {sample_id}"):
            try:
                row = processor.process_single_patch(
                    f=f,
                    img_key=img_key,
                    coords_key=coords_key,
                    barcodes_key=barcodes_key,
                    i=i,
                    output_dir=str(output_dir),
                    sample_id=sample_id,
                    mask_source=mask_source,
                    cellseg_df=cellseg_df,
                    use_cache=use_cache,
                )
            except Exception as e:
                row = {
                    SAMPLE_ID_COLUMN: sample_id,
                    PATCH_IDX_COLUMN: i,
                    STATUS_COLUMN: STATUS_ERROR,
                    ERROR_COLUMN: str(e),
                }

            rows.append(row)

    return rows


def _extract_rows_multi_process(
    patch_path: Path,
    output_dir: Path,
    sample_id: str,
    mask_source: str,
    cellseg_path: Optional[str | Path],
    filters: Optional[Sequence[str]],
    label: int,
    use_cache: bool,
    n_patches: int,
    num_workers: int,
    chunk_size: int,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    chunks = chunk_indices(n_patches, chunk_size=chunk_size)

    mp_context = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
    ) as executor:
        futures = [
            executor.submit(
                process_patch_chunk_worker,
                str(patch_path),
                chunk,
                str(output_dir),
                sample_id,
                mask_source,
                str(cellseg_path) if cellseg_path is not None else None,
                filters,
                label,
                use_cache,
            )
            for chunk in chunks
        ]

        with tqdm(
            total=len(chunks),
            desc=f"[CHUNK-MP] {sample_id}",
            position=0,
            leave=True,
        ) as chunk_pbar, tqdm(
            total=n_patches,
            desc=f"[PATCH-MP] {sample_id}",
            position=1,
            leave=True,
        ) as patch_pbar:
            for future in as_completed(futures):
                chunk_rows, n_done = future.result()

                rows.extend(chunk_rows)
                chunk_pbar.update(1)
                patch_pbar.update(n_done)

    rows = sorted(
        rows,
        key=lambda x: x.get(PATCH_IDX_COLUMN, -1),
    )

    return rows


# =============================================================================
# Sample-level extraction
# =============================================================================

def extract_sample_radiomics(
    patch_path: str | Path,
    output_path: str | Path,
    sample_id: str,
    output_dir: str | Path,
    mask_source: str = MASK_SOURCE_THRESHOLD,
    cellseg_path: Optional[str | Path] = None,
    filters: Optional[Sequence[str]] = None,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    use_cache: bool = True,
    overwrite: bool = False,
    num_workers: int = 1,
    chunk_size: int = 64,
) -> None:
    patch_path = Path(patch_path)
    output_path = Path(output_path)
    output_dir = Path(output_dir)

    output_parquet_path = output_path.with_suffix(".parquet")
    output_h5ad_path = output_path.with_suffix(".h5ad")

    if (
        output_parquet_path.exists()
        and output_h5ad_path.exists()
        and not overwrite
    ):
        print(
            f"[SKIP] {sample_id} already exists: "
            f"{output_parquet_path}, {output_h5ad_path}"
        )
        return

    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    if mask_source == MASK_SOURCE_CELLSEG and cellseg_path is None:
        raise ValueError(
            f"cellseg_path is required when mask_source='{MASK_SOURCE_CELLSEG}'"
        )

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(patch_path, "r") as f:
        img_key, _, _ = find_h5_keys(f)
        n_patches = len(f[img_key])

    if num_workers <= 1:
        rows = _extract_rows_single_process(
            patch_path=patch_path,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_source=mask_source,
            cellseg_path=cellseg_path,
            filters=filters,
            label=label,
            use_cache=use_cache,
            n_patches=n_patches,
        )
    else:
        rows = _extract_rows_multi_process(
            patch_path=patch_path,
            output_dir=output_dir,
            sample_id=sample_id,
            mask_source=mask_source,
            cellseg_path=cellseg_path,
            filters=filters,
            label=label,
            use_cache=use_cache,
            n_patches=n_patches,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

    raw_df = pd.DataFrame(rows)

    processed_df, stats_df = build_processed_feature_df(raw_df)

    save_processed_radiomics_outputs(
        processed_df=processed_df,
        stats_df=stats_df,
        output_parquet_path=output_parquet_path,
        output_h5ad_path=output_h5ad_path,
    )


# =============================================================================
# Oncotree-level extraction
# =============================================================================

def collect_patch_paths(
    patch_dir: Path,
    sample_ids: Optional[Sequence[str]] = None,
) -> list[Path]:
    if sample_ids is None:
        return sorted(patch_dir.glob("*.h5"))

    patch_paths: list[Path] = []

    for sample_id in sample_ids:
        patch_path = patch_dir / f"{sample_id}.h5"
        if patch_path.exists():
            patch_paths.append(patch_path)

    return patch_paths


def extract_all_oncotrees(
    hest_root: str | Path,
    oncotrees: Sequence[str],
    sample_ids: Optional[Sequence[str]] = None,
    output_dirname: str = "radiomics",
    mask_source: str = MASK_SOURCE_THRESHOLD,
    filters: Optional[Sequence[str]] = None,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    use_cache: bool = True,
    overwrite: bool = False,
    num_workers: int = 1,
    chunk_size: int = 64,
) -> None:
    hest_root = Path(hest_root)

    for oncotree in oncotrees:
        patch_dir = hest_root / oncotree / "patches"
        cellseg_dir = hest_root / oncotree / "segment"
        output_dir = hest_root / oncotree / output_dirname

        if not patch_dir.exists():
            print(f"[SKIP] Patch directory not found: {patch_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        patch_paths = collect_patch_paths(
            patch_dir=patch_dir,
            sample_ids=sample_ids,
        )

        print("\n" + "=" * 80)
        print(f"[ONCOTREE] {oncotree}")
        print(f"[PATCH DIR] {patch_dir}")
        print(f"[CELLSEG DIR] {cellseg_dir}")
        print(f"[OUTPUT DIR] {output_dir}")
        print(f"[NUM SAMPLES] {len(patch_paths)}")
        print(f"[MASK SOURCE] {mask_source}")
        print(f"[USE CACHE] {use_cache}")
        print(f"[OVERWRITE] {overwrite}")
        print(f"[NUM WORKERS] {num_workers}")
        print(f"[CHUNK SIZE] {chunk_size}")
        print("=" * 80)

        for patch_path in patch_paths:
            sample_id = patch_path.stem
            output_path = output_dir / f"{sample_id}.parquet"

            cellseg_path = None
            if mask_source == MASK_SOURCE_CELLSEG:
                cellseg_path = cellseg_dir / f"{sample_id}.parquet"

            try:
                print(f"\n[SAMPLE] {sample_id}")

                extract_sample_radiomics(
                    patch_path=patch_path,
                    output_path=output_path,
                    sample_id=sample_id,
                    output_dir=output_dir,
                    mask_source=mask_source,
                    cellseg_path=cellseg_path,
                    filters=filters,
                    label=label,
                    use_cache=use_cache,
                    overwrite=overwrite,
                    num_workers=num_workers,
                    chunk_size=chunk_size,
                )

            except Exception as e:
                print(f"[ERROR] {oncotree}/{sample_id}: {e}")

