from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download

from hestradiomics._utils import ensure_dir, setup_logger
from hestradiomics.config import (
    DOWNLOAD_ROOT, 
    DOWNLOAD_ONCOTREE, 
    DOWNLOAD_REQUIRED,
    DOWNLOAD_OPTIONAL, 
    DOWNLOAD_TECH, 
)


# ============================================================
# Hugging Face
# ============================================================

def huggingface_checkin() -> None:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Set HF_TOKEN.")

    login(token=hf_token)

    try:
        user_info = HfApi().whoami()
        print("Current Hugging Face Account: %s", user_info["name"])
    except Exception as e:
        print("Failed to check Hugging Face account: %s", e)


# ============================================================
# Download HEST
# ============================================================

def build_hest_allow_patterns(
    sample_ids: Sequence[str],
    required_dirs: Sequence[str],
    allowed_dirs: Sequence[str],
) -> list[str]:
    target_dirs = list(dict.fromkeys([*required_dirs, *allowed_dirs]))

    allow_patterns = []

    for dirname in target_dirs:
        for sample_id in sample_ids:
            allow_patterns.append(f"{dirname}/*{sample_id}[_.]**")

    return allow_patterns


def download_hest_by_oncotree(
    download_dir: str | Path,
    metadata_uri: str = "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv",
) -> None:
    download_dir = Path(download_dir)

    print("Loading HEST metadata:", metadata_uri)
    meta_df = pd.read_csv(metadata_uri)

    meta_df = meta_df[
        (meta_df["species"] == "Homo sapiens")
        & (meta_df["oncotree_code"].isin(DOWNLOAD_ONCOTREE))
        & (meta_df["st_technology"].isin(DOWNLOAD_TECH))
    ]

    for oncotree_code in DOWNLOAD_ONCOTREE:
        oncotree_df = meta_df[meta_df["oncotree_code"] == oncotree_code]

        if oncotree_df.empty:
            print(f"[SKIP] No samples found for {oncotree_code}")
            continue

        sample_ids = oncotree_df["id"].astype(str).tolist()

        allow_patterns = build_hest_allow_patterns(
            sample_ids=sample_ids,
            required_dirs=DOWNLOAD_REQUIRED,
            allowed_dirs=DOWNLOAD_OPTIONAL,
        )

        oncotree_dir = ensure_dir(download_dir / oncotree_code)

        print(f"\nStart downloading HEST | oncotree={oncotree_code}")
        print("download_dir:", oncotree_dir)
        print("num_samples:", len(sample_ids))
        print("sample_ids:", sample_ids)

        snapshot_download(
            repo_id="MahmoodLab/hest",
            repo_type="dataset",
            local_dir=str(oncotree_dir),
            allow_patterns=allow_patterns,
        )

        print(f"Completed downloading HEST | oncotree={oncotree_code}")


def run_download(download_root: str | Path) -> None:
    download_root = Path(download_root)

    huggingface_checkin()

    download_hest_by_oncotree(
        download_dir=download_root / "hest",
    )

    print("All download tasks completed")


# ============================================================
# Gene Extraction
# ============================================================

def save_json(path: str | Path, data: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_arg(value: str, name: str, candidates: Sequence[str]) -> None:
    if value not in candidates:
        raise ValueError(f"{name} must be one of {candidates}, got: {value}")


def load_h5ad_list(
    adata_dir: str | Path,
    sample_files: Sequence[str],
) -> list[sc.AnnData]:
    adata_list = []

    for sample_file in sample_files:
        sample_path = Path(adata_dir) / sample_file

        if not sample_path.exists():
            raise FileNotFoundError(f"h5ad not found: {sample_path}")

        print("Loading h5ad: %s", sample_path)
        adata_list.append(sc.read_h5ad(sample_path))

    return adata_list


def get_common_genes(
    adata_list: Sequence[sc.AnnData],
    min_cells_pct: float,
) -> list[str]:
    common_genes = None

    for adata in adata_list:
        adata = adata.copy()

        if min_cells_pct:
            min_cells = int(np.ceil(min_cells_pct * len(adata.obs)))
            sc.pp.filter_genes(adata, min_cells=min_cells)

        genes = np.array(adata.to_df().columns)

        if common_genes is None:
            common_genes = genes
        else:
            common_genes = np.intersect1d(common_genes, genes)

    common_genes = [
        gene for gene in common_genes
        if "BLANK" not in gene and "Control" not in gene
    ]

    print("Found %d common genes", len(common_genes))
    return list(common_genes)


def select_top_k_genes(
    adata_list: Sequence[sc.AnnData],
    k: int,
    criteria: str,
    min_cells_pct: float,
) -> list[str]:
    check_arg(criteria, "criteria", ["mean", "var"])

    common_genes = get_common_genes(
        adata_list=adata_list,
        min_cells_pct=min_cells_pct,
    )

    expression_df = pd.concat(
        [adata.to_df()[common_genes] for adata in adata_list],
        axis=0,
    )

    if criteria == "mean":
        genes = expression_df.mean(axis=0).nlargest(k).index.tolist()

    else:
        stacked_adata = sc.AnnData(expression_df.astype(np.float32))
        sc.pp.filter_genes(stacked_adata, min_cells=0)
        sc.pp.log1p(stacked_adata)
        sc.pp.highly_variable_genes(stacked_adata, n_top_genes=k)

        genes = stacked_adata.var_names[
            stacked_adata.var["highly_variable"]
        ][:k].tolist()

    print("Selected %d genes by %s", len(genes), criteria)
    return genes


def run_gene_extraction(
    download_root: str | Path,
    sample_files: Sequence[str] = ("TENX95.h5ad", "TENX99.h5ad"),
    k_values: Sequence[int] = (50, 100, 250),
    criteria_values: Sequence[str] = ("var",),
    min_cells_pct: float = 0.1,
) -> None:
    download_root = Path(download_root)

    adata_dir = download_root / "adata"
    save_dir = ensure_dir(download_root / "genes")

    adata_list = load_h5ad_list(
        adata_dir=adata_dir,
        sample_files=sample_files,
    )

    summary = []

    for criteria in criteria_values:
        for k in k_values:
            print("Extracting genes | criteria=%s | k=%d", criteria, k)

            genes = select_top_k_genes(
                adata_list=adata_list,
                k=k,
                criteria=criteria,
                min_cells_pct=min_cells_pct,
            )

            save_path = save_dir / f"{criteria}_{k}genes.json"
            save_json(save_path, {"genes": genes})

            summary.append(
                {
                    "criteria": criteria,
                    "k": k,
                    "num_genes": len(genes),
                    "save_path": str(save_path),
                }
            )

            print("Saved gene list: %s", save_path)

    save_json(download_root / "gene_extraction_summary.json", summary)
    print("Gene extraction finished")


# ============================================================


if __name__ == "__main__":
    run_download(
        download_root=DOWNLOAD_ROOT,
    )

    # run_gene_extraction(
    #     download_root=DOWNLOAD_ROOT,
    # )

    print("Pipeline finished successfully")
