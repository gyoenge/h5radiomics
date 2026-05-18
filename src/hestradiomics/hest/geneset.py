from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc

from hestradiomics.config import DownloadConfig
from hestradiomics.utils import ensure_dir


def save_json(
    path: str | Path,
    data: dict | list,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def check_arg(
    value: str,
    name: str,
    candidates: Sequence[str],
) -> None:
    if value not in candidates:
        raise ValueError(
            f"{name} must be one of "
            f"{candidates}, got: {value}"
        )


def load_all_h5ad_from_dir(
    adata_dir: str | Path,
    sample_ids: Optional[tuple[str, ...]] = None,
) -> list[sc.AnnData]:
    adata_dir = Path(adata_dir)

    h5ad_files = sorted(
        adata_dir.glob("*.h5ad")
    )

    if sample_ids is not None:

        selected = set(sample_ids)

        h5ad_files = [
            path
            for path in h5ad_files
            if path.stem in selected
        ]

    if not h5ad_files:
        raise FileNotFoundError(
            f"No h5ad files found in: {adata_dir}"
        )

    adata_list = []

    for h5ad_path in h5ad_files:
        print(f"Loading h5ad: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        adata.obs_names_make_unique()
        adata_list.append(adata)

    return adata_list


def get_common_genes(
    adata_list: Sequence[sc.AnnData],
    min_cells_pct: float,
) -> list[str]:
    common_genes = None

    for adata in adata_list:
        adata = adata.copy()

        if min_cells_pct:

            min_cells = int(
                np.ceil(
                    min_cells_pct
                    * len(adata.obs)
                )
            )

            sc.pp.filter_genes(
                adata,
                min_cells=min_cells,
            )

        genes = np.array(
            adata.to_df().columns
        )

        if common_genes is None:
            common_genes = genes

        else:
            common_genes = np.intersect1d(
                common_genes,
                genes,
            )

    if common_genes is None:
        return []

    common_genes = [
        gene
        for gene in common_genes
        if "BLANK" not in gene
        and "Control" not in gene
    ]

    print(
        f"Found {len(common_genes)} common genes"
    )

    return list(common_genes)


def select_top_k_genes(
    adata_list: Sequence[sc.AnnData],
    k: int,
    criteria: str,
    min_cells_pct: float,
) -> list[str]:
    check_arg(
        criteria,
        "criteria",
        ["mean", "var"],
    )

    common_genes = get_common_genes(
        adata_list=adata_list,
        min_cells_pct=min_cells_pct,
    )

    if not common_genes:
        raise ValueError(
            "No common genes found."
        )

    expression_dfs = []

    for adata in adata_list:
        df = adata.to_df()[common_genes].copy()

        sample_id = adata.uns.get("sample_id", "sample")
        df.index = [f"{sample_id}_{idx}" for idx in df.index]

        expression_dfs.append(df)

    expression_df = pd.concat(
        expression_dfs,
        axis=0,
    )

    if criteria == "mean":

        genes = (
            expression_df.mean(axis=0)
            .nlargest(k)
            .index
            .tolist()
        )

    else:

        stacked_adata = sc.AnnData(
            expression_df.astype(np.float32)
        )

        sc.pp.filter_genes(
            stacked_adata,
            min_cells=0,
        )

        sc.pp.log1p(stacked_adata)

        sc.pp.highly_variable_genes(
            stacked_adata,
            n_top_genes=k,
        )

        genes = stacked_adata.var_names[
            stacked_adata.var[
                "highly_variable"
            ]
        ][:k].tolist()

    print(
        f"Selected {len(genes)} genes "
        f"by {criteria}"
    )

    return genes


def run_gene_extraction(
    download_cfg: DownloadConfig,
    sample_ids: Optional[tuple[str, ...]] = None,
    k_values: Sequence[int] = (
        50,
        100,
        250,
    ),
    criteria_values: Sequence[str] = (
        "var",
    ),
    min_cells_pct: float = 0.1,
) -> None:
    summary = []

    for oncotree_code in download_cfg.oncotrees:

        oncotree_dir = (
            download_cfg.download_dir
            / oncotree_code
        )

        st_dir = oncotree_dir / "st"

        save_dir = ensure_dir(
            oncotree_dir / "genes"
        )

        if not st_dir.exists():
            print(
                f"[SKIP] st directory not found: "
                f"{st_dir}"
            )
            continue

        print(
            f"\n[Gene Extraction] "
            f"ONCOTREE={oncotree_code}"
        )

        try:

            adata_list = load_all_h5ad_from_dir(
                st_dir,
                sample_ids=sample_ids,
            )

        except FileNotFoundError as e:

            print(f"[SKIP] {e}")
            continue

        for criteria in criteria_values:
            for k in k_values:

                genes = select_top_k_genes(
                    adata_list=adata_list,
                    k=k,
                    criteria=criteria,
                    min_cells_pct=min_cells_pct,
                )

                save_path = (
                    save_dir
                    / f"{criteria}_{k}genes.json"
                )

                save_json(
                    save_path,
                    {"genes": genes},
                )

                summary.append(
                    {
                        "oncotree_code": oncotree_code,
                        "criteria": criteria,
                        "k": k,
                        "num_genes": len(genes),
                        "save_path": str(save_path),
                    }
                )

                print(
                    f"Saved gene list: "
                    f"{save_path}"
                )

    save_json(
        download_cfg.download_dir
        / "gene_extraction_summary.json",
        summary,
    )

    print("Gene extraction completed")
