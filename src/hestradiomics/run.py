from __future__ import annotations

from hestradiomics.config import CONFIG
from hestradiomics.hest import (
    run_download,
    run_gene_extraction,
)
from hestradiomics.segment.pipeline import (
    segment_all_oncotrees_from_config,
)


def main():
    sample_ids = CONFIG.sample_ids
    download_cfg = CONFIG.download
    cellseg_cfg = CONFIG.cellseg

    # -------------------------------------------------------------------------
    # 1. Download HEST
    # -------------------------------------------------------------------------
    run_download(
        download_cfg=download_cfg,
        sample_ids=sample_ids,
    )
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 2. Extract Gene Sets
    # -------------------------------------------------------------------------
    run_gene_extraction(
        download_cfg=download_cfg,
        sample_ids=sample_ids,
        k_values=(50, 100, 250),
        criteria_values=("var", "mean"),
        min_cells_pct=0.1,
    )
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 3. Segment HEST Patches
    # -------------------------------------------------------------------------
    segment_paths = segment_all_oncotrees_from_config(
        download_cfg=download_cfg,
        cellseg_cfg=cellseg_cfg,
        sample_ids=sample_ids,
    )
    print("\n-------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
