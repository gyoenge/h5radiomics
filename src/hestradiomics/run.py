from __future__ import annotations

from hestradiomics.config import CONFIG
from hestradiomics.hest import (
    run_download,
    run_gene_extraction,
)


def main():
    sample_ids = CONFIG.sample_ids
    download_cfg = CONFIG.download

    # -------------------------------------------------------------------------
    # 1. Download HEST
    # -------------------------------------------------------------------------
    run_download(
        download_cfg=download_cfg,
        sample_ids=sample_ids,
    )

    # -------------------------------------------------------------------------
    # 2. Extract Gene Sets
    # -------------------------------------------------------------------------
    run_gene_extraction(
        download_cfg=download_cfg,
        sample_ids=sample_ids,

        k_values=(
            50,
            100,
            250,
        ),

        criteria_values=(
            "var",
            "mean",
        ),

        min_cells_pct=0.1,
    )


if __name__ == "__main__":
    main()


