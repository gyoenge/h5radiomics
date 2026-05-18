from __future__ import annotations

from hestradiomics.config import CONFIG
from hestradiomics.hest import (
    huggingface_checkin, 
    download_hest,
    geneset_extraction,
)
from hestradiomics.segment import (
    segment_all_oncotrees,
)
from hestradiomics.analysis import (
    run_visualization_from_config,
    save_overlays_all_oncotrees_from_config, 
)


def main():
    download_dir = CONFIG.download_dir
    sample_ids = CONFIG.sample_ids
    hest_config = CONFIG.hest
    segment_config = CONFIG.segment
    extract_config = CONFIG.extract
    visualize_config = CONFIG.visualize
    statstics_config = CONFIG.statistics

    # -------------------------------------------------------------------------
    # 1. Download HEST
    # -------------------------------------------------------------------------

    print()
    huggingface_checkin()

    download_hest(
        download_dir=download_dir, 
        oncotrees=hest_config.oncotrees, 
        technologies=hest_config.technologies, 
        required_dirs=hest_config.required_dirs, 
        optional_dirs=hest_config.optional_dirs, 
        sample_ids=sample_ids,
    )

    print("[INFO] HEST download completed")
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 2. Extract Gene Sets
    # -------------------------------------------------------------------------

    geneset_extraction(
        download_dir=download_dir, 
        oncotrees=hest_config.oncotrees, 
        sample_ids=sample_ids,
        k_values=(50, 100, 250),
        criteria_values=("var", "mean"),
        min_cells_pct=0.1,
    )

    print("[INFO] Gene sets extraction completed")
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 3. Segment HEST Patches
    # -------------------------------------------------------------------------

    segment_all_oncotrees(
        hest_root=download_dir,
        oncotrees=hest_config.oncotrees,
        model_path=segment_config.model_path,
        sample_ids=sample_ids,
        batch_size=segment_config.batch_size,
        num_workers=segment_config.num_workers,
        device=segment_config.device,
        overwrite=segment_config.overwrite_segment,
    )

    print("[INFO] Segment completed")
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 4. Visualize Patches / Segment Overlays
    # -------------------------------------------------------------------------

    run_visualization_from_config(CONFIG)

    save_overlays_all_oncotrees_from_config(
        download_cfg=hest_config,
        cellseg_cfg=segment_config,
        sample_ids=sample_ids,
    )

    print("[DONE] Visualization completed.")
    print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 5. Extract Radiomics Features
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()