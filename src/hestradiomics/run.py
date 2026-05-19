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
from hestradiomics.extract import (
    extract_all_oncotrees, 
)
from hestradiomics.analysis import (
    patch_visualization_from_oncotrees,
    segment_visualization_from_oncotrees, 
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

    # patch_visualization_from_oncotrees(
    #     download_dir=download_dir,
    #     oncotrees=hest_config.oncotrees,
    #     sample_ids=sample_ids,
    #     vis_ratio=visualize_config.vis_ratio,
    #     overwrite_visualization=visualize_config.overwrite,
    # )

    # print("\n-------------------------------------------------------------------------\n")

    # segment_visualization_from_oncotrees(
    #     hest_root=download_dir,
    #     oncotrees=hest_config.oncotrees,
    #     sample_ids=sample_ids,
    #     vis_ratio=visualize_config.vis_ratio,
    #     overwrite=visualize_config.overwrite,
    #     use_class_color=True,
    # )

    # print("[DONE] Visualization completed.")
    # print("\n-------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # 5. Extract Radiomics Features
    # -------------------------------------------------------------------------

    extract_all_oncotrees(
        hest_root=download_dir,
        oncotrees=hest_config.oncotrees,
        sample_ids=sample_ids,
        output_dirname=extract_config.output_dirname,
        mask_source=extract_config.mask_source,
        overwrite=extract_config.overwrite,
        num_workers=8,
        chunk_size=64,
    )

if __name__ == "__main__":
    main()
