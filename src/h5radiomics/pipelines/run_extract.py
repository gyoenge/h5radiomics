from __future__ import annotations

import os
import json
import argparse
import pandas as pd

from h5radiomics.config.extract import get_extract_default_config
from h5radiomics.utils.config import load_yaml_config, merge_config
from h5radiomics.utils.io import make_parquet_safe
from h5radiomics.utils.paths import (
    get_sample_root,
    get_raw_features_dir,
    get_processed_features_dir,
    get_raw_features_csv_path,
    get_raw_features_parquet_path,
    get_processed_features_csv_path,
    get_processed_features_parquet_path,
    get_processing_stats_csv_path,
    get_processing_config_json_path,
)
from h5radiomics.engines.extract import (
    build_radiomics_extractor,
    extract_radiomics,
    build_processed_feature_df,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract radiomics features from HDF5 files.")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--label", type=int, default=None)

    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--filters", nargs="+", type=str, default=None)

    parser.add_argument("--save_patches", action="store_true")
    parser.add_argument("--no_save_patches", action="store_true")

    parser.add_argument("--num_workers", type=int, default=None)

    # new: cellseg-based extraction options
    parser.add_argument(
        "--mask_source",
        type=str,
        default=None,
        choices=["threshold", "cellseg"],
        help="Mask source for radiomics extraction",
    )
    parser.add_argument(
        "--cellseg_path",
        type=str,
        default=None,
        help="Optional explicit path to cellseg parquet. If omitted and mask_source=cellseg, "
             "use {output_dir}/{sample_id}/cellvitseg/cellseg.parquet",
    )
    parser.add_argument(
        "--celltype_mode",
        type=str,
        default=None,
        choices=["merged", "per_class", "single"],
        help="How to build masks from cellseg result",
    )
    parser.add_argument(
        "--target_cell_type",
        type=str,
        default=None,
        help="Target cell type name when celltype_mode=single",
    )

    return parser.parse_args(args)


def resolve_cellseg_path(config: dict, sample_id: str) -> str | None:
    """
    Priority:
    1. config['cellseg_path'] if explicitly provided
    2. default sample-wise path under output_dir
       {output_dir}/{sample_id}/cellvitseg/cellseg.parquet
    """
    if config.get("mask_source", "threshold") != "cellseg":
        return None

    explicit_path = config.get("cellseg_path")
    if explicit_path:
        return explicit_path

    return os.path.join(
        config["output_dir"],
        sample_id,
        "cellvitseg",
        "cellseg.parquet",
    )


def main(args=None):
    cli_args = parse_args(args)

    defaults = get_extract_default_config()
    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}
    config = merge_config(
        defaults,
        yaml_config,
        cli_args,
        skip_keys=("config", "save_patches", "no_save_patches"),
    )

    if cli_args.save_patches:
        config["save_patches"] = True
    if cli_args.no_save_patches:
        config["save_patches"] = False

    # backward-compatible defaults
    config.setdefault("mask_source", "threshold")
    config.setdefault("celltype_mode", "merged")
    config.setdefault("target_cell_type", None)
    config.setdefault("cellseg_path", None)

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    for sample_id in config["sample_ids"]:
        h5_path = os.path.join(config["input_dir"], f"{sample_id}.h5")
        sample_root = get_sample_root(config["output_dir"], sample_id)
        raw_dir = get_raw_features_dir(config["output_dir"], sample_id)
        processed_dir = get_processed_features_dir(config["output_dir"], sample_id)

        os.makedirs(sample_root, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        if not os.path.exists(h5_path):
            print(f"[ERROR] HDF5 file not found: {h5_path}")
            continue

        cellseg_path = resolve_cellseg_path(config, sample_id)
        if config["mask_source"] == "cellseg":
            print(f"[INFO] sample_id={sample_id} cellseg_path={cellseg_path}")
            if not cellseg_path or not os.path.exists(cellseg_path):
                print(f"[ERROR] cellseg parquet not found: {cellseg_path}")
                continue

        extractor = build_radiomics_extractor(
            classes=config["classes"],
            filters=config["filters"],
            label=config["label"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        result = extract_radiomics(
            h5_path=h5_path,
            output_dir=config["output_dir"],
            sample_id=sample_id,
            extractor=extractor if (config["num_workers"] is None or config["num_workers"] <= 1) else None,
            label=config["label"],
            save_patches=config["save_patches"],
            num_workers=config["num_workers"],
            classes=config["classes"],
            filters=config["filters"],
            image_type_settings=config.get("image_type_settings", {}),
            mask_source=config["mask_source"],
            cellseg_path=cellseg_path,
            celltype_mode=config["celltype_mode"],
            target_cell_type=config["target_cell_type"],
        )

        df = pd.DataFrame(result["rows"])

        if "status" in df.columns:
            print(df["status"].value_counts(dropna=False).head(20))
            if (df["status"] != "ok").any():
                print(df[df["status"] != "ok"]["status"].head(20).tolist())

        csv_path = get_raw_features_csv_path(config["output_dir"], sample_id)
        parquet_path = get_raw_features_parquet_path(config["output_dir"], sample_id)

        df.to_csv(csv_path, index=False)
        make_parquet_safe(df).to_parquet(parquet_path, index=False)

        processing_cfg = config.get("processing", {})
        if processing_cfg.get("save_processed", True):
            processed_df, processed_stats_df = build_processed_feature_df(
                df,
                status_col="status",
                ok_status="ok",
                lower_q=processing_cfg.get("lower_q", 0.01),
                upper_q=processing_cfg.get("upper_q", 0.99),
            )

            processed_csv_path = get_processed_features_csv_path(config["output_dir"], sample_id)
            processed_parquet_path = get_processed_features_parquet_path(config["output_dir"], sample_id)
            processing_stats_csv_path = get_processing_stats_csv_path(config["output_dir"], sample_id)
            processing_config_json_path = get_processing_config_json_path(config["output_dir"], sample_id)

            processed_df.to_csv(processed_csv_path, index=False)
            make_parquet_safe(processed_df).to_parquet(processed_parquet_path, index=False)
            processed_stats_df.to_csv(processing_stats_csv_path, index=False)

            processing_cfg_to_save = {
                "lower_q": processing_cfg.get("lower_q", 0.01),
                "upper_q": processing_cfg.get("upper_q", 0.99),
                "save_processed": processing_cfg.get("save_processed", True),
                "classes": config["classes"],
                "filters": config["filters"],
                "image_type_settings": config.get("image_type_settings", {}),
                "mask_source": config["mask_source"],
                "celltype_mode": config["celltype_mode"],
                "target_cell_type": config["target_cell_type"],
                "cellseg_path": cellseg_path,
            }

            with open(processing_config_json_path, "w", encoding="utf-8") as f:
                json.dump(processing_cfg_to_save, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()