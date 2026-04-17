from __future__ import annotations

import os
import argparse
import pandas as pd

from h5radiomics.config.extract import get_extract_default_config
from h5radiomics.utils.config import load_yaml_config, merge_config
from h5radiomics.utils.io import make_parquet_safe
from h5radiomics.utils.paths import get_feature_output_dir
from h5radiomics.engines.extract import (
    build_radiomics_extractor,
    extract_radiomics,
    build_processed_feature_df,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract radiomics features from HDF5 files.")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--h5_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--label", type=int, default=None)

    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--filters", nargs="+", type=str, default=None)

    parser.add_argument("--save_patches", action="store_true")
    parser.add_argument("--no_save_patches", action="store_true")

    parser.add_argument("--num_workers", type=int, default=None)

    return parser.parse_args(args)


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

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    for sample_id in config["sample_ids"]:
        h5_path = os.path.join(config["h5_dir"], f"{sample_id}.h5")
        output_dir = get_feature_output_dir(config["output_root"], sample_id)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(h5_path):
            print(f"[ERROR] HDF5 file not found: {h5_path}")
            continue

        extractor = build_radiomics_extractor(
            classes=config["classes"],
            filters=config["filters"],
            label=config["label"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        result = extract_radiomics(
            h5_path=h5_path,
            output_root=output_dir,
            extractor=extractor if config["num_workers"] <= 1 else None,
            label=config["label"],
            save_patches=config["save_patches"],
            num_workers=config["num_workers"],
            classes=config["classes"],
            filters=config["filters"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        df = pd.DataFrame(result["rows"])

        csv_path = os.path.join(output_dir, f"{sample_id}_radiomics_features.csv")
        parquet_path = os.path.join(output_dir, f"{sample_id}_radiomics_features.parquet")

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

            processed_df.to_csv(
                os.path.join(output_dir, f"{sample_id}_radiomics_features_processed.csv"),
                index=False,
            )
            make_parquet_safe(processed_df).to_parquet(
                os.path.join(output_dir, f"{sample_id}_radiomics_features_processed.parquet"),
                index=False,
            )
            processed_stats_df.to_csv(
                os.path.join(output_dir, f"{sample_id}_radiomics_features_processed_stats.csv"),
                index=False,
            )


if __name__ == "__main__":
    main()

    