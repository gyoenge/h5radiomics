from __future__ import annotations

import os
import argparse

from h5radiomics.config.statistics import get_statistics_default_config
from h5radiomics.utils.config import (
    load_yaml_config,
    merge_config,
    str_to_bool,
)
from h5radiomics.engines.statistics import (
    process_single_sample,
    process_merged_samples,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Analyze saved radiomics feature tables for both raw and processed features."
    )
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--status_filter",
        type=str,
        default=None,
        help='Filter rows by status, e.g. "ok". Use None to disable filtering.',
    )
    parser.add_argument(
        "--drop_diagnostic",
        type=str,
        default=None,
        help='true/false. Exclude columns starting with "diagnostics_"',
    )
    parser.add_argument(
        "--save_per_sample",
        type=str,
        default=None,
        help="true/false. Save per-sample statistics under each sample directory.",
    )
    parser.add_argument(
        "--save_merged",
        type=str,
        default=None,
        help="true/false. Save merged statistics across all samples.",
    )
    parser.add_argument(
        "--save_representatives",
        type=str,
        default=None,
        help="true/false. Save representative patches for feature statistics.",
    )
    parser.add_argument(
        "--representative_image_col",
        type=str,
        default=None,
        help='Preferred image column: color_path / gray_path / mask_path',
    )
    parser.add_argument(
        "--save_boxplot",
        type=str,
        default=None,
        help="true/false. Save boxplots for feature distributions.",
    )

    return parser.parse_args(args)


def normalize_config_types(config: dict) -> dict:
    config["drop_diagnostic"] = str_to_bool(config.get("drop_diagnostic"))
    config["save_per_sample"] = str_to_bool(config.get("save_per_sample"))
    config["save_merged"] = str_to_bool(config.get("save_merged"))
    config["save_representatives"] = str_to_bool(config.get("save_representatives"))
    config["save_boxplot"] = str_to_bool(config.get("save_boxplot"))

    if config.get("status_filter") in ("None", "none", ""):
        config["status_filter"] = None

    return config


def main(args=None):
    cli_args = parse_args(args)

    defaults = get_statistics_default_config()
    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}
    config = merge_config(defaults, yaml_config, cli_args)
    config = normalize_config_types(config)

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    os.makedirs(config["output_dir"], exist_ok=True)

    results = []
    for sample_id in config["sample_ids"]:
        result = process_single_sample(sample_id, config)
        results.append(result)

    if config.get("save_merged", False):
        process_merged_samples(results, config)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()