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
    parser = argparse.ArgumentParser(description="Analyze saved radiomics feature CSV files.")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)

    parser.add_argument("--status_filter", type=str, default=None)
    parser.add_argument("--drop_diagnostic", type=str, default=None)
    parser.add_argument("--save_per_sample", type=str, default=None)
    parser.add_argument("--save_merged", type=str, default=None)
    parser.add_argument("--save_representatives", type=str, default=None)
    parser.add_argument("--representative_image_col", type=str, default=None)
    parser.add_argument("--save_boxplot", type=str, default=None)

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

    os.makedirs(config["output_root"], exist_ok=True)

    results = []
    for sample_id in config["sample_ids"]:
        results.append(process_single_sample(sample_id, config))

    if config["save_merged"]:
        process_merged_samples(results, config)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()