from __future__ import annotations

import argparse
from typing import List

from h5radiomics.config.base import get_common_default_config
from h5radiomics.config.extract import get_extract_default_config
from h5radiomics.config.statistics import get_statistics_default_config
from h5radiomics.config.segment import get_segment_default_config
from h5radiomics.utils.config import load_yaml_config, merge_config

from h5radiomics.pipelines.run_extract import main as run_extract_main
from h5radiomics.pipelines.run_statistics import main as run_statistics_main
from h5radiomics.pipelines.run_segment import main as run_segment_main


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the full h5radiomics pipeline: segment -> extract -> statistics"
    )

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # common
    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # extract-related
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--filters", nargs="+", type=str, default=None)
    parser.add_argument("--save_patches", action="store_true")
    parser.add_argument("--no_save_patches", action="store_true")

    parser.add_argument(
        "--mask_source",
        type=str,
        default=None,
        choices=["threshold", "cellseg"],
    )
    parser.add_argument("--cellseg_path", type=str, default=None)
    parser.add_argument(
        "--celltype_mode",
        type=str,
        default=None,
        choices=["merged", "per_class", "single"],
    )
    parser.add_argument("--target_cell_type", type=str, default=None)

    # shared runtime
    parser.add_argument("--num_workers", type=int, default=None)

    # statistics-related
    parser.add_argument("--status_filter", type=str, default=None)
    parser.add_argument("--drop_diagnostic", type=str, default=None)
    parser.add_argument("--save_per_sample", type=str, default=None)
    parser.add_argument("--save_merged", type=str, default=None)
    parser.add_argument("--save_representatives", type=str, default=None)
    parser.add_argument("--representative_image_col", type=str, default=None)
    parser.add_argument("--save_boxplot", type=str, default=None)

    # segment-related
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patch_indices", nargs="*", default=None)
    parser.add_argument("--postprocess_threads", type=int, default=None)
    parser.add_argument("--verbose", type=str, default=None)
    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--no_class_color", action="store_true")
    parser.add_argument("--save_geojson_per_patch", action="store_true")

    # stage control
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_statistics", action="store_true")
    parser.add_argument("--skip_segment", action="store_true")

    return parser.parse_args(args)


def build_full_config(cli_args) -> dict:
    config = get_common_default_config()
    config.update(get_extract_default_config())
    config.update(get_statistics_default_config())
    config.update(get_segment_default_config())

    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}
    config = merge_config(
        config,
        yaml_config,
        cli_args,
        skip_keys=("config", "save_patches", "no_save_patches"),
    )

    if cli_args.save_patches:
        config["save_patches"] = True
    if cli_args.no_save_patches:
        config["save_patches"] = False

    config.setdefault("mask_source", "threshold")
    config.setdefault("celltype_mode", "merged")
    config.setdefault("target_cell_type", None)
    config.setdefault("cellseg_path", None)

    return config


def config_to_cli_args_for_extract(config: dict, original_args) -> List[str]:
    args = []

    if original_args.config:
        args += ["--config", original_args.config]

    if config.get("sample_ids"):
        args += ["--sample_ids", *map(str, config["sample_ids"])]

    if config.get("input_dir") is not None:
        args += ["--input_dir", str(config["input_dir"])]

    if config.get("output_dir") is not None:
        args += ["--output_dir", str(config["output_dir"])]

    if config.get("label") is not None:
        args += ["--label", str(config["label"])]

    if config.get("classes"):
        args += ["--classes", *map(str, config["classes"])]

    if config.get("filters"):
        args += ["--filters", *map(str, config["filters"])]

    if config.get("num_workers") is not None:
        args += ["--num_workers", str(config["num_workers"])]

    if config.get("mask_source") is not None:
        args += ["--mask_source", str(config["mask_source"])]

    if config.get("cellseg_path") is not None:
        args += ["--cellseg_path", str(config["cellseg_path"])]

    if config.get("celltype_mode") is not None:
        args += ["--celltype_mode", str(config["celltype_mode"])]

    if config.get("target_cell_type") is not None:
        args += ["--target_cell_type", str(config["target_cell_type"])]

    if config.get("save_patches") is True:
        args += ["--save_patches"]
    elif config.get("save_patches") is False:
        args += ["--no_save_patches"]

    return args


def config_to_cli_args_for_statistics(config: dict, original_args) -> List[str]:
    args = []

    if original_args.config:
        args += ["--config", original_args.config]

    if config.get("sample_ids"):
        args += ["--sample_ids", *map(str, config["sample_ids"])]

    if config.get("output_dir") is not None:
        args += ["--output_dir", str(config["output_dir"])]

    for key in [
        "status_filter",
        "drop_diagnostic",
        "save_per_sample",
        "save_merged",
        "save_representatives",
        "representative_image_col",
        "save_boxplot",
    ]:
        value = config.get(key)
        if value is not None:
            args += [f"--{key}", str(value)]

    return args


def config_to_cli_args_for_segment(config: dict, original_args) -> List[str]:
    args = []

    if original_args.config:
        args += ["--config", original_args.config]

    if config.get("sample_ids"):
        args += ["--sample_ids", *map(str, config["sample_ids"])]

    for key in [
        "input_dir",
        "output_dir",
        "model_dir",
        "model_name",
        "batch_size",
        "num_workers",
        "device",
        "postprocess_threads",
        "verbose",
    ]:
        value = config.get(key)
        if value is not None:
            args += [f"--{key}", str(value)]

    patch_indices = config.get("patch_indices")
    if patch_indices not in (None, "", [], "None", "none"):
        if isinstance(patch_indices, list):
            args += ["--patch_indices", *map(str, patch_indices)]
        else:
            args += ["--patch_indices", str(patch_indices)]

    if original_args.no_overlay:
        args += ["--no_overlay"]
    if original_args.no_class_color:
        args += ["--no_class_color"]
    if original_args.save_geojson_per_patch:
        args += ["--save_geojson_per_patch"]

    return args


def main(args=None):
    cli_args = parse_args(args)
    config = build_full_config(cli_args)

    print("Full pipeline configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 1) segment first
    if not cli_args.skip_segment:
        print("\n" + "=" * 80)
        print("[RUN] Stage 1/3: Run CellViT segmentation")
        print("=" * 80)
        segment_args = config_to_cli_args_for_segment(config, cli_args)
        run_segment_main(segment_args)
    else:
        print("\n[SKIP] Segment stage skipped")

    # 2) extract second
    if not cli_args.skip_extract:
        print("\n" + "=" * 80)
        print("[RUN] Stage 2/3: Extract radiomics features")
        print("=" * 80)
        extract_args = config_to_cli_args_for_extract(config, cli_args)
        run_extract_main(extract_args)
    else:
        print("\n[SKIP] Extract stage skipped")

    # 3) statistics last
    if not cli_args.skip_statistics:
        print("\n" + "=" * 80)
        print("[RUN] Stage 3/3: Compute feature statistics")
        print("=" * 80)
        statistics_args = config_to_cli_args_for_statistics(config, cli_args)
        run_statistics_main(statistics_args)
    else:
        print("\n[SKIP] Statistics stage skipped")

    print("\n" + "=" * 80)
    print("[DONE] Full pipeline finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
