from __future__ import annotations

import os
import argparse

from h5radiomics.config.segment import get_segment_default_config
from h5radiomics.utils.config import (
    load_yaml_config,
    merge_config,
    str_to_bool,
)
from h5radiomics.utils.paths import get_cellvitseg_dir
from h5radiomics.engines.segment import (
    verify_or_download_model,
    infer_cellvit_model_type,
    CellViTInferenceAdapter,
    segment_h5_patches_with_cellvit,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run CellViT segmentation for H5 patch files."
    )
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patch_indices", nargs="*", default=None)
    parser.add_argument("--postprocess_threads", type=int, default=None)

    parser.add_argument("--verbose", type=str, default=None)

    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--no_class_color", action="store_true")
    parser.add_argument("--save_geojson_per_patch", action="store_true")

    return parser.parse_args(args)


def normalize_config_types(config: dict) -> dict:
    config["save_png_overlay"] = str_to_bool(config.get("save_png_overlay"))
    config["use_class_color"] = str_to_bool(config.get("use_class_color"))
    config["save_geojson_per_patch"] = str_to_bool(config.get("save_geojson_per_patch"))
    config["verbose"] = str_to_bool(config.get("verbose"))

    patch_indices = config.get("patch_indices")
    if patch_indices in ("None", "none", "", [], None):
        config["patch_indices"] = None
    elif isinstance(patch_indices, list):
        if len(patch_indices) == 1 and str(patch_indices[0]).lower() == "all":
            config["patch_indices"] = None
        else:
            config["patch_indices"] = [int(x) for x in patch_indices]
    elif isinstance(patch_indices, str):
        if patch_indices.lower() == "all":
            config["patch_indices"] = None
        else:
            config["patch_indices"] = [int(patch_indices)]

    return config


def main(args=None):
    cli_args = parse_args(args)

    defaults = get_segment_default_config()
    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}
    config = merge_config(defaults, yaml_config, cli_args)
    config = normalize_config_types(config)

    if cli_args.no_overlay:
        config["save_png_overlay"] = False
    if cli_args.no_class_color:
        config["use_class_color"] = False
    if cli_args.save_geojson_per_patch:
        config["save_geojson_per_patch"] = True

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    model_path = os.path.join(config["model_dir"], config["model_name"])
    verify_or_download_model(model_path, config["model_name"])

    for sample_id in config["sample_ids"]:
        h5_path = os.path.join(config["input_dir"], f"{sample_id}.h5")
        if not os.path.exists(h5_path):
            print(f"[WARNING] skip {sample_id}: h5 not found -> {h5_path}")
            continue

        sample_output_dir = get_cellvitseg_dir(config["output_dir"], sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        predictor = CellViTInferenceAdapter(
            model_path=model_path,
            model_name=infer_cellvit_model_type(os.path.basename(model_path)),
            output_dir=sample_output_dir,
            device=config["device"],
            verbose=config["verbose"],
        )

        print("=" * 60)
        print(f"[INFO] Processing sample: {sample_id}")
        print(f"[INFO] h5_path: {h5_path}")
        print(f"[INFO] output_dir: {sample_output_dir}")

        segment_h5_patches_with_cellvit(
            h5_path=h5_path,
            output_dir=sample_output_dir,
            model_path=model_path,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            patch_indices=config["patch_indices"],
            save_png_overlay=config["save_png_overlay"],
            use_class_color=config["use_class_color"],
            save_geojson_per_patch=config["save_geojson_per_patch"],
            device=config["device"],
            postprocess_threads=config["postprocess_threads"],
            predictor=predictor,
        )


if __name__ == "__main__":
    main()
