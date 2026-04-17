import os
import argparse 
import pandas as pd 
from h5radiomics.engines.extract_radiomics import (
    build_radiomics_extractor, 
    extract_radiomics, 
    build_processed_feature_df
)
from h5radiomics.utils.config import *
from h5radiomics.utils.utils import make_parquet_safe


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract radiomics features from HDF5 files.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--h5_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--label", type=int, default=None)

    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--filters", nargs="+", type=str, default=None)

    parser.add_argument("--save_patches", action="store_true")
    parser.add_argument("--no_save_patches", action="store_true")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for multiprocessing. 0 or 1 means single-process.",
    )

    return parser.parse_args(args)


def main(args=None): 
    # Parse command-line arguments and merge with defaults and YAML config if provided.
    cli_args = parse_args(args)

    defaults = get_default_config()
    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}

    config = merge_config(defaults, yaml_config, cli_args)

    if cli_args.save_patches:
        config["save_patches"] = True
    if cli_args.no_save_patches:
        config["save_patches"] = False
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Process each sample ID 
    for sample in config["sample_ids"]:
        h5_path = os.path.join(config["h5_dir"], f"{sample}.h5")
        feature_output_root = os.path.join(config["output_root"], "features")
        output_root = os.path.join(feature_output_root, f"{sample}_features")
        os.makedirs(output_root, exist_ok=True)
        print(f"Processing sample {sample} with HDF5 file: {h5_path}")
        print(f"Output will be saved to: {output_root}")

        if not os.path.exists(h5_path):
            print(f"[error] HDF5 file not found: {h5_path}")
            continue

        # Initialize pyradiomics feature extractor with default settings (can be customized as needed)
        extractor = build_radiomics_extractor(
            classes=config["classes"],
            filters=config["filters"],
            label=config["label"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        # Extract radiomics features 
        result = extract_radiomics(
            h5_path=h5_path,
            output_root=output_root,
            extractor=extractor if config["num_workers"] <= 1 else None,
            label=config["label"],
            save_patches=config["save_patches"],
            num_workers=config["num_workers"],
            classes=config["classes"],
            filters=config["filters"],
            image_type_settings=config.get("image_type_settings", {}),
        )

        df = pd.DataFrame(result["rows"])

        # Save results to a CSV file
        csv_path = os.path.join(output_root, f"{sample}_radiomics_features.csv")
        df.to_csv(csv_path, index=False)

        # Parquet 저장
        df_parquet = make_parquet_safe(df)
        parquet_path = os.path.join(output_root, f"{sample}_radiomics_features.parquet")
        df_parquet.to_parquet(parquet_path, index=False)

        print(f"Finished radiomcis extraction of sample {sample}. \
                Total patches: {result['total_num_patches']}.")   
        
        print(f"Saved raw CSV: {csv_path}")
        print(f"Saved raw Parquet: {parquet_path}")

        # Build processed features
        processing_cfg = config.get("processing", {})

        lower_q = processing_cfg.get("lower_q", 0.01)
        upper_q = processing_cfg.get("upper_q", 0.99)
        save_processed = processing_cfg.get("save_processed", True)

        if save_processed:
            processed_df, processed_stats_df = build_processed_feature_df(
                df,
                status_col="status",
                ok_status="ok",
                lower_q=lower_q,
                upper_q=upper_q,
            )

            processed_csv_path = os.path.join(
                output_root, f"{sample}_radiomics_features_processed.csv"
            )
            processed_df.to_csv(processed_csv_path, index=False)

            processed_parquet = make_parquet_safe(processed_df)
            processed_parquet_path = os.path.join(
                output_root, f"{sample}_radiomics_features_processed.parquet"
            )
            processed_parquet.to_parquet(processed_parquet_path, index=False)

            processed_stats_csv_path = os.path.join(
                output_root, f"{sample}_radiomics_features_processed_stats.csv"
            )
            processed_stats_df.to_csv(processed_stats_csv_path, index=False)

            print(f"Saved processed CSV: {processed_csv_path}")
            print(f"Saved processed Parquet: {processed_parquet_path}")
            print(f"Saved processed stats CSV: {processed_stats_csv_path}")
            print(f"Processed config: lower_q={lower_q}, upper_q={upper_q}")


# =========================

if __name__ == "__main__": 
    main() 