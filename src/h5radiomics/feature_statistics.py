# feature_statistics.py
"""
Saved radiomics feature CSV files -> feature-wise statistics + representative patches

Example usage:

(i) Using YAML config file:
cd /root/workspace/h5radiomics/src
python -m h5radiomics.feature_statistics --config ../configs/stats.yaml

(ii) Using command-line arguments:
python -m h5radiomics.feature_statistics \
  --sample_ids TENX95 NCBI785 NCBI783 TENX99 \
  --input_root /root/workspace/h5radiomics/outputs \
  --output_root /root/workspace/h5radiomics/outputs/statistics \
  --status_filter ok \
  --save_representatives true \
  --save_boxplot true
"""

import os
import re
import shutil
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

def get_default_config():
    return {
        "sample_ids": ["TENX95", "NCBI785", "NCBI783", "TENX99"],
        "input_root": "/root/workspace/impl/h5radiomics/output",
        "output_root": "/root/workspace/impl/h5radiomics/statistics",
        "status_filter": "ok",   # None or "ok"
        "drop_diagnostic": True,
        "save_per_sample": True,
        "save_merged": True,

        # representative patch options
        "save_representatives": True,
        "representative_image_col": "color_path",  # fallback: gray_path -> mask_path
        "representative_stats": ["min", "q10", "q25", "q50", "q75", "q90", "max"],

        # boxplot option
        "save_boxplot": True,
    }


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return data


def merge_config(defaults, yaml_config, cli_args):
    config = defaults.copy()

    if yaml_config:
        for k, v in yaml_config.items():
            if v is not None:
                config[k] = v

    for k, v in vars(cli_args).items():
        if k == "config":
            continue
        if v is not None:
            config[k] = v

    return config


# =========================
# Utils
# =========================

def find_feature_csv(input_root, sample_id):
    """
    Expected path:
      {input_root}/features/{sample_id}_features/{sample_id}_radiomics_features.csv
    """
    csv_path = os.path.join(
        input_root,
        "features",
        f"{sample_id}_features",
        f"{sample_id}_radiomics_features.csv"
    )
    return csv_path


def load_feature_csv(csv_path, status_filter="ok"):
    df = pd.read_csv(csv_path)

    if status_filter is not None and "status" in df.columns:
        df = df[df["status"] == status_filter].copy()

    return df


def get_feature_columns(df, drop_diagnostic=True):
    """
    Select only numeric radiomics feature columns.
    Exclude metadata columns and optionally exclude diagnostic columns.
    """
    meta_cols = {
        "patch_idx", "barcode", "color_path", "gray_path", "mask_path",
        "x", "y", "status"
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in meta_cols]

    if drop_diagnostic:
        feature_cols = [c for c in feature_cols if not c.startswith("diagnostics_")]

    return feature_cols


def compute_feature_statistics(df, feature_cols):
    """
    Compute feature-wise statistics.
    Output: rows=feature_name, columns=statistics
    """
    rows = []

    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        row = {
            "feature_name": col,
            "count": int(s.count()),
            "nan_count": int(s.isna().sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q01": s.quantile(0.01),
            "q05": s.quantile(0.05),
            "q10": s.quantile(0.10),
            "q25": s.quantile(0.25),
            "median": s.median(),
            "q50": s.quantile(0.50),
            "q75": s.quantile(0.75),
            "q90": s.quantile(0.90),
            "q95": s.quantile(0.95),
            "q99": s.quantile(0.99),
            "max": s.max(),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
            "abs_mean": s.abs().mean(),
            "zero_count": int((s == 0).sum()),
            "positive_count": int((s > 0).sum()),
            "negative_count": int((s < 0).sum()),
        }
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values("feature_name").reset_index(drop=True)
    return stats_df


def summarize_dataset(df, feature_cols):
    summary = {
        "num_rows": len(df),
        "num_feature_columns": len(feature_cols),
        "num_total_columns": len(df.columns),
    }
    return summary


def save_statistics(stats_df, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}_feature_statistics.csv")
    stats_df.to_csv(csv_path, index=False)
    return csv_path


def sanitize_filename(text):
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180] if len(text) > 180 else text


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_patch_path(row, preferred_col="color_path"):
    """
    Priority:
      1) preferred_col
      2) color_path
      3) gray_path
      4) mask_path
    """
    candidates = [preferred_col, "color_path", "gray_path", "mask_path"]
    seen = set()

    for col in candidates:
        if col in seen:
            continue
        seen.add(col)

        if col in row.index:
            v = row[col]
            if pd.notna(v) and str(v).strip() != "":
                return str(v)

    return None


def get_target_stat_values(series):
    """
    Return desired representative target values for one feature.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None

    return {
        "min": s.min(),
        "q10": s.quantile(0.10),
        "q25": s.quantile(0.25),
        "q50": s.quantile(0.50),
        "q75": s.quantile(0.75),
        "q90": s.quantile(0.90),
        "max": s.max(),
    }


def select_representative_row(df, feature_col, target_value, stat_name):
    """
    Select one actual patch row closest to target_value.

    - min/max: exact idxmin/idxmax
    - quantiles: nearest actual value
    """
    temp = df.copy()
    temp[feature_col] = pd.to_numeric(temp[feature_col], errors="coerce")
    temp = temp[temp[feature_col].notna()].copy()

    if len(temp) == 0:
        return None, None

    if stat_name == "min":
        idx = temp[feature_col].idxmin()
    elif stat_name == "max":
        idx = temp[feature_col].idxmax()
    else:
        temp["_abs_diff_"] = (temp[feature_col] - target_value).abs()
        idx = temp["_abs_diff_"].idxmin()

    row = temp.loc[idx]
    actual_value = float(row[feature_col])
    abs_diff = abs(actual_value - float(target_value))
    return row, abs_diff


def save_representative_patches(df, feature_cols, output_dir, config, prefix="sample"):
    """
    Save representative patch images for each feature into:

      {output_dir}/representative/{feature_name}/{stat_name}__patch{patch_idx}.png

    Also save manifest CSV:
      {output_dir}/representative/representative_manifest.csv
    """
    rep_root = os.path.join(output_dir, "representative")
    ensure_dir(rep_root)

    requested_stats = config.get("representative_stats", ["min", "q10", "q25", "q50", "q75", "q90", "max"])
    preferred_image_col = config.get("representative_image_col", "color_path")

    manifest_rows = []

    for i, feature_col in enumerate(feature_cols, start=1):
        print(f"[INFO] Representative {i}/{len(feature_cols)} : {feature_col}")

        stat_targets = get_target_stat_values(df[feature_col])
        if stat_targets is None:
            continue

        feature_dir = os.path.join(rep_root, sanitize_filename(feature_col))
        ensure_dir(feature_dir)

        for stat_name in requested_stats:
            if stat_name not in stat_targets:
                continue

            target_value = stat_targets[stat_name]
            row, abs_diff = select_representative_row(
                df=df,
                feature_col=feature_col,
                target_value=target_value,
                stat_name=stat_name,
            )

            if row is None:
                manifest_rows.append({
                    "feature_name": feature_col,
                    "stat_name": stat_name,
                    "target_value": target_value,
                    "selected_value": np.nan,
                    "abs_diff": np.nan,
                    "patch_idx": np.nan,
                    "barcode": "",
                    "sample_id": row["sample_id"] if (row is not None and "sample_id" in row.index) else prefix,
                    "source_path": "",
                    "saved_path": "",
                    "status": "no_valid_row",
                })
                continue

            src_path = resolve_patch_path(row, preferred_col=preferred_image_col)
            if src_path is None or not os.path.exists(src_path):
                manifest_rows.append({
                    "feature_name": feature_col,
                    "stat_name": stat_name,
                    "target_value": float(target_value),
                    "selected_value": float(row[feature_col]),
                    "abs_diff": float(abs_diff),
                    "patch_idx": row["patch_idx"] if "patch_idx" in row.index else np.nan,
                    "barcode": row["barcode"] if "barcode" in row.index else "",
                    "sample_id": row["sample_id"] if "sample_id" in row.index else prefix,
                    "source_path": src_path if src_path is not None else "",
                    "saved_path": "",
                    "status": "missing_source_image",
                })
                continue

            patch_idx = row["patch_idx"] if "patch_idx" in row.index else "na"
            barcode = row["barcode"] if "barcode" in row.index else ""
            sample_id = row["sample_id"] if "sample_id" in row.index else prefix

            src_ext = os.path.splitext(src_path)[1]
            if src_ext == "":
                src_ext = ".png"

            out_name = f"{stat_name}__sample_{sanitize_filename(sample_id)}__patch_{patch_idx}{src_ext}"
            dst_path = os.path.join(feature_dir, out_name)

            shutil.copy2(src_path, dst_path)

            manifest_rows.append({
                "feature_name": feature_col,
                "stat_name": stat_name,
                "target_value": float(target_value),
                "selected_value": float(row[feature_col]),
                "abs_diff": float(abs_diff),
                "patch_idx": patch_idx,
                "barcode": barcode,
                "sample_id": sample_id,
                "source_path": src_path,
                "saved_path": dst_path,
                "status": "ok",
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(rep_root, "representative_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[INFO] Saved representative manifest: {manifest_csv}")
    return manifest_csv


def save_sample_feature_boxplot(df, feature_cols, output_dir, prefix):
    """
    Save one combined boxplot image per sample.

    Output:
      {output_dir}/boxplots/{prefix}_feature_boxplot.png

    X-axis: feature names
    Y-axis: feature value distribution
    Overlay: representative patch positions for
             min, q10, q25, q50, q75, q90, max
    """
    if len(feature_cols) == 0:
        return None

    boxplot_dir = os.path.join(output_dir, "boxplots")
    ensure_dir(boxplot_dir)

    requested_stats = ["min", "q10", "q25", "q50", "q75", "q90", "max"]

    data_for_boxplot = []
    scatter_x = []
    scatter_y = []

    for i, feature_col in enumerate(feature_cols, start=1):
        s = pd.to_numeric(df[feature_col], errors="coerce").dropna()
        if len(s) == 0:
            data_for_boxplot.append(np.array([np.nan]))
            continue

        data_for_boxplot.append(s.values)

        stat_targets = get_target_stat_values(df[feature_col])
        if stat_targets is None:
            continue

        for stat_name in requested_stats:
            target_value = stat_targets[stat_name]
            row, _ = select_representative_row(
                df=df,
                feature_col=feature_col,
                target_value=target_value,
                stat_name=stat_name,
            )
            if row is None:
                continue

            actual_value = pd.to_numeric(row[feature_col], errors="coerce")
            if pd.isna(actual_value):
                continue

            scatter_x.append(i)
            scatter_y.append(float(actual_value))

    # feature 수에 비례해서 figure width 조절
    fig_width = max(16, len(feature_cols) * 0.22)
    fig_height = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.boxplot(
        data_for_boxplot,
        showfliers=False,
        patch_artist=False,
        widths=0.5,
    )

    if len(scatter_x) > 0:
        ax.scatter(
            scatter_x,
            scatter_y,
            s=10,
            alpha=0.9,
        )

    ax.set_title(
        f"Radiomics Feature Distribution\nRepresentative patch position overlay",
        fontsize=11
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature value")

    ax.set_xticks(range(1, len(feature_cols) + 1))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(boxplot_dir, f"{prefix}_feature_boxplot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved sample feature boxplot: {save_path}")
    return save_path


def extract_feature_class(feature_name: str) -> str:
    """
    Examples:
      original_firstorder_Mean -> firstorder
      wavelet-HHH_glcm_Contrast -> glcm
      log-sigma-3-0-mm-3D_glszm_ZoneEntropy -> glszm
    """
    parts = str(feature_name).split("_")
    if len(parts) >= 2:
        return parts[1].lower()
    return "unknown"


def zscore_features(df, feature_cols):
    """
    Feature-wise z-score normalization:
      z = (x - mean) / std

    std == 0 or NaN 이면 0으로 처리
    """
    out = df.copy()

    for col in feature_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        mean = s.mean()
        std = s.std()

        if pd.isna(std) or std == 0:
            out[col] = 0.0
        else:
            out[col] = (s - mean) / std

    return out


def minmax_rescale_features(df, feature_cols):
    """
    Feature-wise [0,1] rescale:
      x' = (x - min) / (max - min)

    max == min or NaN 이면 0으로 처리
    """
    out = df.copy()

    for col in feature_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        vmin = s.min()
        vmax = s.max()

        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            out[col] = 0.0
        else:
            out[col] = (s - vmin) / (vmax - vmin)

    return out


def save_sample_feature_boxplots_by_class(
    df,
    feature_cols,
    output_dir,
    prefix,
    file_tag="",
    title_tag="raw"
):
    """
    Save boxplot images split by feature class.

    Examples:
      {output_dir}/boxplots/{prefix}_firstorder_feature_boxplot.png
      {output_dir}/boxplots/{prefix}_z-score_firstorder_feature_boxplot.png
      {output_dir}/boxplots/{prefix}_z-score-01-rescale_firstorder_feature_boxplot.png
    """
    if len(feature_cols) == 0:
        return []

    boxplot_dir = os.path.join(output_dir, "boxplots")
    ensure_dir(boxplot_dir)

    requested_stats = ["min", "q10", "q25", "q50", "q75", "q90", "max"]

    class_to_features = {}
    for col in feature_cols:
        cls = extract_feature_class(col)
        class_to_features.setdefault(cls, []).append(col)

    saved_paths = []

    for feature_class, class_features in sorted(class_to_features.items()):
        data_for_boxplot = []
        scatter_x = []
        scatter_y = []

        for i, feature_col in enumerate(class_features, start=1):
            s = pd.to_numeric(df[feature_col], errors="coerce").dropna()
            if len(s) == 0:
                data_for_boxplot.append(np.array([np.nan]))
                continue

            data_for_boxplot.append(s.values)

            stat_targets = get_target_stat_values(df[feature_col])
            if stat_targets is None:
                continue

            for stat_name in requested_stats:
                target_value = stat_targets[stat_name]
                row, _ = select_representative_row(
                    df=df,
                    feature_col=feature_col,
                    target_value=target_value,
                    stat_name=stat_name,
                )
                if row is None:
                    continue

                actual_value = pd.to_numeric(row[feature_col], errors="coerce")
                if pd.isna(actual_value):
                    continue

                scatter_x.append(i)
                scatter_y.append(float(actual_value))

        if len(data_for_boxplot) == 0:
            continue

        fig_width = max(12, len(class_features) * 0.35)
        fig_height = 6

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.boxplot(
            data_for_boxplot,
            showfliers=False,
            patch_artist=False,
            widths=0.5,
        )

        if len(scatter_x) > 0:
            ax.scatter(
                scatter_x,
                scatter_y,
                s=10,
                alpha=0.9,
            )

        ax.set_title(
            f"Radiomics Feature Distribution ({title_tag}, {feature_class})\nRepresentative patch position overlay",
            fontsize=11
        )
        ax.set_xlabel("Feature")
        ax.set_ylabel("Feature value")

        ax.set_xticks(range(1, len(class_features) + 1))
        ax.set_xticklabels(class_features, rotation=90, fontsize=7)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()

        if file_tag:
            filename = f"{prefix}_{file_tag}_{sanitize_filename(feature_class)}_feature_boxplot.png"
        else:
            filename = f"{prefix}_{sanitize_filename(feature_class)}_feature_boxplot.png"

        save_path = os.path.join(boxplot_dir, filename)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved class boxplot: {save_path}")
        saved_paths.append(save_path)

    return saved_paths


# =========================
# Main logic
# =========================

def process_single_sample(sample_id, config):
    csv_path = find_feature_csv(config["input_root"], sample_id)

    if not os.path.exists(csv_path):
        print(f"[error] CSV not found for sample {sample_id}: {csv_path}")
        return None

    print(f"[INFO] Loading sample: {sample_id}")
    print(f"[INFO] CSV path: {csv_path}")

    df = load_feature_csv(
        csv_path=csv_path,
        status_filter=config["status_filter"],
    )

    feature_cols = get_feature_columns(
        df=df,
        drop_diagnostic=config["drop_diagnostic"],
    )

    summary = summarize_dataset(df, feature_cols)
    print(f"[INFO] sample={sample_id}, rows={summary['num_rows']}, features={summary['num_feature_columns']}")

    if len(feature_cols) == 0:
        print(f"[warning] No feature columns found for sample {sample_id}")
        return None

    stats_df = compute_feature_statistics(df, feature_cols)

    output_dir = os.path.join(config["output_root"], f"{sample_id}_stats")

    if config["save_per_sample"]:
        csv_out = save_statistics(stats_df, output_dir, sample_id)
        print(f"[INFO] Saved per-sample statistics CSV : {csv_out}")

    if config.get("save_representatives", False):
        save_representative_patches(
            df=df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            config=config,
            prefix=sample_id,
        )

    if config.get("save_boxplot", False):
        save_sample_feature_boxplot(
            df=df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix=sample_id,
        )
        
        # 1) raw
        save_sample_feature_boxplots_by_class(
            df=df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix=sample_id,
            file_tag="",
            title_tag="raw",
        )

        # 2) z-score
        df_z = zscore_features(df, feature_cols)
        save_sample_feature_boxplots_by_class(
            df=df_z,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix=sample_id,
            file_tag="z-score",
            title_tag="z-score",
        )

        # 3) z-score -> [0,1] rescale
        df_z01 = minmax_rescale_features(df_z, feature_cols)
        save_sample_feature_boxplots_by_class(
            df=df_z01,
            feature_cols=feature_cols,
            output_dir=output_dir,
            prefix=sample_id,
            file_tag="z-score-01-rescale",
            title_tag="z-score -> [0,1] rescale",
        )

    return {
        "sample_id": sample_id,
        "df": df,
        "feature_cols": feature_cols,
        "stats_df": stats_df,
    }


def process_merged_samples(results, config):
    valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        print("[warning] No valid sample results to merge.")
        return

    merged_df_list = []
    for r in valid_results:
        temp = r["df"].copy()
        temp["sample_id"] = r["sample_id"]
        merged_df_list.append(temp)

    merged_df = pd.concat(merged_df_list, axis=0, ignore_index=True)

    feature_cols = get_feature_columns(
        df=merged_df,
        drop_diagnostic=config["drop_diagnostic"],
    )

    print(f"[INFO] Merged rows={len(merged_df)}, features={len(feature_cols)}")

    merged_stats_df = compute_feature_statistics(merged_df, feature_cols)

    output_dir = os.path.join(config["output_root"], "merged_stats")
    csv_out = save_statistics(merged_stats_df, output_dir, "merged")
    print(f"[INFO] Saved merged statistics CSV : {csv_out}")

    merged_csv_path = os.path.join(output_dir, "merged_filtered_features.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"[INFO] Saved merged filtered feature table: {merged_csv_path}")

    if config.get("save_representatives", False):
        save_representative_patches(
            df=merged_df,
            feature_cols=feature_cols,
            output_dir=output_dir,
            config=config,
            prefix="merged",
        )


# =========================
# Args
# =========================

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Analyze saved radiomics feature CSV files.")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--sample_ids", nargs="+", type=str, default=None)
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)

    parser.add_argument("--status_filter", type=str, default=None,
                        help='Filter rows by status, e.g. "ok". Use None to disable filtering.')

    parser.add_argument("--drop_diagnostic", type=str, default=None,
                        help='true/false. Exclude columns starting with "diagnostics_"')

    parser.add_argument("--save_per_sample", type=str, default=None,
                        help="true/false")
    parser.add_argument("--save_merged", type=str, default=None,
                        help="true/false")

    parser.add_argument("--save_representatives", type=str, default=None,
                        help="true/false")
    parser.add_argument("--representative_image_col", type=str, default=None,
                        help='Preferred image column: color_path / gray_path / mask_path')
    
    parser.add_argument("--save_boxplot", type=str, default=None,
                        help="true/false")

    return parser.parse_args(args)


def str_to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean value from: {v}")


def normalize_config_types(config):
    config["drop_diagnostic"] = str_to_bool(config.get("drop_diagnostic"))
    config["save_per_sample"] = str_to_bool(config.get("save_per_sample"))
    config["save_merged"] = str_to_bool(config.get("save_merged"))
    config["save_representatives"] = str_to_bool(config.get("save_representatives"))
    config["save_boxplot"] = str_to_bool(config.get("save_boxplot"))

    if config.get("status_filter") in ("None", "none", ""):
        config["status_filter"] = None

    return config


# =========================
# Entry point
# =========================

def main(args=None):
    cli_args = parse_args(args)

    defaults = get_default_config()
    yaml_config = load_yaml_config(cli_args.config) if cli_args.config else {}
    config = merge_config(defaults, yaml_config, cli_args)
    config = normalize_config_types(config)

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    os.makedirs(config["output_root"], exist_ok=True)

    results = []
    for sample_id in config["sample_ids"]:
        result = process_single_sample(sample_id, config)
        results.append(result)

    if config["save_merged"]:
        process_merged_samples(results, config)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()