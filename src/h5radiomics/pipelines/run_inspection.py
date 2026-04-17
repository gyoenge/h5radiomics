from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import h5py


# =========================================================
# Expected structure
# =========================================================
EXPECTED_PATCH_DIRS = [
    "color",
    "gray",
    "mask",
    "masked_color",
    "masked_gray",
]

EXPECTED_RAW_FILES = [
    "features.csv",
    "features.parquet",
]

EXPECTED_PROCESSED_FILES = [
    "features.csv",
    "features.parquet",
    "processing_stats.csv",
    "processing_config.json",
]

EXPECTED_STAT_FILES = [
    "stats.csv",
    "stats.parquet",
]

EXPECTED_CELLVIT_FILES = [
    "cellseg.geojson",
    "cellseg.parquet",
    "metadata.csv",
    "summary.json",
]

IMG_KEYS = ("img", "imgs", "images")
BARCODE_KEYS = ("barcode", "barcodes")


# =========================================================
# Dataclasses
# =========================================================
@dataclass
class H5InspectionResult:
    sample_id: str
    exists: bool
    h5_path: str
    keys: List[str]
    img_key: Optional[str]
    img_shape: Optional[List[int]]
    has_coords: bool
    has_barcodes: bool
    num_patches: Optional[int]
    valid: bool
    errors: List[str]


@dataclass
class OutputInspectionResult:
    sample_id: str
    sample_root: str
    exists: bool

    # extract
    patches_exists: bool
    patch_dirs_ok: bool
    patch_file_counts: Dict[str, int]

    raw_exists: bool
    raw_files_ok: bool

    processed_exists: bool
    processed_files_ok: bool

    # statistics
    statistics_root_exists: bool
    statistics_raw_exists: bool
    statistics_raw_files_ok: bool
    statistics_raw_has_representative: bool
    statistics_raw_has_boxplots: bool

    statistics_processed_exists: bool
    statistics_processed_files_ok: bool
    statistics_processed_has_representative: bool
    statistics_processed_has_boxplots: bool

    # segment
    cellvitseg_exists: bool
    cellvitseg_files_ok: bool
    cellvitseg_overlay_exists: bool

    # stage summary
    extract_done: bool
    processed_done: bool
    statistics_raw_done: bool
    statistics_processed_done: bool
    segment_done: bool

    valid_structure: bool
    errors: List[str]


# =========================================================
# Helpers
# =========================================================
def find_first_existing_key(keys: List[str], candidates: tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in keys:
            return k
    return None


def safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def count_files(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        1 for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
    )


def check_expected_files(base_dir: str, expected_files: List[str]) -> bool:
    if not os.path.isdir(base_dir):
        return False
    existing = set(os.listdir(base_dir))
    return all(name in existing for name in expected_files)


def load_processing_config_if_possible(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================================================
# H5 inspection
# =========================================================
def inspect_single_h5(h5_dir: str, sample_id: str) -> H5InspectionResult:
    h5_path = os.path.join(h5_dir, f"{sample_id}.h5")
    errors: List[str] = []

    if not os.path.isfile(h5_path):
        return H5InspectionResult(
            sample_id=sample_id,
            exists=False,
            h5_path=h5_path,
            keys=[],
            img_key=None,
            img_shape=None,
            has_coords=False,
            has_barcodes=False,
            num_patches=None,
            valid=False,
            errors=[f"H5 file not found: {h5_path}"],
        )

    keys: List[str] = []
    img_key: Optional[str] = None
    img_shape: Optional[List[int]] = None
    has_coords = False
    has_barcodes = False
    num_patches: Optional[int] = None
    valid = True

    try:
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            img_key = find_first_existing_key(keys, IMG_KEYS)
            has_coords = "coords" in f
            has_barcodes = find_first_existing_key(keys, BARCODE_KEYS) is not None

            if img_key is None:
                errors.append("Missing image dataset key: expected one of img/imgs/images")
                valid = False
            else:
                img_shape = list(f[img_key].shape)
                if len(img_shape) != 4:
                    errors.append(f"Image dataset must be 4D, got shape={img_shape}")
                    valid = False
                else:
                    num_patches = img_shape[0]

                    # expected layout: (N,H,W,3) or (N,3,H,W)
                    if not (
                        (img_shape[-1] == 3)
                        or (len(img_shape) >= 2 and img_shape[1] == 3)
                    ):
                        errors.append(
                            f"Unexpected image layout: expected (N,H,W,3) or (N,3,H,W), got {img_shape}"
                        )
                        valid = False

            if has_coords and num_patches is not None:
                coords_shape = list(f["coords"].shape)
                if len(coords_shape) != 2 or coords_shape[1] != 2:
                    errors.append(f"coords should have shape (N,2), got {coords_shape}")
                    valid = False
                elif coords_shape[0] != num_patches:
                    errors.append(
                        f"coords first dimension mismatch: coords={coords_shape[0]}, img={num_patches}"
                    )
                    valid = False

    except Exception as e:
        errors.append(f"Failed to read H5: {e}")
        valid = False

    return H5InspectionResult(
        sample_id=sample_id,
        exists=True,
        h5_path=h5_path,
        keys=keys,
        img_key=img_key,
        img_shape=img_shape,
        has_coords=has_coords,
        has_barcodes=has_barcodes,
        num_patches=num_patches,
        valid=valid,
        errors=errors,
    )


def inspect_h5_inputs(h5_dir: str, sample_ids: Optional[List[str]] = None) -> List[H5InspectionResult]:
    if sample_ids is None:
        sample_ids = [
            os.path.splitext(name)[0]
            for name in safe_listdir(h5_dir)
            if name.endswith(".h5")
        ]
    return [inspect_single_h5(h5_dir, sid) for sid in sample_ids]


# =========================================================
# Output inspection
# =========================================================
def inspect_single_output(output_dir: str, sample_id: str) -> OutputInspectionResult:
    errors: List[str] = []

    sample_root = os.path.join(output_dir, sample_id)
    exists = os.path.isdir(sample_root)

    patches_root = os.path.join(sample_root, "patches")
    features_root = os.path.join(sample_root, "features")
    raw_dir = os.path.join(features_root, "raw")
    processed_dir = os.path.join(features_root, "processed")
    statistics_root = os.path.join(features_root, "statistics")
    statistics_raw_dir = os.path.join(statistics_root, "raw")
    statistics_processed_dir = os.path.join(statistics_root, "processed")
    cellvitseg_dir = os.path.join(sample_root, "cellvitseg")
    overlay_dir = os.path.join(cellvitseg_dir, "overlay")

    # patches
    patches_exists = os.path.isdir(patches_root)
    patch_file_counts = {
        name: count_files(os.path.join(patches_root, name))
        for name in EXPECTED_PATCH_DIRS
    }
    patch_dirs_ok = patches_exists and all(
        os.path.isdir(os.path.join(patches_root, name))
        for name in EXPECTED_PATCH_DIRS
    )

    # features raw / processed
    raw_exists = os.path.isdir(raw_dir)
    raw_files_ok = check_expected_files(raw_dir, EXPECTED_RAW_FILES)

    processed_exists = os.path.isdir(processed_dir)
    processed_files_ok = check_expected_files(processed_dir, EXPECTED_PROCESSED_FILES)

    # statistics
    statistics_root_exists = os.path.isdir(statistics_root)

    statistics_raw_exists = os.path.isdir(statistics_raw_dir)
    statistics_raw_files_ok = check_expected_files(statistics_raw_dir, EXPECTED_STAT_FILES)
    statistics_raw_has_representative = os.path.isdir(os.path.join(statistics_raw_dir, "representative"))
    statistics_raw_has_boxplots = os.path.isdir(os.path.join(statistics_raw_dir, "boxplots"))

    statistics_processed_exists = os.path.isdir(statistics_processed_dir)
    statistics_processed_files_ok = check_expected_files(statistics_processed_dir, EXPECTED_STAT_FILES)
    statistics_processed_has_representative = os.path.isdir(os.path.join(statistics_processed_dir, "representative"))
    statistics_processed_has_boxplots = os.path.isdir(os.path.join(statistics_processed_dir, "boxplots"))

    # segment
    cellvitseg_exists = os.path.isdir(cellvitseg_dir)
    cellvitseg_files_ok = check_expected_files(cellvitseg_dir, EXPECTED_CELLVIT_FILES)
    cellvitseg_overlay_exists = os.path.isdir(overlay_dir)

    # stage summary
    extract_done = patch_dirs_ok and raw_files_ok
    processed_done = processed_files_ok
    statistics_raw_done = (
        statistics_raw_exists
        and statistics_raw_files_ok
        and statistics_raw_has_representative
        and statistics_raw_has_boxplots
    )
    statistics_processed_done = (
        statistics_processed_exists
        and statistics_processed_files_ok
        and statistics_processed_has_representative
        and statistics_processed_has_boxplots
    )
    segment_done = cellvitseg_exists and cellvitseg_files_ok and cellvitseg_overlay_exists

    valid_structure = True

    if not exists:
        errors.append(f"Sample output directory not found: {sample_root}")
        valid_structure = False
    else:
        if not patch_dirs_ok:
            errors.append("Patch directory structure is incomplete")
            valid_structure = False
        if raw_exists and not raw_files_ok:
            errors.append("Raw feature directory exists but required files are missing")
            valid_structure = False
        if processed_exists and not processed_files_ok:
            errors.append("Processed feature directory exists but required files are missing")
            valid_structure = False
        if statistics_raw_exists and not statistics_raw_files_ok:
            errors.append("Statistics/raw exists but required files are missing")
            valid_structure = False
        if statistics_processed_exists and not statistics_processed_files_ok:
            errors.append("Statistics/processed exists but required files are missing")
            valid_structure = False
        if cellvitseg_exists and not cellvitseg_files_ok:
            errors.append("cellvitseg exists but required files are missing")
            valid_structure = False

    return OutputInspectionResult(
        sample_id=sample_id,
        sample_root=sample_root,
        exists=exists,
        patches_exists=patches_exists,
        patch_dirs_ok=patch_dirs_ok,
        patch_file_counts=patch_file_counts,
        raw_exists=raw_exists,
        raw_files_ok=raw_files_ok,
        processed_exists=processed_exists,
        processed_files_ok=processed_files_ok,
        statistics_root_exists=statistics_root_exists,
        statistics_raw_exists=statistics_raw_exists,
        statistics_raw_files_ok=statistics_raw_files_ok,
        statistics_raw_has_representative=statistics_raw_has_representative,
        statistics_raw_has_boxplots=statistics_raw_has_boxplots,
        statistics_processed_exists=statistics_processed_exists,
        statistics_processed_files_ok=statistics_processed_files_ok,
        statistics_processed_has_representative=statistics_processed_has_representative,
        statistics_processed_has_boxplots=statistics_processed_has_boxplots,
        cellvitseg_exists=cellvitseg_exists,
        cellvitseg_files_ok=cellvitseg_files_ok,
        cellvitseg_overlay_exists=cellvitseg_overlay_exists,
        extract_done=extract_done,
        processed_done=processed_done,
        statistics_raw_done=statistics_raw_done,
        statistics_processed_done=statistics_processed_done,
        segment_done=segment_done,
        valid_structure=valid_structure,
        errors=errors,
    )


def infer_sample_ids_from_outputs(output_dir: str) -> List[str]:
    if not os.path.isdir(output_dir):
        return []
    return [
        name for name in safe_listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, name))
    ]


def inspect_outputs(output_dir: str, sample_ids: Optional[List[str]] = None) -> List[OutputInspectionResult]:
    if sample_ids is None:
        sample_ids = infer_sample_ids_from_outputs(output_dir)
    return [inspect_single_output(output_dir, sid) for sid in sample_ids]


# =========================================================
# Summary + printing
# =========================================================
def summarize(h5_results: List[H5InspectionResult], out_results: List[OutputInspectionResult]) -> Dict[str, Any]:
    return {
        "num_h5_checked": len(h5_results),
        "num_h5_valid": sum(r.valid for r in h5_results),
        "num_output_checked": len(out_results),
        "num_output_valid": sum(r.valid_structure for r in out_results),
        "num_extract_done": sum(r.extract_done for r in out_results),
        "num_processed_done": sum(r.processed_done for r in out_results),
        "num_statistics_raw_done": sum(r.statistics_raw_done for r in out_results),
        "num_statistics_processed_done": sum(r.statistics_processed_done for r in out_results),
        "num_segment_done": sum(r.segment_done for r in out_results),
    }


def print_h5_results(results: List[H5InspectionResult]) -> None:
    print("\n" + "=" * 80)
    print("[H5 INPUT INSPECTION]")
    print("=" * 80)
    if not results:
        print("No H5 files found.")
        return

    for r in results:
        print(f"- sample_id: {r.sample_id}")
        print(f"  exists      : {r.exists}")
        print(f"  valid       : {r.valid}")
        print(f"  h5_path     : {r.h5_path}")
        print(f"  img_key     : {r.img_key}")
        print(f"  img_shape   : {r.img_shape}")
        print(f"  num_patches : {r.num_patches}")
        print(f"  has_coords  : {r.has_coords}")
        print(f"  has_barcodes: {r.has_barcodes}")
        if r.errors:
            print(f"  errors      : {r.errors}")


def print_output_results(results: List[OutputInspectionResult]) -> None:
    print("\n" + "=" * 80)
    print("[OUTPUT INSPECTION]")
    print("=" * 80)
    if not results:
        print("No output samples found.")
        return

    for r in results:
        print(f"- sample_id: {r.sample_id}")
        print(f"  exists                     : {r.exists}")
        print(f"  valid_structure            : {r.valid_structure}")
        print(f"  extract_done               : {r.extract_done}")
        print(f"  processed_done             : {r.processed_done}")
        print(f"  statistics_raw_done        : {r.statistics_raw_done}")
        print(f"  statistics_processed_done  : {r.statistics_processed_done}")
        print(f"  segment_done               : {r.segment_done}")
        print(f"  patch_file_counts          : {r.patch_file_counts}")
        if r.errors:
            print(f"  errors                     : {r.errors}")


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")


# =========================================================
# CLI
# =========================================================
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Inspect h5radiomics data_root: H5 inputs, outputs, and formatting."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing h5/ and outputs/",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        default=None,
        help="Optional subset of sample IDs to inspect",
    )

    parser.add_argument("--h5input", action="store_true", help="Inspect H5 input files only")
    parser.add_argument("--output", action="store_true", help="Inspect outputs only")
    parser.add_argument("--all", action="store_true", help="Inspect both H5 inputs and outputs")

    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save inspection result as JSON",
    )

    return parser.parse_args(args)


def main(args=None):
    cli_args = parse_args(args)

    data_root = cli_args.data_root
    h5_dir = os.path.join(data_root, "h5")
    output_dir = os.path.join(data_root, "outputs")

    run_h5 = cli_args.h5input
    run_output = cli_args.output

    if cli_args.all or (not cli_args.h5input and not cli_args.output):
        run_h5 = True
        run_output = True

    h5_results: List[H5InspectionResult] = []
    out_results: List[OutputInspectionResult] = []

    if run_h5:
        h5_results = inspect_h5_inputs(h5_dir, cli_args.sample_ids)
        print_h5_results(h5_results)

    if run_output:
        out_results = inspect_outputs(output_dir, cli_args.sample_ids)
        print_output_results(out_results)

    summary = summarize(h5_results, out_results)
    print_summary(summary)

    if cli_args.save_json is not None:
        payload = {
            "data_root": data_root,
            "summary": summary,
            "h5_results": [asdict(x) for x in h5_results],
            "output_results": [asdict(x) for x in out_results],
        }
        with open(cli_args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Saved inspection JSON: {cli_args.save_json}")


if __name__ == "__main__":
    main()
