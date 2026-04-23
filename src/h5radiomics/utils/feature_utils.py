from typing import Dict, Any


# ------------------------------------------------------------------------------
# naming / small helpers
# ------------------------------------------------------------------------------

def normalize_class_name(class_name: Any) -> str:
    return str(class_name).strip().lower().replace(" ", "_")


def strip_shape2d_prefix(name: str) -> str:
    base_name = name.lower().replace("original_shape2d_", "")
    base_name = base_name.replace("original_shape2_d_", "")
    return base_name


def make_feature_prefix(*parts: str) -> str:
    cleaned = [normalize_class_name(p) for p in parts if p]
    return "_".join(cleaned) + "_"


def make_error_row(patch_idx: int, message: str) -> Dict[str, Any]:
    return {
        "patch_idx": patch_idx,
        "barcode": None,
        "color_path": "",
        "gray_path": "",
        "mask_path": "",
        "x": None,
        "y": None,
        "status": f"error: {message}",
    }


def update_status_once(row: Dict[str, Any], status: str) -> None:
    if row.get("status") == "ok":
        row["status"] = status


def safe_update_features(row: Dict[str, Any], fn, error_status: str) -> None:
    try:
        feats = fn()
        if feats:
            row.update(feats)
    except Exception as e:
        update_status_once(row, f"{error_status}: {repr(e)}")
