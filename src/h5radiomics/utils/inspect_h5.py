from __future__ import annotations

from pathlib import Path
import h5py


def inspect_h5_file(h5_path: str | Path) -> dict:
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    summary = {
        "datasets": {},
        "groups": [],
    }

    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                summary["groups"].append(name)
            elif isinstance(obj, h5py.Dataset):
                summary["datasets"][name] = {
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                }

        f.visititems(visitor)

    return summary

