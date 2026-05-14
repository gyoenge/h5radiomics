from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text) -> str:
    if text is None or text == "":
        return ""

    text = str(text)
    text = re.sub(r'[\\/:\*\?"<>\|\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:180]


def make_base_name(idx: int, barcode: str | None = None) -> str:
    if barcode:
        return f"patch_{idx:06d}_{sanitize_filename(barcode)}"
    return f"patch_{idx:06d}"


def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def convert_value(v):
        if pd.isna(v) if not isinstance(v, (list, tuple, np.ndarray, dict)) else False:
            return np.nan

        if isinstance(v, np.generic):
            return v.item()

        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return v.item()
            return json.dumps(v.tolist(), ensure_ascii=False)

        if isinstance(v, (list, tuple, dict)):
            return json.dumps(v, ensure_ascii=False)

        return v

    for col in df.columns:
        df[col] = df[col].map(convert_value)

    return df
