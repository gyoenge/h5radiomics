from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from h5radiomics.engines.extractors.constants import *


# ------------------------------------------------------------------------------
# post-processing helpers
# ------------------------------------------------------------------------------

def is_processed_feature_column(col: str) -> bool:
    col_lower = col.lower()

    if col_lower.startswith(MORPH_FEATURE_PREFIX):
        return True

    if not (col_lower.startswith(PATCH_FEATURE_PREFIX) or col_lower.startswith(CELLSEG_FEATURE_PREFIX)):
        return False

    remainder = col_lower.split("_", 1)[1] if "_" in col_lower else ""
    return any(token in remainder for token in RADIOMICS_IMAGE_PREFIXES)


def get_radiomics_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if is_processed_feature_column(col)]


def clip_feature_series(
    s: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.Series, Dict[str, float]]:
    s_num = pd.to_numeric(s, errors="coerce")

    valid = s_num.dropna()
    if valid.empty:
        return s_num, {
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "min_after_clip": np.nan,
            "max_after_clip": np.nan,
        }

    lower_bound = valid.quantile(lower_q)
    upper_bound = valid.quantile(upper_q)

    clipped = s_num.clip(lower=lower_bound, upper=upper_bound)

    return clipped, {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "mean": float(clipped.mean()) if not pd.isna(clipped.mean()) else np.nan,
        "std": float(clipped.std(ddof=0)) if not pd.isna(clipped.std(ddof=0)) else np.nan,
        "min_after_clip": float(clipped.min()) if not pd.isna(clipped.min()) else np.nan,
        "max_after_clip": float(clipped.max()) if not pd.isna(clipped.max()) else np.nan,
    }


def z_normalize_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    mean = s_num.mean()
    std = s_num.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s_num)), index=s_num.index, dtype=float)

    return (s_num - mean) / std


def minmax_rescale_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    min_val = s_num.min()
    max_val = s_num.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.zeros(len(s_num)), index=s_num.index, dtype=float)

    return (s_num - min_val) / (max_val - min_val)


def build_processed_feature_df(
    df: pd.DataFrame,
    status_col: str = "status",
    ok_status: str = STATUS_OK,
    lower_q: float = DEFAULT_CLIP_LOWER_Q,
    upper_q: float = DEFAULT_CLIP_UPPER_Q,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_df = df.copy()
    feature_cols = get_radiomics_feature_columns(df)

    if not feature_cols:
        sample_cols = df.columns[:30].tolist()
        raise ValueError(
            f"No radiomics/morphology feature columns found to process. "
            f"Sample columns: {sample_cols}"
        )

    stats_rows = []
    ok_mask = processed_df[status_col] == ok_status

    for col in feature_cols:
        original_series = pd.to_numeric(processed_df.loc[ok_mask, col], errors="coerce")

        clipped, clip_stats = clip_feature_series(
            original_series,
            lower_q=lower_q,
            upper_q=upper_q,
        )

        z_norm = z_normalize_series(clipped)
        scaled = minmax_rescale_series(z_norm)

        processed_df.loc[ok_mask, col] = scaled.astype(float)
        processed_df.loc[~ok_mask, col] = np.nan

        stats_rows.append(
            {
                "feature": col,
                "lower_q": lower_q,
                "upper_q": upper_q,
                **clip_stats,
                "z_mean": float(z_norm.mean()) if len(z_norm.dropna()) else np.nan,
                "z_std": float(z_norm.std(ddof=0)) if len(z_norm.dropna()) else np.nan,
                "scaled_min": float(scaled.min()) if len(scaled.dropna()) else np.nan,
                "scaled_max": float(scaled.max()) if len(scaled.dropna()) else np.nan,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    return processed_df, stats_df
