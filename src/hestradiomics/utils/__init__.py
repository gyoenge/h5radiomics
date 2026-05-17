from .config import *
from .h5 import *
from .io import *
from .paths import *
from .patchio import * 
from .feature_utils import * 
from .maskgeo import * 


from typing import Optional

def filter_sample_ids(
    all_sample_ids: list[str],
    selected_sample_ids: Optional[tuple[str, ...]] = None,
) -> list[str]:
    if selected_sample_ids is None:
        return all_sample_ids

    selected = set(selected_sample_ids)
    return [sid for sid in all_sample_ids if sid in selected]


__all__ = []

