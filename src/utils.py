# utils.py
# Utility functions for HDF5 files containing image patches and associated metadata
import numpy as np

def get_img_key(f):
    if "img" in f:
        return "img"
    elif "imgs" in f:
        return "imgs"
    elif "images" in f:
        return "images"
    else:
        raise KeyError("img/imgs/images key를 찾지 못했습니다.")

def get_coords_key(f):
    return "coords" if "coords" in f else None

def get_barcodes_key(f):
    if "barcodes" in f:
        return "barcodes"
    elif "barcode" in f:
        return "barcode"
    else:
        return None
    
def to_str_barcode(barcode):
    if barcode is None:
        return ""
    if isinstance(barcode, np.ndarray) and barcode.shape:
        barcode = barcode[0]
    if isinstance(barcode, bytes):
        return barcode.decode("utf-8")
    return str(barcode)

def sanitize_filename(text):
    if text is None or text == "":
        return ""
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = str(text)
    for ch in bad_chars:
        out = out.replace(ch, "_")
    return out

def make_base_name(idx, barcode=None):
    if barcode:
        barcode = sanitize_filename(barcode)
        return f"patch_{idx:06d}_{barcode}"
    return f"patch_{idx:06d}"

