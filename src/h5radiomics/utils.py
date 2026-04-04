# utils.py
# Utility functions for HDF5 files containing image patches and associated metadata
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Handling metadata keys in HDF5 files

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

# Filename sanitization and base name generation

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

# Handling HDF5 files containing image patches

def load_h5_patches(h5_path):
    with h5py.File(h5_path, 'r') as f:
        imgs = f['img'][:]           # (N, H, W, C) or (N, C, H, W)
        coords = f['coords'][:]     # (N, 2)
        barcodes = f['barcode'][:] # (N, 1)

        # barcode decode
        barcodes = [b[0].decode('utf-8') for b in barcodes]

    return imgs, coords, barcodes

def show_patch(img):
    if img.shape[0] == 3:  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    
    plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.show()