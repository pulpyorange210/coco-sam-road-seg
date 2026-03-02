import numpy as np
from pycocotools import mask as mask_utils

def binmask_to_rle(mask_bool: np.ndarray) -> dict:
    """
    Convert HxW boolean mask to COCO RLE (jsonable).
    """
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
