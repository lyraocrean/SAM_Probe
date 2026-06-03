"""
Data loading and preprocessing.

NYU Depth v2 is loaded with streaming=True, fetching one image at a time
without downloading the full dataset (~3 GB). Iteration stops after
num_samples images have been processed.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from datasets import load_dataset

# SAM normalisation constants (same as ImageNet)
_PIXEL_MEAN = torch.tensor([123.675, 116.28,  103.53]).view(3, 1, 1)
_PIXEL_STD  = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)


def preprocess_image(pil_img: Image.Image, img_size: int = 1024) -> torch.Tensor:
    """Resize and normalise a PIL RGB image to a [3, img_size, img_size] float32 tensor using SAM's ImageNet constants."""
    pil_img = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32)       # [H, W, 3]
    t = torch.from_numpy(arr).permute(2, 0, 1)      # [3, H, W]
    return (t - _PIXEL_MEAN) / _PIXEL_STD


def preprocess_depth(depth_data, feat_grid: int = 64) -> torch.Tensor:
    """
    Resize and normalise a depth map to a [feat_grid, feat_grid] float32 tensor in [0, 1].
    Auto-detects mm vs m units (divides by 1000 if max > 100).
    Applies per-image min-max normalisation.
    """
    if isinstance(depth_data, Image.Image):
        arr = np.array(depth_data, dtype=np.float32)
    else:
        arr = np.array(depth_data, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.max() > 100:
        arr = arr / 1000.0

    d = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    d = F.interpolate(d, size=(feat_grid, feat_grid),
                      mode="bilinear", align_corners=False).squeeze()  # [G,G]

    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    return d


def _find_depth_key(example: dict) -> str:
    for key in ("depth_map", "depth", "annotation", "label"):
        if key in example:
            return key
    raise KeyError(f"No depth field found. Available keys: {list(example.keys())}")


def stream_nyu(dataset_name: str, dataset_split: str) -> object:
    """返回一个 HuggingFace streaming dataset 对象（不触发下载）。"""
    print(f"[data] Streaming {dataset_name} ({dataset_split}) …")
    return load_dataset(dataset_name, split=dataset_split,
                        streaming=True, trust_remote_code=True)
