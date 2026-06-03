"""
Feature extractors for the SAM and random-baseline encoders.

Design: the SAM image encoder's neck is bypassed; only the raw 768-dim
Transformer block outputs are used. Features are stored as float16 to
reduce memory; each mini-batch is cast to float32 during probe training.

Two extractors with identical architecture:
  build_sam_extractor    <- loads sam_vit_b.pth, frozen
  build_random_extractor <- same architecture, randomly initialised, frozen
"""

import sys
from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import preprocess_image, preprocess_depth, _find_depth_key


class _PreNeckEncoder(nn.Module):
    """
    Runs the patch embed and Transformer blocks of ImageEncoderViT, skipping the neck.
    Returns [B, 64, 64, 768].
    """
    def __init__(self, image_encoder: nn.Module):
        super().__init__()
        self.patch_embed = image_encoder.patch_embed
        self.pos_embed   = image_encoder.pos_embed
        self.blocks      = image_encoder.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x   # [B, 64, 64, 768]


def build_sam_extractor(sam_repo_parent: str, checkpoint: str,
                        device: str, use_fp16: bool = True) -> nn.Module:
    """Load the SAM ViT-B image encoder from a checkpoint, freeze its weights, and return a neck-less extractor."""
    sys.path.insert(0, sam_repo_parent)
    from segment_anything.build_sam import build_sam_vit_b

    print("[model] Loading SAM ViT-B checkpoint ...")
    sam = build_sam_vit_b(checkpoint=checkpoint)
    enc = sam.image_encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()

    ext = _PreNeckEncoder(enc).eval().to(device)
    if use_fp16:
        ext = ext.half()
    print("[model] SAM encoder loaded (frozen, fp16).")
    return ext


def build_random_extractor(sam_repo_parent: str,
                            device: str, use_fp16: bool = True) -> nn.Module:
    """Randomly-initialised ViT-B with identical architecture to SAM, frozen. Used as the control baseline."""
    sys.path.insert(0, sam_repo_parent)
    from segment_anything.modeling.image_encoder import ImageEncoderViT

    print("[model] Building randomly-initialised ViT-B (control baseline) ...")
    enc = ImageEncoderViT(
        depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12, patch_size=16, qkv_bias=True,
        use_rel_pos=True, rel_pos_zero_init=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14, out_chans=256,
    )
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()

    ext = _PreNeckEncoder(enc).eval().to(device)
    if use_fp16:
        ext = ext.half()
    print("[model] Random ViT-B built (frozen, fp16).")
    return ext


@torch.no_grad()
def extract_single_streaming(extractor: nn.Module,
                              ds_stream,
                              num_samples: int,
                              device: str,
                              img_size: int = 1024,
                              feat_grid: int = 64,
                              use_fp16: bool = True,
                              n_vis: int = 4):
    """
    Single-model streaming pass over the dataset.

    Features are stored as float16 to halve memory use:
      500 x 64 x 64 x 768 x 2 bytes ~ 3.1 GB

    Mini-batches are cast to float32 on the fly during probe training;
    the full feature tensor is never expanded to fp32.

    Returns:
      feats     : [N, 64, 64, 768]  float16, in CPU RAM
      depths    : [N, 64, 64]       float32
      vis_images: list of [3, 1024, 1024] float32 (first n_vis images, for visualisation)
      vis_feats : list of [4096, 768] float16 (patch features for the first n_vis images)
    """
    feats_list, depths_list = [], []
    vis_images, vis_feats = [], []
    depth_key = None

    for i, ex in enumerate(tqdm(ds_stream, total=num_samples,
                                 desc="  Extracting features", ncols=80)):
        if i >= num_samples:
            break

        if depth_key is None:
            depth_key = _find_depth_key(ex)
            print(f"\n  Depth field: '{depth_key}'")

        img_t = preprocess_image(ex["image"], img_size)       # [3,H,W] f32
        dep_t = preprocess_depth(ex[depth_key], feat_grid)    # [G,G] f32

        x = img_t.unsqueeze(0).to(device)
        if use_fp16:
            x = x.half()

        feat = extractor(x).cpu().half()                      # [1,64,64,768] f16

        feats_list.append(feat)
        depths_list.append(dep_t)

        if i < n_vis:
            vis_images.append(img_t)
            vis_feats.append(feat.squeeze(0).view(-1, 768))   # [4096,768] f16

    feats  = torch.cat(feats_list, dim=0)    # [N, 64, 64, 768] f16 ~ 3.1 GB
    depths = torch.stack(depths_list)        # [N, 64, 64] f32  ~ 62 MB
    return feats, depths, vis_images, vis_feats
