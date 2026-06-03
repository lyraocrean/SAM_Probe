"""Quick sanity check: streams 3 images and tests both encoders and a probe forward pass."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config import SAM_REPO_PARENT, SAM_CHECKPOINT, DATASET_NAME, DATASET_SPLIT, DEVICE, USE_FP16
from data_loader import stream_nyu
from feature_extractor import build_sam_extractor, build_random_extractor, extract_single_streaming
from probe import LinearDepthProbe

device = DEVICE if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("\n[1] Build encoders ...")
sam_ext  = build_sam_extractor(SAM_REPO_PARENT, SAM_CHECKPOINT, device, USE_FP16)
rand_ext = build_random_extractor(SAM_REPO_PARENT, device, USE_FP16)

print("\n[2] SAM: streaming 3 images ...")
ds = stream_nyu(DATASET_NAME, DATASET_SPLIT)
sam_feats, depths, vis_imgs, sam_vis = extract_single_streaming(
    sam_ext, ds, num_samples=3, device=device, use_fp16=USE_FP16)
print(f"  sam_feats:  {sam_feats.shape}  dtype={sam_feats.dtype}")
print(f"  depths:     {depths.shape}")
print(f"  vis_feats:  {sam_vis[0].shape}")

print("\n[3] Random ViT: streaming 3 images ...")
ds2 = stream_nyu(DATASET_NAME, DATASET_SPLIT)
rand_feats, _, _, rand_vis = extract_single_streaming(
    rand_ext, ds2, num_samples=3, device=device, use_fp16=USE_FP16)
print(f"  rand_feats: {rand_feats.shape}  dtype={rand_feats.dtype}")

print("\n[4] Probe forward pass ...")
probe = LinearDepthProbe(feat_dim=768)
out = probe(sam_vis[0].float())
print(f"  Output shape: {out.shape}")   # [4096]

print("\n✓ All checks passed. Ready to run main.py.")
