"""
All paths and hyperparameters are centralised here. Edit only this file to reconfigure the experiment.
"""

# ── SAM paths (EDIT THESE before running) ────────────────────────────
# Parent directory of the segment_anything repo clone.
# It must contain a segment_anything/ subdirectory.
# Clone from: https://github.com/facebookresearch/segment-anything
SAM_REPO_PARENT = "/path/to/segment-anything-parent"

# Full path to the SAM ViT-B checkpoint file (.pth).
# Download sam_vit_b_01ec64.pth from:
# https://github.com/facebookresearch/segment-anything#model-checkpoints
SAM_CHECKPOINT  = "/path/to/sam_vit_b_01ec64.pth"

# ── Dataset ───────────────────────────────────────────────────────────
# NYU Depth v2 on HuggingFace, streamed — no full dataset download required
DATASET_NAME  = "sayakpaul/nyu_depth_v2"
DATASET_SPLIT = "validation"  # NYU Depth v2 provides train and validation splits
NUM_SAMPLES   = 500   # use fewer images for a faster run

# ── Architecture (ViT-B / SAM encoder) ───────────────────────────────
IMG_SIZE   = 1024              # SAM requires 1024x1024 input
PATCH_SIZE = 16                # SAM ViT-B patch size
FEAT_GRID  = IMG_SIZE // PATCH_SIZE   # 64, spatial size of the feature map
EMBED_DIM  = 768               # ViT-B embedding dim per patch (before neck)

# ── Feature extraction ────────────────────────────────────────────────
ENCODER_BATCH_SIZE = 2   # 2 for 8 GB VRAM; set to 1 if you get OOM
USE_FP16           = True

# ── Linear probe training ─────────────────────────────────────────────
PROBE_BATCH_SIZE = 2048   # patch-level mini-batch; features live in CPU RAM
PROBE_LR         = 1e-3
PROBE_EPOCHS     = 30
TRAIN_RATIO      = 0.8

# ── Misc ──────────────────────────────────────────────────────────────
SEED       = 42
DEVICE     = "cuda"
OUTPUT_DIR = "outputs"   # where output PNGs and results.csv are saved
