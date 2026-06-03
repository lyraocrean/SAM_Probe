"""
SAM Depth Probing experiment

Two-pass memory-friendly design:
  Pass 1: Stream NYU data -> extract SAM features (fp16) -> train SAM probe -> eval -> free
  Pass 2: Stream again   -> extract Random ViT features  -> train rand probe -> eval
  Final:  Print comparison, save visualizations, save checkpoint for fast re-vis

Run modes:
  python main.py              # full experiment (~17 min)
  python main.py --vis-only   # reload saved checkpoint, regenerate plots only (~5 sec)
"""

import os
import sys
import torch

from config import (
    SAM_REPO_PARENT, SAM_CHECKPOINT,
    DATASET_NAME, DATASET_SPLIT, NUM_SAMPLES,
    IMG_SIZE, FEAT_GRID,
    USE_FP16,
    PROBE_BATCH_SIZE, PROBE_LR, PROBE_EPOCHS, TRAIN_RATIO,
    SEED, DEVICE, OUTPUT_DIR,
)
from data_loader import stream_nyu
from feature_extractor import (
    build_sam_extractor, build_random_extractor, extract_single_streaming,
)
from probe import (
    LinearDepthProbe, train_probe, evaluate, visualize, print_comparison,
)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
RESULTS_CSV     = os.path.join(OUTPUT_DIR, "results.csv")


def save_results_csv(sam_result: dict, rand_result: dict):
    import csv
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = [
        ["model",        "rmse",              "mae",             "pearson_r"],
        ["SAM ViT-B",    sam_result["rmse"],   sam_result["mae"],  sam_result["pearson_r"]],
        ["Random ViT-B", rand_result["rmse"],  rand_result["mae"], rand_result["pearson_r"]],
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[results] Saved to {RESULTS_CSV}")


def save_checkpoint(sam_probe, rand_probe, sam_result, rand_result,
                    vis_images, depths, sam_vis_feats, rand_vis_feats):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        "sam_probe":      sam_probe.state_dict(),
        "rand_probe":     rand_probe.state_dict(),
        "sam_result":     sam_result,
        "rand_result":    rand_result,
        "vis_images":     [t.half() for t in vis_images],   # save as fp16
        "depths":         depths[:4],                        # only vis samples needed
        "sam_vis_feats":  sam_vis_feats,
        "rand_vis_feats": rand_vis_feats,
    }, CHECKPOINT_PATH)
    print(f"[checkpoint] Saved to {CHECKPOINT_PATH}")


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    sam_probe  = LinearDepthProbe(feat_dim=768)
    rand_probe = LinearDepthProbe(feat_dim=768)
    sam_probe.load_state_dict(ckpt["sam_probe"])
    rand_probe.load_state_dict(ckpt["rand_probe"])
    vis_images     = [t.float() for t in ckpt["vis_images"]]
    depths         = ckpt["depths"]
    sam_vis_feats  = ckpt["sam_vis_feats"]
    rand_vis_feats = ckpt["rand_vis_feats"]
    return (sam_probe, rand_probe, ckpt["sam_result"], ckpt["rand_result"],
            vis_images, depths, sam_vis_feats, rand_vis_feats)


def run_full(device):
    # ── Build encoders ────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  Step 1/5: Build encoders")
    print("="*50)
    sam_ext  = build_sam_extractor(SAM_REPO_PARENT, SAM_CHECKPOINT, device, USE_FP16)
    rand_ext = build_random_extractor(SAM_REPO_PARENT, device, USE_FP16)

    # ── Pass 1: SAM ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  Step 2/5: SAM feature extraction (streaming {NUM_SAMPLES} images)")
    print("="*50)
    ds1 = stream_nyu(DATASET_NAME, DATASET_SPLIT)
    sam_feats, depths, vis_images, sam_vis_feats = extract_single_streaming(
        sam_ext, ds1, NUM_SAMPLES, device, IMG_SIZE, FEAT_GRID, USE_FP16)

    print(f"\n[info] SAM features: {sam_feats.shape}  {sam_feats.dtype}")

    print("\n" + "="*50)
    print("  Step 3/5: Train SAM linear probe")
    print("="*50)
    sam_probe, sam_xval, sam_yval = train_probe(
        sam_feats, depths, device,
        batch_size=PROBE_BATCH_SIZE, lr=PROBE_LR,
        epochs=PROBE_EPOCHS, train_ratio=TRAIN_RATIO, seed=SEED)
    sam_result = evaluate(sam_probe, sam_xval, sam_yval, device, "SAM ViT-B probe")

    del sam_feats
    torch.cuda.empty_cache()

    # ── Pass 2: Random ViT ────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  Step 4/5: Random ViT feature extraction (streaming {NUM_SAMPLES} images)")
    print("="*50)
    ds2 = stream_nyu(DATASET_NAME, DATASET_SPLIT)
    rand_feats, _, _, rand_vis_feats = extract_single_streaming(
        rand_ext, ds2, NUM_SAMPLES, device, IMG_SIZE, FEAT_GRID, USE_FP16)

    print(f"\n[info] Random ViT features: {rand_feats.shape}  {rand_feats.dtype}")
    print("\n  Training Random ViT linear probe ...")
    rand_probe, rand_xval, rand_yval = train_probe(
        rand_feats, depths, device,
        batch_size=PROBE_BATCH_SIZE, lr=PROBE_LR,
        epochs=PROBE_EPOCHS, train_ratio=TRAIN_RATIO, seed=SEED)
    rand_result = evaluate(rand_probe, rand_xval, rand_yval, device, "Random ViT-B probe")

    del rand_feats
    torch.cuda.empty_cache()

    # ── Save checkpoint ───────────────────────────────────────────────
    save_checkpoint(sam_probe, rand_probe, sam_result, rand_result,
                    vis_images, depths, sam_vis_feats, rand_vis_feats)

    return sam_probe, rand_probe, sam_result, rand_result, vis_images, depths, sam_vis_feats, rand_vis_feats


def main():
    torch.manual_seed(SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    vis_only = "--vis-only" in sys.argv

    if vis_only:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"[error] No checkpoint found at {CHECKPOINT_PATH}. Run without --vis-only first.")
            sys.exit(1)
        print(f"[vis-only] Loading checkpoint from {CHECKPOINT_PATH} ...")
        (sam_probe, rand_probe, sam_result, rand_result,
         vis_images, depths, sam_vis_feats, rand_vis_feats) = load_checkpoint()
        sam_probe  = sam_probe.to(device)
        rand_probe = rand_probe.to(device)
    else:
        (sam_probe, rand_probe, sam_result, rand_result,
         vis_images, depths, sam_vis_feats, rand_vis_feats) = run_full(device)

    # ── Results & visualization ───────────────────────────────────────
    print("\n" + "="*50)
    print("  Step 5/5: Results & Visualization")
    print("="*50)

    print_comparison(sam_result, rand_result)
    save_results_csv(sam_result, rand_result)

    visualize(
        vis_images, depths,
        sam_probe, rand_probe,
        sam_vis_feats, rand_vis_feats,
        device, OUTPUT_DIR, n_samples=2)

    print(f"\n[done] Outputs saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
