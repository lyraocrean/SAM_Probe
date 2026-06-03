"""
Linear probe: training, evaluation, and visualisation.

Design: each of the 64x64 = 4096 patches per image is treated as an
independent sample. Each sample is a (768-dim feature vector, normalised
depth scalar) pair. The probe is a single Linear(768 -> 1) layer trained
with MSE loss.

Metrics:
  RMSE      : root mean squared error (lower is better)
  MAE       : mean absolute error (lower is better)
  Pearson r : linear correlation between predicted and true depth (higher is better)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")   # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt


# ── 模型 ──────────────────────────────────────────────────────────────

class LinearDepthProbe(nn.Module):
    """Single linear layer mapping 768-dim patch features to a scalar normalised depth value."""
    def __init__(self, feat_dim: int = 768):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)   # [...] → [...]


# ── 训练 ──────────────────────────────────────────────────────────────

def train_probe(feats: torch.Tensor, depths: torch.Tensor,
                device: str, batch_size: int = 2048,
                lr: float = 1e-3, epochs: int = 30,
                train_ratio: float = 0.8, seed: int = 42):
    """
    Train the linear probe on pre-extracted features.

    feats:  [N, 64, 64, 768]  -- patch features for all images
    depths: [N, 64, 64]       -- corresponding normalised depth values

    Training runs on CPU (features already live in CPU RAM);
    the probe model and loss are moved to device for gradient computation.
    """
    N, H, W, C = feats.shape

    # Flatten all patches across all images into independent samples.
    # Keep fp16 storage to save RAM; cast to fp32 only within each mini-batch.
    x_all = feats.view(N * H * W, C)    # [N*4096, 768]  fp16 or fp32
    y_all = depths.view(N * H * W)      # [N*4096]  fp32

    # Discard patches where the depth label is zero (invalid pixels).
    valid = y_all > 0
    x_all = x_all[valid]
    y_all = y_all[valid]

    # Train / validation split
    torch.manual_seed(seed)
    perm = torch.randperm(len(x_all))
    n_train = int(train_ratio * len(perm))
    tr_idx, val_idx = perm[:n_train], perm[n_train:]

    x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    print(f"  Train patches: {len(x_tr):,}   Val patches: {len(x_val):,}")

    probe = LinearDepthProbe(feat_dim=C).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        probe.train()
        perm_ep = torch.randperm(len(x_tr))
        epoch_loss = 0.0
        n_batch = 0

        for start in range(0, len(x_tr), batch_size):
            idx = perm_ep[start: start + batch_size]
            xb = x_tr[idx].to(device).float()   # fp16→fp32 only for this mini-batch
            yb = y_tr[idx].to(device)

            pred = probe(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batch += 1

        if ep % 10 == 0 or ep == epochs:
            probe.eval()
            with torch.no_grad():
                val_pred = probe(x_val.to(device).float()).cpu()
                val_loss = F.mse_loss(val_pred, y_val).item()
            print(f"  Epoch {ep:3d}/{epochs}  "
                  f"train_MSE={epoch_loss/n_batch:.5f}  "
                  f"val_MSE={val_loss:.5f}")

    return probe, x_val, y_val


# ── 评估 ──────────────────────────────────────────────────────────────

def evaluate(probe: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor,
             device: str, label: str = "") -> dict:
    """Compute RMSE, MAE, and Pearson r on the validation set."""
    probe.eval()
    with torch.no_grad():
        pred = probe(x_val.to(device).float()).cpu().numpy()
    true = y_val.numpy()

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r, _ = pearsonr(pred, true)

    print(f"\n{'='*50}")
    print(f"  [{label}]")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  Pearson r: {r:.4f}")
    print(f"{'='*50}")

    return {"label": label, "rmse": rmse, "mae": mae, "pearson_r": r,
            "pred": pred, "true": true}


# ── 可视化 ────────────────────────────────────────────────────────────

def visualize(images: list, depths: torch.Tensor,
              sam_probe: nn.Module, rand_probe: nn.Module,
              sam_vis_feats: list, rand_vis_feats: list,
              device: str, output_dir: str, n_samples: int = 2):
    """
    Save side-by-side depth prediction figures for the first n_samples images:
      input image | ground-truth depth | SAM probe prediction | random ViT probe prediction

    sam_vis_feats / rand_vis_feats: list of [4096, 768] tensors (patch features for the first few images)
    """
    os.makedirs(output_dir, exist_ok=True)

    sam_probe.eval()
    rand_probe.eval()

    for i in range(min(n_samples, len(images), len(sam_vis_feats), len(rand_vis_feats))):
        sf = sam_vis_feats[i].to(device).float()    # [4096, 768]
        rf = rand_vis_feats[i].to(device).float()

        with torch.no_grad():
            sam_pred  = sam_probe(sf).cpu().view(64, 64).numpy()
            rand_pred = rand_probe(rf).cpu().view(64, 64).numpy()

        # 原图（反归一化回 0-1 显示）
        img_np = images[i].permute(1, 2, 0).numpy()
        _mean = np.array([123.675, 116.28, 103.53])
        _std  = np.array([ 58.395,  57.12,  57.375])
        img_np = (img_np * _std + _mean).clip(0, 255).astype(np.uint8)

        gt_depth = depths[i].numpy()    # [64, 64]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Input Image (RGB)")
        axes[0].axis("off")

        axes[1].imshow(gt_depth, cmap="plasma")
        axes[1].set_title("Ground Truth Depth (normalized)")
        axes[1].axis("off")

        im2 = axes[2].imshow(sam_pred, cmap="plasma",
                              vmin=gt_depth.min(), vmax=gt_depth.max())
        axes[2].set_title("SAM Probe Prediction")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(rand_pred, cmap="plasma",
                              vmin=gt_depth.min(), vmax=gt_depth.max())
        axes[3].set_title("Random ViT Probe Prediction")
        axes[3].axis("off")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        fig.suptitle(f"Sample {i+1}: Depth Probe — SAM ViT-B vs Random ViT-B", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"[vis] Saved -> {out_path}")


def print_comparison(sam_result: dict, rand_result: dict):
    """Print side-by-side comparison of both probes."""
    print("\n" + "=" * 60)
    print("  Final Comparison")
    print("=" * 60)
    print(f"  {'Metric':<15} {'SAM ViT-B':>15} {'Random ViT-B':>15}")
    print(f"  {'-'*45}")
    for key, name in [("rmse", "RMSE ↓"), ("mae", "MAE ↓"), ("pearson_r", "Pearson r ↑")]:
        sv = sam_result[key]
        rv = rand_result[key]
        print(f"  {name:<15} {sv:>15.4f} {rv:>15.4f}")
    print("=" * 60)
    dr = sam_result["pearson_r"] - rand_result["pearson_r"]
    print(f"\n  SAM Pearson r exceeds Random ViT by {dr:+.4f}")
    if dr > 0.1:
        print("  -> Clearly above random: SAM has encoded depth information.")
    elif dr > 0.02:
        print("  -> Slightly above random: some depth signal, but weak.")
    else:
        print("  -> On par with random: no clear depth encoding detected.")
    print()
