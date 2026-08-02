#!/usr/bin/env python3
# Sampling Script for Connectivity Optimized Model
# Fixes dimensions error and uses GT guidance

import os, argparse, json, math, torch, numpy as np, bitsandbytes as bnb
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from einops import rearrange
from timm.models.layers import trunc_normal_
import logging
import random


# ---------------- Model Components (Must match training) ----------------
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(8, 8, 8), in_channels=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b e d h w -> b (d h w) e')


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)


def window_reverse(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim;
        self.window_size = window_size;
        self.num_heads = num_heads
        head_dim = dim // num_heads;
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C))


class SwinTransformer3DBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 8, 8), shift_size=(0, 0, 0)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(dim * 4, dim))
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x, grid_size):
        D, H, W = grid_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        shifted = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                             dims=(1, 2, 3)) if any(self.shift_size) else x
        xw = window_partition(shifted, self.window_size)
        aw = self.attn(xw)
        shifted = window_reverse(aw, self.window_size, D, H, W)
        x = torch.roll(shifted, shifts=self.shift_size, dims=(1, 2, 3)) if any(self.shift_size) else shifted
        x = x.view(B, D * H * W, C)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x): return self.act(self.conv(self.up(x)))


class VoxelDiT(nn.Module):
    def __init__(self, input_size=(192, 192, 192), patch_size=(8, 8, 8), in_channels=1, embed_dim=768, depth=16,
                 num_heads=12, window_size=(8, 8, 8), use_checkpoint=True, cond_dim=1):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        self.grid_size = tuple(s // p for s, p in zip(input_size, self.patch_size))
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_embed = PatchEmbed3D(self.patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.time_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.blocks = nn.ModuleList([
            SwinTransformer3DBlock(embed_dim, num_heads, window_size,
                                   shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(ws // 2 for ws in window_size)) for i
            in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.decoder = nn.ModuleList()
        current_channels = embed_dim
        num_ups = int(round(math.log2(self.patch_size[0])))
        for i in range(num_ups):
            next_ch = max(embed_dim // (2 ** (i + 1)), in_channels * 8)
            self.decoder.append(UpBlock3D(current_channels, next_ch))
            current_channels = next_ch
        self.final_conv = nn.Conv3d(current_channels, in_channels, kernel_size=3, padding=1)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0);
            nn.init.constant_(m.weight, 1.0)

    def _time_emb(self, t, dim):
        f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
        return torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], dim=-1)

    def forward(self, x, t, cond=None):
        x = self.patch_embed(x) + self.pos_embed
        t_embed = self.time_embed(self._time_emb(t, 768))
        x = x + t_embed.unsqueeze(1)
        if self.cond_dim > 0:
            if cond is None: cond = torch.zeros((x.size(0), self.cond_dim), device=x.device)
            x = x + self.cond_mlp(cond).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, self.grid_size)
        x = self.norm_out(x)
        x = rearrange(x, 'b (d h w) e -> b e d h w', d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2])
        for up in self.decoder: x = up(x)
        return self.final_conv(x)


# ---------------- Dataset ----------------
class NpyRockDataset(Dataset):
    def __init__(self, npy_dir, sub_volume_depth=192, train_resolution=192, pore_value=0):
        self.npy_files = sorted(list(Path(npy_dir).glob("*.npy")))
        self.sub_volume_depth = sub_volume_depth
        self.train_resolution = train_resolution
        self.pore_value = pore_value
        print(f"Dataset: Found {len(self.npy_files)} files in {npy_dir}")

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        arr = np.load(self.npy_files[idx])
        if arr.dtype == np.bool_:
            mask = (~arr) if self.pore_value == 0 else arr.copy()
        else:
            mask = (arr == self.pore_value)
        vol = torch.from_numpy(mask.astype(np.float32))
        D, H, W = vol.shape
        if D > self.sub_volume_depth:
            start = torch.randint(0, D - self.sub_volume_depth + 1, (1,)).item()
            vol = vol[start:start + self.sub_volume_depth, :, :]
        else:
            pad = self.sub_volume_depth - D
            vol = F.pad(vol, (0, 0, 0, 0, 0, pad))
        vol = vol.unsqueeze(0).unsqueeze(0)
        if vol.shape[-1] != self.train_resolution:
            vol = F.interpolate(vol, size=(self.sub_volume_depth, self.train_resolution, self.train_resolution),
                                mode='trilinear', align_corners=False)
        # Returns [1, D, H, W]
        return vol.squeeze(0)


# ---------------- Helpers ----------------
def calc_s2_on_tensor(prob, lags):
    prob = prob.float()
    vals = []
    for r in lags:
        r = int(r)
        if r <= 0:
            vals.append(prob.mean(dim=(1, 2, 3, 4)))
            continue
        vD = (prob[:, :, :-r, :, :] * prob[:, :, r:, :, :]).mean(dim=(1, 2, 3, 4))
        vH = (prob[:, :, :, :-r, :] * prob[:, :, :, r:, :]).mean(dim=(1, 2, 3, 4))
        vW = (prob[:, :, :, :, :-r] * prob[:, :, :, :, r:]).mean(dim=(1, 2, 3, 4))
        vals.append((vD + vH + vW) / 3.0)
    return torch.stack(vals, dim=-1)


def compute_dataset_stats(dataset, lags, sample_n=100):
    indices = np.random.choice(len(dataset), min(len(dataset), sample_n), replace=False)
    phi_vals, s2_vals = [], []
    for i in tqdm(indices, desc="Calculating Stats"):
        x0 = dataset[i].unsqueeze(0)  # [1, 1, D, H, W]
        phi_vals.append(x0.mean().item())
        s2 = calc_s2_on_tensor(x0, lags)
        s2_vals.append(s2.squeeze(0).numpy())
    mu_phi = float(np.mean(phi_vals))
    std_phi = float(np.std(phi_vals)) if len(phi_vals) > 1 else 1.0
    s2_arr = np.stack(s2_vals, axis=0)
    mu_s2 = np.mean(s2_arr, axis=0)
    std_s2 = np.std(s2_arr, axis=0)
    std_s2[std_s2 < 1e-6] = 1.0
    return mu_phi, std_phi, mu_s2, std_s2


@torch.no_grad()
def sample_and_save(unet, shape, cond_vector, device, save_path, timesteps=1000):
    unet.eval()
    xt = torch.randn(shape, device=device)
    cond = cond_vector.to(device)

    for t in reversed(range(timesteps)):
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        lc = unet(xt, tt, cond=cond)
        l0 = lc
        x0 = torch.tanh(l0)
        tf = torch.tensor([t], device=device, dtype=torch.float)
        abar = torch.cos(((tf / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        abar_prev = torch.cos((((tf - 1) / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2 if t > 0 else torch.tensor(
            1.0, device=device)
        alpha_t = abar / abar_prev
        beta_t = 1 - alpha_t
        mean = torch.sqrt(abar_prev) * beta_t / (1 - abar) * x0 + torch.sqrt(alpha_t) * (1 - abar_prev) / (
                    1 - abar) * xt
        xt = mean + torch.sqrt(torch.clamp(((1 - abar_prev) / (1 - abar)) * beta_t, min=1e-20)) * torch.randn_like(
            xt) if t > 0 else mean

    prob = (xt + 1) / 2
    binary = (prob > 0.5).float().cpu().numpy().squeeze()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, binary)

    img_vol = ((1.0 - binary) * 255).astype(np.uint8)
    slice_dir = save_path.parent.parent / "slices_visuals"
    slice_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_vol[img_vol.shape[0] // 2], 'L').save(slice_dir / f"{save_path.stem}_mid.png")


# ---------------- Main ----------------
def main(args):
    acc = Accelerator(mixed_precision=args.mixed_precision)
    out_dir = Path(args.output_dir)
    sample_dir = out_dir / "samples"

    if acc.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

    print("Computing Dataset Stats...")

    ds = NpyRockDataset(args.data_dir, sub_volume_depth=args.train_depth, train_resolution=args.train_resolution,
                        pore_value=args.pore_value)
    mu_phi, std_phi, mu_s2, std_s2 = compute_dataset_stats(ds, args.s2_lags)

    # cond_dim matches training: 1 (phi) + len(s2)
    cond_dim = 1 + len(args.s2_lags)
    model = VoxelDiT(
        input_size=(args.train_depth, args.train_resolution, args.train_resolution),
        patch_size=args.patch_size,
        depth=args.model_depth,
        num_heads=args.num_heads,
        window_size=(args.window_size, args.window_size, args.window_size),
        cond_dim=cond_dim
    )

    if args.ckpt_path is not None:
        print(f"Loading checkpoint: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No checkpoint path provided, using random weights!")

    model.to(acc.device)
    model.eval()

    # Setup stats tensors
    t_mu_phi = torch.tensor(mu_phi, device=acc.device)
    t_std_phi = torch.tensor(std_phi, device=acc.device)
    t_mu_s2 = torch.tensor(mu_s2, device=acc.device)
    t_std_s2 = torch.tensor(std_s2, device=acc.device)

    # Sampling Loop
    all_indices = np.arange(len(ds))
    replace = True if len(ds) < 100 else False
    chosen_indices = np.random.choice(all_indices, 100, replace=replace)

    # Save indices record
    with open(out_dir / "sampled_indices_record_fixed.txt", "w") as f:
        f.write(",".join(map(str, chosen_indices)))

    print("Starting Sampling...")
    for i, idx in enumerate(tqdm(chosen_indices)):
        # FIXED: Only one unsqueeze to make it (1, 1, D, H, W)
        # ds[idx] -> [1, D, H, W]
        # unsqueeze(0) -> [1, 1, D, H, W]
        gt_vol = ds[idx].unsqueeze(0).to(acc.device)

        with torch.no_grad():
            val_phi = gt_vol.mean(dim=(1, 2, 3, 4))  # Scalar [1]
            val_s2 = calc_s2_on_tensor(gt_vol, args.s2_lags)  # Vector [1, Lags]

            z_phi_val = (val_phi - t_mu_phi) / t_std_phi
            z_s2_val = (val_s2 - t_mu_s2) / t_std_s2

            # z_phi_val is [1], needs to be [1,1] for cat
            cond_vec = torch.cat([z_phi_val.unsqueeze(1), z_s2_val], dim=1)

        save_name = sample_dir / f"gen_refGT_{idx:04d}_seq_{i:03d}.npy"
        sample_and_save(
            model,
            (1, 1, args.train_depth, args.train_resolution, args.train_resolution),
            cond_vec,
            acc.device,
            save_name,
            timesteps = args.timesteps
        )
    print("Sampling Finished Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ketton Sampling")

    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Ketton NPY dataset (for guidance stats)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .pth checkpoint")

    # Model Params (Must match training)
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--window_size", type=int, default=12, help="Swin window size")
    parser.add_argument("--model_depth", type=int, default=16, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=12, help="Attention heads")

    # Sampling Params
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision")  # <--- 新增这行

    # Data Dimensions
    parser.add_argument("--train_depth", type=int, default=192, help="Volume depth")
    parser.add_argument("--train_resolution", type=int, default=192, help="Volume resolution")
    parser.add_argument("--s2_lags", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128], help="S2 lags")
    parser.add_argument("--pore_value", type=int, default=1, help="Pore value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Apply Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)