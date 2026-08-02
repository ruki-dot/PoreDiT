#!/usr/bin/env python3
# Modified based on 'sample_phi_cond_tiled_large_mixed.py'
# Fix: Solves Porosity Drift via Global Coherent Noise in Mixed Precision
# GPU: 1

import os, math, argparse, numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm
from timm.models.layers import trunc_normal_
from PIL import Image


# ----------------- 模型结构 -----------------

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_channels=1, embed_dim=768):
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
        self.qkv = nn.Linear(dim, dim * 3);
        self.proj = nn.Linear(dim, dim);
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformer3DBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(4, 4, 4), shift_size=(0, 0, 0)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim);
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.window_size = window_size;
        self.shift_size = shift_size

    def forward(self, x, grid_size):
        D, H, W = grid_size;
        B, L, C = x.shape;
        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)
        if any(self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        xw = window_partition(x, self.window_size)
        xw = xw.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        aw = self.attn(xw)
        x = window_reverse(aw, self.window_size, D, H, W)
        if any(self.shift_size):
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))
        x = x.view(B, D * H * W, C);
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1);
        self.act = nn.GELU()

    def forward(self, x): return self.act(self.conv(self.up(x)))


class VoxelDiT(nn.Module):
    def __init__(self, input_size=(256, 256, 256), patch_size=(16, 16, 16), in_channels=1, embed_dim=768, depth=16,
                 num_heads=12, window_size=(4, 4, 4), cond_dim=1):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        self.grid_size = tuple(s // p for s, p in zip(input_size, self.patch_size))
        self.window_size = window_size;
        self.cond_dim = cond_dim
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_embed = PatchEmbed3D(self.patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.time_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.blocks = nn.ModuleList([SwinTransformer3DBlock(embed_dim, num_heads, window_size,
                                                            (0, 0, 0) if (i % 2 == 0) else tuple(
                                                                ws // 2 for ws in window_size)) for i in range(depth)])
        self.norm_out = nn.LayerNorm(embed_dim);
        self.decoder = nn.ModuleList()
        current = embed_dim;
        num_ups = int(round(math.log2(self.patch_size[0])))
        for i in range(num_ups):
            nxt = max(embed_dim // (2 ** (i + 1)), in_channels * 8)
            self.decoder.append(UpBlock3D(current, nxt));
            current = nxt
        self.final_conv = nn.Conv3d(current, in_channels, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        for m in self.modules():
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
        t_emb = self.time_embed(self._time_emb(t, 768))
        x = x + t_emb.unsqueeze(1)
        if self.cond_dim > 0:
            if cond is None: cond = torch.zeros((x.size(0), self.cond_dim), device=x.device)
            x = x + self.cond_mlp(cond).unsqueeze(1)
        for blk in self.blocks: x = blk(x, self.grid_size)
        x = self.norm_out(x)
        x = rearrange(x, 'b (d h w) e -> b e d h w', d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2])
        for up in self.decoder: x = up(x)
        return self.final_conv(x)


def safe_load_pretrained(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    own = model.state_dict()
    k = "cond_mlp.0.weight"
    if k in sd and k in own and sd[k].shape != own[k].shape:
        if sd[k].dim() == 2 and sd[k].shape[0] == own[k].shape[0] and own[k].shape[1] == 1:
            sd[k] = sd[k].mean(dim=1, keepdim=True)
        else:
            sd.pop(k, None);
            sd.pop("cond_mlp.0.bias", None)
    model.load_state_dict(sd, strict=False)
    print(f"Loaded ckpt from {ckpt_path}")
    return model


# ----------------- 辅助函数 -----------------

def calculate_gt_stats(dataset_dir, pore_value=0):
    files = sorted(list(Path(dataset_dir).glob("*.npy")))
    if not files: return 0.0, 1.0
    phis = []
    print(f"Analyzing {len(files)} GT files for Z-score calculation...")
    for f in tqdm(files[:200], desc="Dataset Stats"):  # Sample 200 is enough
        arr = np.load(f)
        mask = (~arr) if (arr.dtype == np.bool_ and pore_value == 0) else (arr == pore_value)
        phis.append(mask.mean())
    return np.mean(phis), max(np.std(phis), 0.02)


def sliding_starts(full_size, tile_size, overlap):
    assert tile_size <= full_size
    step = tile_size - overlap
    starts = list(range(0, full_size - tile_size, step))
    starts.append(full_size - tile_size)
    return starts


def make_window_3d(tile_size, device, dtype):
    w1d = torch.hann_window(tile_size, device=device, dtype=torch.float32) + 1e-2
    w1 = w1d.view(1, 1, -1, 1, 1);
    w2 = w1d.view(1, 1, 1, -1, 1);
    w3 = w1d.view(1, 1, 1, 1, -1)
    return (w1 * w2 * w3).to(dtype=dtype)


# ----------------- 核心采样逻辑 -----------------

@torch.no_grad()
def sample_tiled_coherent(model, device, out_dir,
                          timesteps=1000, z_phi_eval=0.0,
                          full_size=1024, tile_size=256, overlap=64,
                          xt_dtype=torch.float16):
    out_dir = Path(out_dir);
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 存储层：使用 fp16 存储大张量，避免显存爆炸
    xt = torch.randn((1, 1, full_size, full_size, full_size), device=device, dtype=xt_dtype)
    cond = torch.tensor([[z_phi_eval]], device=device, dtype=torch.float32)

    starts_z = sliding_starts(full_size, tile_size, overlap)
    starts_y = sliding_starts(full_size, tile_size, overlap)
    starts_x = sliding_starts(full_size, tile_size, overlap)

    # 权重窗使用 fp16 存储
    w3d = make_window_3d(tile_size, device=device, dtype=xt_dtype)

    print(f"Sampling {full_size}^3 with Global Coherent Noise (Mixed Precision)")

    for t in tqdm(reversed(range(timesteps)), total=timesteps, desc=f"DDPM"):
        tt = torch.full((1,), t, device=device, dtype=torch.long)
        tf = torch.tensor([t], device=device, dtype=torch.float32)

        # 参数计算使用 fp32
        abar = torch.cos(((tf / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        abar_prev = torch.cos((((tf - 1) / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2 if t > 0 else torch.tensor(
            1.0, device=device)
        alpha_t = abar / abar_prev;
        beta_t = 1 - alpha_t

        abar_v = abar.view(1, 1, 1, 1, 1);
        abar_prev_v = abar_prev.view(1, 1, 1, 1, 1)
        alpha_v = alpha_t.view(1, 1, 1, 1, 1);
        beta_v = beta_t.view(1, 1, 1, 1, 1)

        xt_next = torch.zeros_like(xt);
        weight_sum = torch.zeros_like(xt)

        # ---------------- 生成全局噪声 (FP16存储) ----------------
        # 这一步保证了相邻Patch在重叠区域使用的是同一份噪声，防止方差抵消
        if t > 0:
            global_noise = torch.randn((1, 1, full_size, full_size, full_size), device=device, dtype=xt_dtype)
        else:
            global_noise = None
        # -------------------------------------------------------------

        for z0 in starts_z:
            z1 = z0 + tile_size
            for y0 in starts_y:
                y1 = y0 + tile_size
                for x0 in starts_x:
                    x1 = x0 + tile_size

                    # 切片并转为 fp32 进行计算，避免全白问题
                    sub = xt[:, :, z0:z1, y0:y1, x0:x1]
                    sub_f = sub.to(torch.float32)

                    logits = model(sub_f, tt, cond=cond)
                    x0_hat = torch.tanh(logits)  # [-1, 1]

                    mean = torch.sqrt(abar_prev_v) * beta_v / (1 - abar_v) * x0_hat \
                           + torch.sqrt(alpha_v) * (1 - abar_prev_v) / (1 - abar_v) * sub_f

                    if t > 0:
                        var = ((1 - abar_prev_v) / (1 - abar_v)) * beta_v
                        # 从全局噪声切片，并转为 fp32 参与计算
                        noise_patch = global_noise[:, :, z0:z1, y0:y1, x0:x1].to(torch.float32)
                        sub_next_f = mean + torch.sqrt(torch.clamp(var, min=1e-20)) * noise_patch
                    else:
                        sub_next_f = mean

                    # 算完转回 fp16 存储
                    sub_next = sub_next_f.to(xt_dtype)
                    xt_next[:, :, z0:z1, y0:y1, x0:x1] += sub_next * w3d
                    weight_sum[:, :, z0:z1, y0:y1, x0:x1] += w3d

        xt = xt_next / torch.clamp(weight_sum, min=1e-6)
        if global_noise is not None: del global_noise
        torch.cuda.empty_cache()

    # Save
    xt_cpu = xt.cpu()
    del xt;
    torch.cuda.empty_cache()
    D = xt_cpu.shape[2]
    print(f"Saving slices to {out_dir}")
    for z in range(D):
        slice_half = xt_cpu[0, 0, z]
        slice_f = slice_half.float()
        prob = (slice_f + 1) / 2
        binary = (prob > 0.5).numpy().astype('uint8')
        img = Image.fromarray((1 - binary) * 255, mode='L')
        img.save(out_dir / f"slice_{z:04d}.png")


def main():
    parser = argparse.ArgumentParser(description="Large 3D sampling with Coherent Noise Fix")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to ground truth dataset (for porosity calibration)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for generated slices")
    parser.add_argument("--full_size", type=int, default=1024, help="Size of the generated volume (default: 1024)")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps (reduce for fast testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, full_size={args.full_size}, timesteps={args.timesteps}")

    # 1. 自动计算准确的 GT 统计值
    mu_phi, sigma_phi = calculate_gt_stats(args.dataset_dir)
    target_phi = mu_phi
    z_eval = (target_phi - mu_phi) / sigma_phi  # 理论上是 0.0，但计算出来更严谨
    print(f"GT Mean Porosity: {mu_phi:.4f}, using z_eval: {z_eval:.4f}")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    model = VoxelDiT(cond_dim=1)
    model = safe_load_pretrained(model, args.ckpt_path)
    model.to(device)  # 保持模型在 FP32 (默认)
    model.eval()

    sample_tiled_coherent(
        model, device,
        out_dir=args.out_dir,
        z_phi_eval=z_eval,
        full_size=args.full_size,
        tile_size=256,
        overlap=64,
        xt_dtype=torch.float16,
        timesteps=args.timesteps
    )


if __name__ == "__main__":
    main()