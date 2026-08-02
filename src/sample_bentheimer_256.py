#!/usr/bin/env python3
# coding: utf-8
"""
使用 DiT φ-条件模型从 checkpoint 采样 3D 体素体积（单进程版）。
- 不再用 `accelerate`，单卡模式，减少显存占用。
- 条件 φ*：
    1) 若 --phi_target 非空：所有样本用同一个 φ*
    2) 否则若提供 --phi_npz：从 npz["phi"] 中有放回抽样 φ*
    3) 否则：从 N(mu_phi, sigma_phi) 中采样 φ* 并裁剪到 [0,1]
"""

import os
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from tqdm.auto import tqdm
from PIL import Image


# -------------------------------
# 基本模块：PatchEmbed / SwinBlock / UpBlock
# -------------------------------

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_channels=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b e d h w -> b (d h w) e')
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(-1,
                     window_size[0] * window_size[1] * window_size[2],
                     C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    B = int(
        windows.shape[0]
        / (D * H * W
           / window_size[0]
           / window_size[1]
           / window_size[2])
    )
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class SwinTransformer3DBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(4, 4, 4),
                 shift_size=(0, 0, 0)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x, grid_size):
        D, H, W = grid_size
        B, L, C = x.shape
        assert L == D * H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        if any(self.shift_size):
            shifted = torch.roll(
                x,
                shifts=(-self.shift_size[0],
                        -self.shift_size[1],
                        -self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted = x

        x_windows = window_partition(shifted, self.window_size)
        x_windows = x_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            C,
        )
        attn_windows = self.attn(x_windows)

        shifted = window_reverse(
            attn_windows, self.window_size, D, H, W
        )

        if any(self.shift_size):
            x = torch.roll(
                shifted,
                shifts=self.shift_size,
                dims=(1, 2, 3),
            )
        else:
            x = shifted

        x = x.view(B, D * H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode='trilinear',
            align_corners=False,
        )
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=3, padding=1
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.act(x)
        return x


# -------------------------------
# VoxelDiT 主干
# -------------------------------

class VoxelDiT(nn.Module):
    def __init__(self,
                 input_size=(256, 256, 256),
                 patch_size=(16, 16, 16),
                 in_channels=1,
                 embed_dim=768,
                 depth=16,
                 num_heads=12,
                 window_size=(4, 4, 4),
                 use_checkpoint=True,
                 cond_dim=1):
        super().__init__()

        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.grid_size = tuple(
            s // p for s, p in zip(input_size, self.patch_size)
        )
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )

        self.patch_embed = PatchEmbed3D(
            self.patch_size, in_channels, embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        blocks = []
        for i in range(depth):
            shift = (0, 0, 0) if (i % 2 == 0) else tuple(
                ws // 2 for ws in window_size
            )
            blocks.append(
                SwinTransformer3DBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(embed_dim)

        self.decoder = nn.ModuleList()
        current_channels = embed_dim
        num_ups = int(round(math.log2(self.patch_size[0])))
        for i in range(num_ups):
            next_ch = max(embed_dim // (2 ** (i + 1)),
                          in_channels * 8)
            self.decoder.append(
                UpBlock3D(current_channels, next_ch)
            )
            current_channels = next_ch

        self.final_conv = nn.Conv3d(
            current_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _time_emb(self, t, dim):
        f = torch.exp(
            -math.log(10000)
            * torch.arange(dim // 2, device=t.device)
            / (dim // 2)
        )
        return torch.cat(
            [torch.sin(t[:, None] * f),
             torch.cos(t[:, None] * f)],
            dim=-1,
        )

    def forward(self, x, t, cond=None):
        # patch + pos
        x = self.patch_embed(x) + self.pos_embed

        # 时间嵌入
        t_embed = self.time_embed(self._time_emb(t, 768))
        x = x + t_embed.unsqueeze(1)

        # φ 条件
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(
                    (x.size(0), self.cond_dim),
                    device=x.device,
                )
            x = x + self.cond_mlp(cond).unsqueeze(1)

        # Transformer blocks
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, self.grid_size,
                    use_reentrant=False,
                )
            else:
                x = blk(x, self.grid_size)

        x = self.norm_out(x)
        x = rearrange(
            x,
            'b (d h w) e -> b e d h w',
            d=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2],
        )

        # decoder
        for up in self.decoder:
            x = up(x)
        x = self.final_conv(x)
        return x


# -------------------------------
# 加载 checkpoint（适配 cond_mlp 形状）
# -------------------------------

def safe_load_pretrained(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    own = model.state_dict()
    adapted, removed = [], []

    k = "cond_mlp.0.weight"
    if k in sd and k in own:
        w_old, w_new = sd[k], own[k]
        if w_old.shape != w_new.shape:
            if (w_old.dim() == 2 and w_new.dim() == 2
                    and w_old.shape[0] == w_new.shape[0]
                    and w_new.shape[1] == 1):
                sd[k] = w_old.mean(dim=1, keepdim=True)
                adapted.append(k)
            else:
                sd.pop(k)
                removed.append(k)
                b = "cond_mlp.0.bias"
                if b in sd:
                    sd.pop(b)
                    removed.append(b)

    k2 = "cond_mlp.2.weight"
    if k2 in sd and k2 in own:
        w_old2, w_new2 = sd[k2], own[k2]
        if w_old2.shape != w_new2.shape:
            sd.pop(k2)
            removed.append(k2)
            b2 = "cond_mlp.2.bias"
            if b2 in sd:
                sd.pop(b2)
                removed.append(b2)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Loaded pretrained from {ckpt_path}. "
        f"Adapted:{adapted} Removed:{removed} "
        f"Missing:{len(missing)} Unexpected:{len(unexpected)}"
    )
    if len(missing) > 0:
        print("Missing keys (first 10):", missing[:10])
    if len(unexpected) > 0:
        print("Unexpected keys (first 10):", unexpected[:10])


# -------------------------------
# 单个样本的 DDPM 反演
# -------------------------------

@torch.no_grad()
def sample_one_volume(model,
                      device,
                      z_phi_eval: float,
                      timesteps: int = 1000,
                      cfg_scale: float = 1.0,
                      shape=(1, 1, 256, 256, 256),
                      sample_index: int = 0):
    model.eval()
    B = shape[0]
    xt = torch.randn(shape, device=device)

    cond = torch.full(
        (B, 1),
        float(z_phi_eval),
        device=device,
        dtype=torch.float32,
    )
    null = torch.zeros_like(cond)

    desc = f"sample_{sample_index:03d}"
    for t in tqdm(
        reversed(range(timesteps)),
        total=timesteps,
        desc=desc,
    ):
        tt = torch.full(
            (B,),
            t,
            device=device,
            dtype=torch.long,
        )

        lc = model(xt, tt, cond=cond)
        if cfg_scale != 1.0:
            lu = model(xt, tt, cond=null)
            l0 = lu + cfg_scale * (lc - lu)
        else:
            l0 = lc

        x0 = torch.tanh(l0)

        tf = torch.tensor(
            [t], device=device, dtype=torch.float32
        )
        abar = torch.cos(
            ((tf / timesteps) + 0.008) / 1.008
            * math.pi * 0.5
        ) ** 2
        if t > 0:
            abar_prev = torch.cos(
                (((tf - 1) / timesteps) + 0.008) / 1.008
                * math.pi * 0.5
            ) ** 2
        else:
            abar_prev = torch.tensor(
                [1.0], device=device, dtype=torch.float32
            )

        alpha_t = abar / abar_prev
        beta_t = 1.0 - alpha_t

        mean = (
            torch.sqrt(abar_prev) * beta_t / (1.0 - abar) * x0
            + torch.sqrt(alpha_t)
            * (1.0 - abar_prev) / (1.0 - abar) * xt
        )

        if t > 0:
            var = ((1.0 - abar_prev) / (1.0 - abar)) * beta_t
            xt = mean + torch.sqrt(
                torch.clamp(var, min=1e-20)
            ) * torch.randn_like(xt)
        else:
            xt = mean

    prob = (xt + 1.0) / 2.0
    binary = (prob > 0.5).float()
    return binary


# -------------------------------
# 主入口
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample 3D voxel volumes from phi-conditioned DiT (single process, per-sample phi from npz)."
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to DiT checkpoint, e.g., epoch_0300.pth"
    )
    parser.add_argument(
        "--cond_stats_path", type=str, required=True,
        help="Path to cond_stats.json (containing mu_phi, sigma_phi)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        required=True,
        help="Output directory for samples, each sample will be saved in a subfolder"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Total number of samples to generate (default: 100)"
    )
    parser.add_argument(
        "--depth", type=int, default=256,
        help="Volume depth D (default: 256)"
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="横向分辨率 H=W (默认 256)"
    )
    parser.add_argument(
        "--patch_size", type=int, default=16,
        help="DiT patch_size, must match training (default: 16)"
    )
    parser.add_argument(
        "--model_depth", type=int, default=16,
        help="Transformer depth, must match training (default: 16)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12,
        help="Number of attention heads, must match training (default: 12)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Total diffusion timesteps (default: 1000)"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=1.0,
        help="Classifier-free guidance scale (default: 1.0 -> no CFG)"
    )
    parser.add_argument(
        "--phi_target", type=float, default=None,
        help="If specified, use this phi for all samples; otherwise sample from phi_npz or Gaussian distribution"
    )
    parser.add_argument(
        "--phi_npz", type=str, default=None,
        help="Path to npz file containing real porosity array 'phi'. If provided, sample phi_target from it with replacement."
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed (default: 1234)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use, e.g., 'cuda', 'cuda:0', 'cpu'"
    )

    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA 不可用，自动切换到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] using {device}")

    # 读取 cond_stats.json
    cond_stats_path = Path(args.cond_stats_path)
    if not cond_stats_path.exists():
        raise FileNotFoundError(
            f"cond_stats.json not found: {cond_stats_path}"
        )
    with open(cond_stats_path, "r") as f:
        stats = json.load(f)
    mu_phi = float(stats["mu_phi"])
    sigma_phi = float(stats["sigma_phi"])
    print(f"[cond_stats] mu_phi = {mu_phi:.6f}, sigma_phi = {sigma_phi:.6f}")

    # 输出目录
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[output] samples will be saved under: {out_root}")

    # 采样 φ_target
    rng = np.random.default_rng(args.seed)

    if args.phi_target is not None:
        phi_targets = np.full(
            args.num_samples,
            float(args.phi_target),
            dtype=np.float32,
        )
        print(f"[phi] 使用固定 φ_target = {args.phi_target:.6f} 生成所有样本")
    elif args.phi_npz is not None:
        npz_path = Path(args.phi_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"phi_npz not found: {npz_path}")
        data = np.load(npz_path)
        if "phi" not in data:
            raise KeyError(
                f"{npz_path} 中未找到键 'phi'，请确认 npz 文件内容"
            )
        phi_data = np.asarray(data["phi"], dtype=np.float32)
        phi_data = phi_data[np.isfinite(phi_data)]
        if phi_data.size == 0:
            raise ValueError(f"{npz_path} 中 'phi' 为空或无有效值")

        print(f"[phi_npz] 加载 {phi_data.size} 个真实 φ，用于采样分布")
        print(
            f"[phi_npz] data mean={phi_data.mean():.6f}, "
            f"std={phi_data.std():.6f}, "
            f"min={phi_data.min():.6f}, max={phi_data.max():.6f}"
        )

        phi_targets = rng.choice(
            phi_data,
            size=args.num_samples,
            replace=True,
        ).astype(np.float32)
        print(
            f"[phi_samples] 采样得到 φ_target: "
            f"mean={phi_targets.mean():.6f}, "
            f"std={phi_targets.std():.6f}, "
            f"min={phi_targets.min():.6f}, max={phi_targets.max():.6f}"
        )
    else:
        phi_targets = rng.normal(
            loc=mu_phi,
            scale=sigma_phi,
            size=args.num_samples,
        ).astype(np.float32)
        phi_targets = np.clip(phi_targets, 0.0, 1.0)
        print(
            f"[phi_normal] 从 N(mu_phi,sigma_phi) 采样 φ_target: "
            f"mean={phi_targets.mean():.6f}, "
            f"std={phi_targets.std():.6f}, "
            f"min={phi_targets.min():.6f}, max={phi_targets.max():.6f}"
        )

    # 转成 z_phi
    z_targets = (phi_targets - mu_phi) / sigma_phi
    np.save(out_root / "phi_targets.npy", phi_targets)
    np.save(out_root / "z_phi_targets.npy", z_targets)
    print("[phi_targets] 已保存 phi_targets.npy / z_phi_targets.npy")

    # 构建模型 & 加载权重
    input_size = (args.depth, args.resolution, args.resolution)
    model = VoxelDiT(
        input_size=input_size,
        patch_size=(args.patch_size,) * 3,
        in_channels=1,
        embed_dim=768,
        depth=args.model_depth,
        num_heads=args.num_heads,
        window_size=(4, 4, 4),
        use_checkpoint=False,
        cond_dim=1,
    )
    model.to(device)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    print(f"[ckpt] loading from: {ckpt_path}")
    safe_load_pretrained(model, str(ckpt_path))
    model.eval()

    shape = (1, 1, args.depth, args.resolution, args.resolution)
    total_samples = args.num_samples

    # 逐个样本采样 + 存盘
    for idx in range(total_samples):
        z_phi = float(z_targets[idx])
        phi_t = float(phi_targets[idx])

        vol = sample_one_volume(
            model=model,
            device=device,
            z_phi_eval=z_phi,
            timesteps=args.timesteps,
            cfg_scale=args.cfg_scale,
            shape=shape,
            sample_index=idx,
        )

        # 转成黑孔隙 / 白固体的 8bit 图像
        out = (1.0 - vol).squeeze(0).squeeze(0)
        out = out.detach().cpu().numpy().astype(np.float32)
        out = (out * 255.0).astype(np.uint8)

        sample_dir = out_root / f"sample_{idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        with open(sample_dir / "phi_target.txt", "w") as f:
            f.write(f"{phi_t:.8f}\n")

        D = out.shape[0]
        for z in range(D):
            img = Image.fromarray(out[z], mode="L")
            img.save(sample_dir / f"slice_{z:04d}.png")

    print("[done] all samples generated.")


if __name__ == "__main__":
    main()
