#!/usr/bin/env python3
# Optimized for Connectivity: Added StyleLoss (Fixed Weight) and adjusted defaults
import os, argparse, json, math, torch, numpy as np, bitsandbytes as bnb
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from einops import rearrange
from timm.models.layers import trunc_normal_
import logging
import random


# ---------------- Losses ----------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1 - (2. * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class FFTLoss(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits).float()
        target = target.float()
        pf = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm='ortho')
        tf = torch.fft.rfftn(target, dim=(-3, -2, -1), norm='ortho')
        return F.l1_loss(torch.view_as_real(pf), torch.view_as_real(tf))


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[[[-1, -1, -1], [-1, -2, -1], [-1, -1, -1]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 1, 1], [1, 2, 1], [1, 1, 1]]]], dtype=torch.float32).unsqueeze(1)
        self.ks = [k, k.permute(0, 1, 3, 2, 4), k.permute(0, 1, 4, 2, 3)]

    def forward(self, pred_logits, target):
        p = torch.sigmoid(pred_logits).float()
        t = target.float()
        self.ks = [kk.to(p.device) for kk in self.ks]
        loss_accum = 0.0
        for kk in self.ks:
            gp = F.conv3d(p, kk, padding='same')
            gt = F.conv3d(t, kk, padding='same')
            loss_accum += F.l1_loss(gp, gt)
        return loss_accum / 3.0


class StyleLoss(nn.Module):
    def __init__(self): super().__init__()

    def gram(self, x):
        b, c, d, h, w = x.size()
        f = x.view(b, c, d * h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G.div(c * d * h * w)

    def forward(self, logits, target):
        p = torch.sigmoid(logits).float()
        t = target.float()
        return F.l1_loss(self.gram(p), self.gram(t))


class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIM3DLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                      window_size).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(
            _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2);
        mu2_sq = mu2.pow(2);
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2;
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, logits, target):
        img1 = torch.sigmoid(logits);
        img2 = target
        if self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.window.to(img1.device).type_as(img1)
        return 1.0 - self._ssim(img1, img2, window, self.window_size, self.channel)


# ---------------- Model ----------------
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
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, self.grid_size, use_reentrant=False)
            else:
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
        return vol.squeeze(0)


def bit_diffusion_forward(x0, t, timesteps=1000):
    x0_map = 2 * x0 - 1
    alpha_bar = torch.cos(((t / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    noise = torch.randn_like(x0_map)
    return torch.sqrt(alpha_bar.view(-1, 1, 1, 1, 1)) * x0_map + torch.sqrt(1 - alpha_bar.view(-1, 1, 1, 1, 1)) * noise


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
        x0 = dataset[i].unsqueeze(0)
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
    np.save(save_path, binary)
    img_vol = ((1.0 - binary) * 255).astype(np.uint8)
    slice_dir = save_path.parent / (save_path.stem + "_slices")
    slice_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_vol[img_vol.shape[0] // 2], 'L').save(slice_dir / "slice_mid.png")


# ---------------- Main ----------------
def main(args):
    seed = args.seed
    torch.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    acc = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])

    out_dir = Path(args.output_dir)
    if acc.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=out_dir / "training_log.txt", level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console = logging.StreamHandler();
        console.setLevel(logging.INFO);
        logging.getLogger('').addHandler(console)
        logging.info(f"Connectivity Opt Training. Out: {args.output_dir}")
        logging.info(
            f"Params: G={args.grad_loss_weight}, F={args.fft_loss_weight}, Style={args.style_loss_weight}, Win={args.window_size}")

    ckpt_dir = out_dir / "checkpoints";
    sample_dir = out_dir / "samples"
    if acc.is_main_process: ckpt_dir.mkdir(exist_ok=True); sample_dir.mkdir(exist_ok=True)

    ds = NpyRockDataset(args.data_dir, sub_volume_depth=args.train_depth, train_resolution=args.train_resolution,
                        pore_value=args.pore_value)

    mu_phi, std_phi, mu_s2, std_s2 = compute_dataset_stats(ds, args.s2_lags)
    if acc.is_main_process: logging.info(f"Stats: Mu_Phi={mu_phi:.4f}, Mu_S2={mu_s2}")

    cond_dim = 1 + len(args.s2_lags)
    model = VoxelDiT(input_size=(args.train_depth, args.train_resolution, args.train_resolution),
                     patch_size=args.patch_size, depth=args.model_depth, num_heads=args.num_heads,
                     window_size=(args.window_size, args.window_size, args.window_size), cond_dim=cond_dim)

    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_train_epochs)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model, opt, dl, lr_scheduler = acc.prepare(model, opt, dl, lr_scheduler)

    dice_loss = DiceLoss();
    fft_loss = FFTLoss();
    grad_loss = GradientLoss();
    ssim_loss = SSIM3DLoss(window_size=11);
    style_loss = StyleLoss()
    t_mu_phi = torch.tensor(mu_phi, device=acc.device);
    t_std_phi = torch.tensor(std_phi, device=acc.device)
    t_mu_s2 = torch.tensor(mu_s2, device=acc.device);
    t_std_s2 = torch.tensor(std_s2, device=acc.device)

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dl, disable=not acc.is_local_main_process, desc=f"Ep {epoch + 1}")
        for x0 in pbar:
            curr_phi = x0.mean(dim=(1, 2, 3, 4))
            z_phi = (curr_phi - t_mu_phi) / t_std_phi
            curr_s2 = calc_s2_on_tensor(x0, args.s2_lags)
            z_s2 = (curr_s2 - t_mu_s2) / t_std_s2
            cond_target = torch.cat([z_phi.unsqueeze(1), z_s2], dim=1)

            if np.random.rand() < args.cfg_dropout_prob:
                cond_input = torch.zeros_like(cond_target)
            else:
                cond_input = cond_target

            t = torch.randint(0, 1000, (x0.shape[0],), device=acc.device).long()
            xt = bit_diffusion_forward(x0, t, 1000)
            logits = model(xt, t, cond=cond_input)

            l_bce = F.binary_cross_entropy_with_logits(logits, x0)
            l_dice = dice_loss(logits, x0);
            l_ssim = ssim_loss(logits, x0);
            l_fft = fft_loss(logits, x0);
            l_grad = grad_loss(logits, x0)
            l_style = style_loss(logits, x0)

            # Weighted Sum: Style weight now reduced to 10.0 (from 1000.0)
            loss = (l_bce + l_dice) + (l_fft * args.fft_loss_weight) + (l_ssim * args.ssim_loss_weight) + (
                        l_grad * args.grad_loss_weight) + (l_style * args.style_loss_weight)

            acc.backward(loss);
            acc.clip_grad_norm_(model.parameters(), 1.0);
            opt.step();
            opt.zero_grad()
            epoch_loss += loss.item();
            pbar.set_postfix(L=loss.item(), G=l_grad.item())

        lr_scheduler.step()
        if acc.is_main_process: logging.info(f"Epoch {epoch + 1} Avg Loss: {epoch_loss / len(dl):.6f}")

        if (epoch + 1) % 50 == 0 or epoch == args.num_train_epochs - 1:
            acc.wait_for_everyone()
            if acc.is_main_process:
                torch.save(acc.unwrap_model(model).state_dict(), ckpt_dir / f"epoch_{epoch + 1:04d}.pth")
                logging.info(f"Checkpoint saved: {epoch + 1}")

    acc.wait_for_everyone()
    if acc.is_main_process:
        logging.info("Starting Sampling...")
        model = acc.unwrap_model(model);
        model.eval()
        all_indices = np.arange(len(ds))
        replace = True if len(ds) < 100 else False
        chosen_indices = np.random.choice(all_indices, 100, replace=replace)
        with open(out_dir / "sampled_indices_record.txt", "w") as f:
            f.write(",".join(map(str, chosen_indices)))

        for i, idx in enumerate(tqdm(chosen_indices)):
            gt_vol = ds[idx].unsqueeze(0).to(acc.device)
            with torch.no_grad():
                val_phi = gt_vol.mean(dim=(1, 2, 3, 4));
                val_s2 = calc_s2_on_tensor(gt_vol, args.s2_lags)
                z_phi_val = (val_phi - t_mu_phi) / t_std_phi;
                z_s2_val = (val_s2 - t_mu_s2) / t_std_s2
                cond_vec = torch.cat([z_phi_val.unsqueeze(1), z_s2_val], dim=1)
            sample_and_save(model, (1, 1, args.train_depth, args.train_resolution, args.train_resolution), cond_vec,
                            acc.device, sample_dir / f"gen_refGT_{idx:04d}_seq_{i:03d}.npy")
        logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ketton Connectivity-Optimized Training")

    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing NPY files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs")

    # Model Architecture
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size (default: 8)")
    parser.add_argument("--window_size", type=int, default=12, help="Swin window size (default: 12)")
    parser.add_argument("--model_depth", type=int, default=16, help="Transformer depth (default: 16)")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads (default: 12)")

    # Training Hyperparams
    parser.add_argument("--num_train_epochs", type=int, default=300, help="Total training epochs (default: 300)")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per GPU (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode")
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.15,
                        help="Condition dropout probability (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Specialized Loss Weights (Connectivity Optimization)
    parser.add_argument("--fft_loss_weight", type=float, default=0.5, help="Weight for FFT loss (frequency domain)")
    parser.add_argument("--grad_loss_weight", type=float, default=0.2,
                        help="Weight for Gradient loss (edge preservation)")
    parser.add_argument("--ssim_loss_weight", type=float, default=0.2, help="Weight for 3D SSIM loss")
    parser.add_argument("--style_loss_weight", type=float, default=10.0,
                        help="Weight for Style loss (texture consistency)")

    # Data & Conditioning
    parser.add_argument("--train_depth", type=int, default=192, help="Training crop depth (default: 192)")
    parser.add_argument("--train_resolution", type=int, default=192, help="Training crop resolution (default: 192)")
    parser.add_argument("--s2_lags", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128],
                        help="Lags for S2 correlation function")
    parser.add_argument("--pore_value", type=int, default=1, help="Value representing pores in NPY (default: 1)")

    args = parser.parse_args()
    main(args)