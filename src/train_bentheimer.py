#!/usr/bin/env python3
# rev6b: 以目标 φ 为条件（cond_dim=1）；加载旧 ckpt 时自适配 cond_mlp 输入维度（9->1 均值收缩或跳过）
#  - 训练：cond = z_phi_target = (phi(x0)-mu_phi)/sigma_phi
#  - 条件损失：MSE( z_phi_pred, z_phi_target )
#  - CFG: 10~20% 条件 dropout
#  - 评测：--phi_target_eval 指定 φ*（未指定则用训练集均值，等价 z=0）
#  - 数值稳健：clamp/nan_to_num/grad clip；PNG 导出：孔隙=黑

import os, argparse, json, math, torch, numpy as np, bitsandbytes as bnb
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from einops import rearrange
from timm.models.layers import trunc_normal_

# ---------------- Losses ----------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits); probs = probs.view(-1); targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1 - (2. * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

class FFTLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits).float(); target = target.float()
        pf = torch.fft.rfftn(pred, dim=(-3,-2,-1)); tf = torch.fft.rfftn(target, dim=(-3,-2,-1))
        pf = torch.stack((pf.real, pf.imag), -1); tf = torch.stack((tf.real, tf.imag), -1)
        return F.l1_loss(pf, tf)

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[[[-1,-1,-1],[-1,-2,-1],[-1,-1,-1]],
                           [[0,0,0],[0,0,0],[0,0,0]],
                           [[1,1,1],[1,2,1],[1,1,1]]]], dtype=torch.float32).unsqueeze(1)
        self.ks = [k, k.permute(0,1,3,2,4), k.permute(0,1,4,2,3)]
    def forward(self, pred_logits, target):
        p = torch.sigmoid(pred_logits).float(); t = target.float()
        self.ks = [kk.to(p.device) for kk in self.ks]
        gp = [F.conv3d(p, kk, padding='same') for kk in self.ks]
        gt = [F.conv3d(t, kk, padding='same') for kk in self.ks]
        return sum(F.l1_loss(torch.abs(a), torch.abs(b)) for a,b in zip(gp,gt))/3.0

class StyleLoss(nn.Module):
    def __init__(self): super().__init__()
    def gram(self, x):
        b,c,d,h,w = x.size(); f = x.view(b,c,d*h*w); G = torch.bmm(f, f.transpose(1,2)); return G.div(c*d*h*w)
    def forward(self, pred_logits, target):
        p = torch.sigmoid(pred_logits).float(); t = target.float()
        return F.l1_loss(self.gram(p), self.gram(t))

# ---------------- Model ----------------
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_channels=1, embed_dim=768):
        super().__init__(); self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): x = self.proj(x); return rearrange(x, 'b e d h w -> b (d h w) e')

def window_partition(x, window_size):
    B,D,H,W,C = x.shape
    x = x.view(B, D//window_size[0], window_size[0], H//window_size[1], window_size[1], W//window_size[2], window_size[2], C)
    return x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, window_size[0]*window_size[1]*window_size[2], C)

def window_reverse(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D*H*W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D//window_size[0], H//window_size[1], W//window_size[2], window_size[0], window_size[1], window_size[2], -1)
    return x.permute(0,1,4,2,5,3,6,7).contiguous().view(B, D, H, W, -1)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__(); self.dim=dim; self.window_size=window_size; self.num_heads=num_heads; head_dim=dim//num_heads; self.scale=head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3); self.proj = nn.Linear(dim, dim); self.softmax=nn.Softmax(dim=-1)
    def forward(self, x):
        B_,N,C=x.shape; qkv=self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]; q=q*self.scale; attn = self.softmax(q @ k.transpose(-2,-1))
        return self.proj((attn @ v).transpose(1,2).reshape(B_,N,C))

class SwinTransformer3DBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(4,4,4), shift_size=(0,0,0)):
        super().__init__(); self.norm1=nn.LayerNorm(dim); self.attn=WindowAttention3D(dim,window_size,num_heads); self.norm2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,4*dim), nn.GELU(), nn.Linear(dim*4, dim)); self.window_size=window_size; self.shift_size=shift_size
    def forward(self, x, grid_size):
        D,H,W=grid_size; B,L,C=x.shape; shortcut=x; x=self.norm1(x); x=x.view(B,D,H,W,C)
        shifted = torch.roll(x, shifts=(-self.shift_size[0],-self.shift_size[1],-self.shift_size[2]), dims=(1,2,3)) if any(self.shift_size) else x
        xw = window_partition(shifted, self.window_size); xw = xw.view(-1, self.window_size[0]*self.window_size[1]*self.window_size[2], C)
        aw = self.attn(xw); shifted = window_reverse(aw, self.window_size, D,H,W)
        x = torch.roll(shifted, shifts=self.shift_size, dims=(1,2,3)) if any(self.shift_size) else shifted
        x = x.view(B, D*H*W, C); x = shortcut + x; return x + self.mlp(self.norm2(x))

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1); self.act = nn.GELU()
    def forward(self, x): return self.act(self.conv(self.up(x)))

class VoxelDiT(nn.Module):
    def __init__(self, input_size=(256,256,256), patch_size=(16,16,16), in_channels=1, embed_dim=768, depth=16, num_heads=12, window_size=(4,4,4),
                 use_checkpoint=True, cond_dim=1):
        super().__init__()
        self.patch_size=patch_size if isinstance(patch_size, tuple) else (patch_size,)*3
        self.grid_size=tuple(s//p for s,p in zip(input_size, self.patch_size))
        self.window_size=window_size; self.use_checkpoint=use_checkpoint
        num_patches=self.grid_size[0]*self.grid_size[1]*self.grid_size[2]
        self.patch_embed=PatchEmbed3D(self.patch_size, in_channels, embed_dim)
        self.pos_embed=nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.time_embed=nn.Sequential(nn.Linear(embed_dim,embed_dim), nn.SiLU(), nn.Linear(embed_dim,embed_dim))
        self.cond_dim = cond_dim
        if cond_dim>0: self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.blocks=nn.ModuleList([SwinTransformer3DBlock(embed_dim, num_heads, window_size, (0,0,0) if (i%2==0) else tuple(ws//2 for ws in window_size)) for i in range(depth)])
        self.norm_out=nn.LayerNorm(embed_dim)
        self.decoder = nn.ModuleList()
        current_channels = embed_dim; num_ups = int(round(math.log2(self.patch_size[0])))
        for i in range(num_ups):
            next_ch = max(embed_dim // (2**(i+1)), in_channels*8)
            self.decoder.append(UpBlock3D(current_channels, next_ch)); current_channels = next_ch
        self.final_conv = nn.Conv3d(current_channels, in_channels, kernel_size=3, padding=1)
        trunc_normal_(self.pos_embed, std=.02); self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02); 
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    def _time_emb(self, t, dim):
        f = torch.exp(-math.log(10000) * torch.arange(dim//2, device=t.device) / (dim//2));
        return torch.cat([torch.sin(t[:,None]*f), torch.cos(t[:,None]*f)], dim=-1)
    def forward(self, x, t, cond=None):
        x = self.patch_embed(x) + self.pos_embed
        t_embed = self.time_embed(self._time_emb(t, 768)); x = x + t_embed.unsqueeze(1)
        if self.cond_dim>0:
            if cond is None: cond = torch.zeros((x.size(0), self.cond_dim), device=x.device)
            x = x + self.cond_mlp(cond).unsqueeze(1)
        for blk in self.blocks:
            x = checkpoint(blk, x, self.grid_size, use_reentrant=False) if (self.use_checkpoint and self.training) else blk(x, self.grid_size)
        x = self.norm_out(x); x = rearrange(x, 'b (d h w) e -> b e d h w', d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2])
        for up in self.decoder: x = up(x)
        return self.final_conv(x)

# ---------------- Dataset ----------------
class NpyRockDataset(Dataset):
    def __init__(self, npy_dir, sub_volume_depth=256, train_resolution=256, pore_value=0):
        self.npy_files = sorted(list(Path(npy_dir).glob("*.npy"))); self.sub_volume_depth=sub_volume_depth; self.train_resolution=train_resolution; self.pore_value=pore_value
        print(f"Found {len(self.npy_files)} NPY files. Pore value in NPY assumed to be {self.pore_value}. (Will map pores->1)")
    def __len__(self): return len(self.npy_files)
    def __getitem__(self, idx):
        arr = np.load(self.npy_files[idx])
        if arr.dtype==np.bool_: mask = (~arr) if self.pore_value==0 else arr.copy()
        else: mask = (arr==self.pore_value)
        vol = torch.from_numpy(mask.astype(np.float32))
        D=vol.shape[0]
        if D>self.sub_volume_depth:
            start = torch.randint(0, D-self.sub_volume_depth+1, (1,)).item(); vol = vol[start:start+self.sub_volume_depth, :, :]
        vol = vol.unsqueeze(0)
        if vol.shape[2]!=self.train_resolution or vol.shape[3]!=self.train_resolution:
            vol = F.interpolate(vol.unsqueeze(0), size=(self.sub_volume_depth, self.train_resolution, self.train_resolution), mode='trilinear', align_corners=False).squeeze(0)
        return vol

# ---------------- Diffusion utils ----------------
def bit_diffusion_forward(x0, t, timesteps=1000):
    x0_map = 2*x0 - 1
    alpha_bar = torch.cos(((t/timesteps)+0.008)/1.008*math.pi*0.5)**2
    noise = torch.randn_like(x0_map)
    return torch.sqrt(alpha_bar.view(-1,1,1,1,1))*x0_map + torch.sqrt(1-alpha_bar.view(-1,1,1,1,1))*noise

# ---------------- φ 统计 ----------------
def compute_phi_stats_from_metrics(metrics_dir):
    if metrics_dir is None:
        return None
    import csv
    csv_path = Path(metrics_dir)/"metrics_summary.csv"
    if csv_path.exists():
        phi = []
        with open(csv_path,"r") as f:
            for row in csv.DictReader(f):
                phi.append(float(row["phi"]))
        if len(phi)>0:
            mu = float(np.mean(phi)); std = float(np.std(phi)); std = float(max(std, 0.02, 1e-6))
            return mu, std
    return None

def compute_phi_stats_fallback(dataset, sample_n=64):
    vals=[]
    for i in range(min(len(dataset), sample_n)):
        x0 = dataset[i].float()
        vals.append(float(x0.mean().item()))
    mu = float(np.mean(vals)); std = float(np.std(vals)); std = float(max(std, 0.02, 1e-6))
    return mu, std

# ---------------- 代理指标（fp32 + 防非数） ----------------
def tv_surface_density(prob):
    prob = prob.float()
    dz = prob[:,:,1:,:,:]-prob[:,:,:-1,:,:]
    dy = prob[:,:,:,1:,:]-prob[:,:,:,:-1,:]
    dx = prob[:,:,:,:,1:]-prob[:,:,:,:,:-1]
    tv = (dz.abs().mean(dim=(1,2,3,4)) + dy.abs().mean(dim=(1,2,3,4)) + dx.abs().mean(dim=(1,2,3,4))) / 3.0
    return tv

def s2_directional(prob, lags):
    prob = prob.float()
    B = prob.size(0); vals=[]
    for r in lags:
        r = int(r) if isinstance(r, (int,)) else int(r.item())
        if r<=0: vals.append(torch.zeros(B, device=prob.device)); continue
        vD = (prob[:,:, :-r, :, :] * prob[:,:, r:, :, :]).mean(dim=(1,2,3,4))
        vH = (prob[:,:,:, :-r, :] * prob[:,:,:, r:, :]).mean(dim=(1,2,3,4))
        vW = (prob[:,:,:,:, :-r]   * prob[:,:,:,:, r:]).mean(dim=(1,2,3,4))
        vals.append((vD+vH+vW)/3.0)
    return torch.stack(vals, dim=-1)

def pred_metrics_from_logits(logits, lags):
    prob = torch.sigmoid(logits.float())
    phi  = prob.mean(dim=(1,2,3,4)).clamp(1e-6, 1-1e-6)
    Sv   = tv_surface_density(prob).clamp(min=1e-6)
    k    = (phi**3) / ( (5.0)*(Sv**2)*((1.0-phi)**2) + 1e-12 )
    s2   = s2_directional(prob, lags)
    phi = torch.nan_to_num(phi, nan=0.5); Sv = torch.nan_to_num(Sv, nan=1.0)
    k   = torch.nan_to_num(k, nan=0.0, posinf=1e3, neginf=0.0)
    s2  = torch.nan_to_num(s2, nan=0.0, posinf=1.0, neginf=0.0)
    return phi, Sv, k, s2

# ---------------- 评测与保存（PNG：孔隙=黑） ----------------
@torch.no_grad()
def evaluate_and_save(acc, unet, epoch, ckpt_dir, samples_dir, shape, z_phi_eval, s2_lags, cfg_scale=1.0, timesteps=1000):
    if not acc.is_main_process: return
    if hasattr(unet,'use_checkpoint'): acc.unwrap_model(unet).use_checkpoint = True
    name = f"epoch_{epoch + 1:04d}" if epoch >= 0 else "initial_state"
    unet.eval(); device = acc.device
    xt = torch.randn(shape, device=device)
    cond = torch.tensor([[z_phi_eval]], device=device, dtype=torch.float32).repeat(shape[0],1)
    null = torch.zeros_like(cond)
    for t in tqdm(reversed(range(timesteps)), total=timesteps, desc="DDPM"):
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        lc = unet(xt, tt, cond=cond)
        if cfg_scale!=1.0:
            lu = unet(xt, tt, cond=null); l0 = lu + cfg_scale*(lc - lu)
        else:
            l0 = lc
        x0 = torch.tanh(l0)  # [-1,1]
        tf = torch.tensor([t], device=device, dtype=torch.float)
        abar = torch.cos(((tf / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        abar_prev = torch.cos((((tf - 1) / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2 if t>0 else torch.tensor(1.0, device=device)
        alpha_t = abar / abar_prev; beta_t = 1 - alpha_t
        mean = torch.sqrt(abar_prev) * beta_t/(1-abar) * x0 + torch.sqrt(alpha_t)*(1-abar_prev)/(1-abar) * xt
        if t>0:
            var = ((1 - abar_prev) / (1 - abar)) * beta_t
            xt = mean + torch.sqrt(torch.clamp(var, min=1e-20)) * torch.randn_like(xt)
        else:
            xt = mean
    prob = (xt + 1)/2; binary = (prob>0.5).float()
    out = ((1.0 - binary).squeeze(0).squeeze(0).cpu().numpy()*255).astype(np.uint8)  # 孔隙=黑
    if epoch >= 0:
        torch.save(acc.unwrap_model(unet).state_dict(), ckpt_dir / f"{name}.pth")
    out_dir = samples_dir / name; out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(out.shape[0]): Image.fromarray(out[i],'L').save(out_dir/f"slice_{i:04d}.png")
    if hasattr(unet,'use_checkpoint'): acc.unwrap_model(unet).use_checkpoint = False

# ---------------- 安全加载 ckpt（自适配 cond_mlp） ----------------
def safe_load_pretrained(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    own = model.state_dict()
    adapted, removed = [], []

    # 处理 cond_mlp.0.weight（输入维度变了：如 9 -> 1）
    k = "cond_mlp.0.weight"
    if k in sd and k in own:
        w_old, w_new = sd[k], own[k]
        if w_old.shape != w_new.shape:
            # 如果只是不匹配输入维度，且输出维度一致，做均值收缩
            if w_old.dim()==2 and w_new.dim()==2 and w_old.shape[0]==w_new.shape[0] and w_new.shape[1]==1:
                sd[k] = w_old.mean(dim=1, keepdim=True)
                adapted.append(k)
            else:
                sd.pop(k); removed.append(k)
                b = "cond_mlp.0.bias"
                if b in sd: sd.pop(b); removed.append(b)

    # 其他潜在不匹配（极少见）：直接在 load_state_dict 里通过 strict=False 处理
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded pretrained from {ckpt_path}. Adapted:{adapted} Removed:{removed} Missing:{len(missing)} Unexpected:{len(unexpected)}")
    if len(missing)>0: print("Missing keys (first 10):", missing[:10])
    if len(unexpected)>0: print("Unexpected keys (first 10):", unexpected[:10])

# ---------------- Main ----------------
def main(args):
    acc = Accelerator(mixed_precision=args.mixed_precision)
    out = Path(args.output_dir)
    rec = out/"record"; samp = out/"test"; ckpt = out/"checkpoints"
    for p in [out, rec, samp, ckpt]: p.mkdir(parents=True, exist_ok=True)

    # 数据与 φ 统计
    ds = NpyRockDataset(args.data_dir, sub_volume_depth=args.train_depth, train_resolution=args.train_resolution, pore_value=args.pore_value)
    stats = compute_phi_stats_from_metrics(args.metrics_dir)
    if stats is None:
        stats = compute_phi_stats_fallback(ds)
    mu_phi, sc_phi = stats
    with open(out/"cond_stats.json","w") as f: json.dump({"mu_phi":mu_phi,"sigma_phi":sc_phi}, f, indent=2)

    # 模型（cond_dim=1，仅 φ）
    model = VoxelDiT(input_size=(args.train_depth, args.train_resolution, args.train_resolution),
                     patch_size=(args.patch_size,)*3, depth=args.model_depth, num_heads=args.num_heads, cond_dim=1)

    # 安全加载 ckpt（自适配 cond_mlp）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        safe_load_pretrained(model, args.pretrained_path)

    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate)

    dice = DiceLoss(); fft = FFTLoss(); grad = GradientLoss(); style = StyleLoss()
    model, opt, dl = acc.prepare(model, opt, dl)
    mu_phi_t = torch.tensor(mu_phi, dtype=torch.float32, device=acc.device)
    sc_phi_t = torch.tensor(sc_phi, dtype=torch.float32, device=acc.device)
    s2l = torch.tensor(args.s2_lags, dtype=torch.int64, device=acc.device)

    timesteps = 1000
    for epoch in range(args.num_train_epochs):
        model.train(); acc.unwrap_model(model).use_checkpoint = True
        pbar = tqdm(total=len(dl), disable=not acc.is_local_main_process, desc=f"Epoch {epoch+1}")
        for step, x0 in enumerate(dl):
            # 条件: 每个样本自己的 φ 作为目标
            phi_target = x0.mean(dim=(1,2,3,4)).to(acc.device)
            z_phi_target = (phi_target - mu_phi_t) / sc_phi_t

            # 前向加噪
            t  = torch.randint(0, timesteps, (x0.shape[0],), device=acc.device).long()
            xt = bit_diffusion_forward(x0.to(acc.device), t, timesteps)

            # classifier-free dropout
            if args.cfg_dropout_prob>0.0 and np.random.rand() < args.cfg_dropout_prob:
                cond_batch = torch.zeros((x0.size(0), 1), device=acc.device)     # z=0
            else:
                cond_batch = z_phi_target.unsqueeze(1)                           # shape (B,1)

            logits = model(xt, t, cond=cond_batch)

            # 重建
            loss = F.binary_cross_entropy_with_logits(logits, x0.to(acc.device)) + dice(logits, x0.to(acc.device))
            if args.grad_loss_weight>0: loss += args.grad_loss_weight * grad(logits, x0.to(acc.device))
            if args.fft_loss_weight>0:  loss += args.fft_loss_weight  * fft(logits, x0.to(acc.device))
            if args.style_loss_weight>0:loss += args.style_loss_weight* style(logits, x0.to(acc.device))

            # 条件损失（把预测 φ 拉向目标 φ）
            phi_pred, _, _, _ = pred_metrics_from_logits(logits, s2l)
            z_phi_pred = (phi_pred - mu_phi_t) / sc_phi_t
            z_phi_pred = torch.nan_to_num(z_phi_pred, nan=0.0, posinf=1e6, neginf=-1e6)
            cond_loss = ((z_phi_pred - z_phi_target)**2).mean()

            # warm-up 控制条件项生效
            cond_coeff = 0.0 if epoch < args.cond_warmup_epochs else args.cond_loss_weight
            loss = loss + cond_coeff * cond_loss

            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

            pbar.update(1)
            if acc.is_local_main_process:
                pbar.set_postfix(loss=float(loss.item()), cond=float(cond_loss.item()))
        pbar.close()

        # 评测/保存：使用指定 φ*（若未设，则用均值 → z=0）
        if (epoch + 1) % args.eval_every == 0 or epoch == args.num_train_epochs - 1:
            phi_eval = args.phi_target_eval if args.phi_target_eval is not None else mu_phi
            z_eval = float((phi_eval - mu_phi)/sc_phi)
            evaluate_and_save(acc, model, epoch,
                              ckpt_dir=ckpt, samples_dir=samp,
                              shape=(1,1,args.eval_depth, args.eval_resolution, args.eval_resolution),
                              z_phi_eval=z_eval, s2_lags=s2l,
                              cfg_scale=args.cfg_scale, timesteps=timesteps)
    if acc.is_main_process: print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D voxel DiT finetune with target-phi conditioning (rev6b, safe ckpt load)")
    # 路径
    parser.add_argument("--data_dir", type=str, default="./dataset/NPY/Bentheimer")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--metrics_dir", type=str, default=None)
    # 模型
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--model_depth", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=12)
    # 训练
    parser.add_argument("--num_train_epochs", type=int, default=80)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no","fp16","bf16"])
    parser.add_argument("--fft_loss_weight", type=float, default=0.0)
    parser.add_argument("--grad_loss_weight", type=float, default=0.008)
    parser.add_argument("--style_loss_weight", type=float, default=0.0)
    parser.add_argument("--cond_loss_weight", type=float, default=1.5)
    parser.add_argument("--cond_warmup_epochs", type=int, default=3)
    # 采样/尺寸
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--train_depth", type=int, default=256)
    parser.add_argument("--eval_depth", type=int, default=256)
    parser.add_argument("--train_resolution", type=int, default=256)
    parser.add_argument("--eval_resolution", type=int, default=256)
    # 条件与 CFG
    parser.add_argument("--s2_lags", type=int, nargs="+", default=[8,16,32,64,96,128])
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.15)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--pore_value", type=int, default=0, choices=[0,1])
    # 评测时的目标 φ*
    parser.add_argument("--phi_target_eval", type=float, default=None, help="if set, use this porosity at eval; else use dataset mean")
    args = parser.parse_args(); main(args)
