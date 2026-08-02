#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import torch
import warnings
from scipy.ndimage import label
from tqdm import tqdm
from PIL import Image

# ================= 全局物理配置 =================
# 严格保持原文件参数
LATTICE_VISCOSITY = 0.166667
BODY_FORCE = 1e-5
MAX_ITER = 4000


# ================= 核心算法 =================

def get_largest_connected_component(vol_bool):
    if vol_bool.sum() == 0: return vol_bool
    labeled, num = label(vol_bool, structure=np.ones((3, 3, 3)))
    if num == 0: return vol_bool
    counts = np.bincount(labeled.ravel())
    if len(counts) <= 1: return vol_bool
    largest_idx = np.argmax(counts[1:]) + 1
    return (labeled == largest_idx)


class LBMSolver:

    def __init__(self, vol_bool, device):
        self.device = device
        self.vol_clean = get_largest_connected_component(vol_bool)
        self.phi_effective = self.vol_clean.sum() / self.vol_clean.size
        self.mask_pore = torch.from_numpy(self.vol_clean).to(self.device, dtype=torch.bool)
        self.mask_solid = ~self.mask_pore
        self.nz, self.nx, self.ny = self.mask_pore.shape

        self.c = torch.tensor([
            [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
        ], device=self.device, dtype=torch.int64)

        w_np = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float32)
        self.w = torch.from_numpy(w_np).to(self.device)
        self.inv_idx = torch.tensor([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15],
                                    device=self.device, dtype=torch.long)

        self.f = torch.zeros((19, self.nz, self.nx, self.ny), device=self.device, dtype=torch.float32)
        for i in range(19): self.f[i] = self.w[i]
        self.tau = 3.0 * LATTICE_VISCOSITY + 0.5
        self.omega = 1.0 / self.tau

    def step(self):
        rho = torch.sum(self.f, dim=0)
        ux = torch.zeros_like(rho);
        uy = torch.zeros_like(rho);
        uz = torch.zeros_like(rho)
        for i in range(19):
            val = self.f[i]
            cx, cy, cz = self.c[i].tolist()
            if cx != 0: ux += val * cx
            if cy != 0: uy += val * cy
            if cz != 0: uz += val * cz
        inv_rho = 1.0 / (rho + 1e-10)
        ux *= inv_rho;
        uy *= inv_rho;
        uz *= inv_rho

        # 施加外力 (Z轴方向)
        uz += self.tau * BODY_FORCE * inv_rho

        usq = ux ** 2 + uy ** 2 + uz ** 2
        f_next = torch.empty_like(self.f)
        for i in range(19):
            cx, cy, cz = self.c[i].tolist()
            cu = cx * ux + cy * uy + cz * uz
            feq = self.w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq)
            f_next[i] = (1.0 - self.omega) * self.f[i] + self.omega * feq
        for i in range(19):
            s = self.c[i].tolist()
            if s != [0, 0, 0]: f_next[i] = torch.roll(f_next[i], shifts=s, dims=(0, 1, 2))
        f_solid = f_next[:, self.mask_solid]
        f_next[:, self.mask_solid] = f_solid[self.inv_idx, :]
        self.f = f_next
        return torch.abs(uz[self.mask_pore]).mean().item()

    def run(self, steps):
        for _ in range(100): self.step()
        u_sum = 0.0
        sample_count = 0
        try:
            for i in range(steps):
                u = self.step()
                if np.isnan(u): return 0.0, self.phi_effective
                if i > steps - 200:
                    u_sum += u
                    sample_count += 1
        except Exception:
            return 0.0, self.phi_effective
        return u_sum / max(1, sample_count), self.phi_effective


# ================= 数据读取 =================

def load_data_smart(path):
    """
    Strictly follows load_npy_smart logic from perm_suite_final.py
    Also handles image folders for generated samples.
    """
    try:
        if os.path.isdir(path):
            # 文件夹模式 (PNG序列)
            files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.tif")))
            if not files: return None
            img0 = np.array(Image.open(files[0]))
            vol = np.zeros((len(files), img0.shape[0], img0.shape[1]), dtype=bool)
            for i, f in enumerate(files):
                img = np.array(Image.open(f))
                if img.ndim == 3: img = img[..., 0]
                vol[i] = (img > 127)
            # 极性判断
            if vol.mean() > 0.6: vol = ~vol
            return vol
        else:
            # NPY 模式
            arr = np.load(path)
            if arr.ndim == 4: arr = arr[0]

            # 归一化与类型处理
            if arr.dtype in [np.float32, np.float64]:
                if arr.max() > 1.0: arr = arr / arr.max()
                arr = (arr > 0.5)
            else:
                if arr.max() > 1:
                    arr = (arr > 127)
                else:
                    arr = (arr > 0)

            # 极性自动判断
            phi_raw = arr.sum() / arr.size
            if phi_raw > 0.6:
                vol = ~arr
            else:
                vol = arr
            return vol.astype(bool)
    except Exception as e:
        print(f"[Warn] Failed to load {path}: {e}")
        return None


# ================= 主程序 =================

def main(args):
    print(f"Permeability Evaluation (LBM)")
    print(f"Sample Dir: {args.sample_dir}")
    print(f"Resolution: {args.resolution} um")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 兼容文件夹模式和NPY模式
    if os.path.isdir(args.sample_dir):
        # 检查是否包含子文件夹（针对图像序列）
        subdirs = [os.path.join(args.sample_dir, d) for d in os.listdir(args.sample_dir)
                   if os.path.isdir(os.path.join(args.sample_dir, d))]
        # 检查是否包含 npy 文件
        npys = sorted(glob.glob(os.path.join(args.sample_dir, "*.npy")))
        samples = sorted(subdirs + npys)
    else:
        samples = [args.sample_dir]

    results = []
    print(f"Processing {len(samples)} samples...")

    for p in tqdm(samples):
        name = os.path.basename(p)
        if name.startswith("stats") or name.endswith(".txt"): continue

        vol = load_data_smart(p)
        if vol is None: continue

        try:
            solver = LBMSolver(vol, device.index if device.type == 'cuda' else 'cpu')
            gpu_id = args.gpu if torch.cuda.is_available() else 0

            # 重新实例化以匹配ID传入
            class LBMSolver_Adapted(LBMSolver):
                def __init__(self, vol_bool, dev_id):
                    super().__init__(vol_bool, dev_id)  # 调用父类逻辑

            solver_instance = LBMSolver_Adapted(vol, gpu_id)
            u_pore, phi_eff = solver_instance.run(MAX_ITER)

            # 计算逻辑严格遵循 perm_suite_final.py
            if u_pore > 0:
                k_lb = (u_pore * phi_eff) * LATTICE_VISCOSITY / BODY_FORCE
                k_phys = k_lb * (args.resolution ** 2)
                k_darcy = k_phys / 0.986923
            else:
                k_darcy = 0.0

            results.append(f"{name},{phi_eff:.6f},{k_darcy:.6f}")
            del solver_instance
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error on {name}: {e}")
            results.append(f"{name},0.0,0.0")

    # 保存结果
    out_path = os.path.join(args.output_dir, "metric_permeability.txt")
    with open(out_path, "w") as f:
        f.write("Sample,Phi,Perm_Darcy\n")
        f.write("\n".join(results))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str,
                        default="./samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--output_dir", type=str, default="./output/metrics")
    parser.add_argument("--gpu", type=int, default=0)
    # 默认分辨率 Bentheimer = 2.25
    parser.add_argument("--resolution", type=float, default=2.25)

    args = parser.parse_args()
    main(args)