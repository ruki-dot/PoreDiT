#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_large_png_porosity.py

本地检查 512^3 / 1024^3 PNG 大样本孔隙度，帮助核对 Table 1。

会输出：
  Phi_black_pore_raw: 按论文图像约定，黑色 < 128 为孔隙
  Phi_auto_raw:       按原 LBM 脚本，先 img > 127，再自动极性判断
  Phi_LCC:            最大连通孔隙团对应的有效孔隙度
  LargestClusterFraction: 最大连通孔隙团 / 总孔隙体素

512 示例：
python compute_large_png_porosity.py ^
  --sample_dir "C:\\Users\\黄意卓\\Desktop\\rock\\code\\samples\\Bentheimer\\resolution_512\\phi_cond_512_calibrated" ^
  --output_dir "C:\\Users\\黄意卓\\Desktop\\rock\\code\\revision_experiments\\porosity_check" ^
  --name Gen_512

1024 示例：
python compute_large_png_porosity.py ^
  --sample_dir "C:\\Users\\黄意卓\\Desktop\\rock\\code\\samples\\Bentheimer\\resolution_1024\\phi_cond_1024_calibrated_final" ^
  --output_dir "C:\\Users\\黄意卓\\Desktop\\rock\\code\\revision_experiments\\porosity_check" ^
  --name Gen_1024
"""

import os
import csv
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def read_png_gray(path):
    return np.array(Image.open(path).convert("L"))


def load_png_stack(sample_dir):
    files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {sample_dir}")
    first = read_png_gray(files[0])
    vol = np.zeros((len(files), first.shape[0], first.shape[1]), dtype=np.uint8)
    for i, f in enumerate(files):
        vol[i] = read_png_gray(f)
    return vol


def lcc_stats(vol_bool):
    if vol_bool.sum() == 0:
        return 0, 0, 0.0
    labeled, num = label(vol_bool, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return int(vol_bool.sum()), 0, 0.0
    sizes = np.bincount(labeled.ravel())
    pore_sizes = sizes[1:]
    total_pore = int(vol_bool.sum())
    largest = int(pore_sizes.max()) if len(pore_sizes) else 0
    frac = largest / total_pore if total_pore > 0 else 0.0
    return total_pore, largest, frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default="sample")
    parser.add_argument("--skip_lcc", action="store_true", help="1024^3 内存不够时可先跳过 LCC")
    parser.add_argument("--clean_min_size", type=int, default=0, help="可选：remove_small_objects 的 min_size，0 表示不启用")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] {args.sample_dir}")
    vol_gray = load_png_stack(args.sample_dir)
    total_voxels = int(vol_gray.size)

    # 论文中图像一般是黑色孔隙，白色基质
    vol_black = vol_gray < 128
    phi_black = float(vol_black.mean())

    # 原 LBM 脚本的自动极性逻辑
    vol_auto = vol_gray > 127
    if vol_auto.mean() > 0.6:
        vol_auto = ~vol_auto
    phi_auto = float(vol_auto.mean())

    phi_clean = ""
    if args.clean_min_size > 0:
        try:
            from skimage import morphology
            vol_clean = morphology.remove_small_objects(vol_black, min_size=args.clean_min_size, connectivity=3)
            phi_clean = float(vol_clean.mean())
        except Exception as e:
            phi_clean = f"ERROR: {e}"

    total_pore, largest_pore, lcc_frac = "", "", ""
    phi_lcc = ""
    if not args.skip_lcc:
        print("[LCC] labeling connected components; 1024^3 may need large memory.")
        total_pore, largest_pore, lcc_frac = lcc_stats(vol_auto)
        phi_lcc = largest_pore / total_voxels if total_voxels else 0.0

    row = {
        "Name": args.name,
        "SampleDir": args.sample_dir,
        "ShapeZ": vol_gray.shape[0],
        "ShapeY": vol_gray.shape[1],
        "ShapeX": vol_gray.shape[2],
        "TotalVoxels": total_voxels,
        "Phi_black_pore_raw": phi_black,
        "Phi_auto_raw": phi_auto,
        "Phi_clean_min_size": phi_clean,
        "Phi_LCC": phi_lcc,
        "TotalPore_auto": total_pore,
        "LargestClusterPore": largest_pore,
        "LargestClusterFraction": lcc_frac,
    }

    out_csv = out_dir / f"porosity_{args.name}.csv"
    out_txt = out_dir / f"porosity_{args.name}.txt"

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    with out_txt.open("w", encoding="utf-8") as f:
        for k, v in row.items():
            f.write(f"{k}: {v}\n")

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_txt}")
    print(f"Phi_black_pore_raw = {phi_black:.8f}")
    print(f"Phi_auto_raw       = {phi_auto:.8f}")
    if phi_lcc != "":
        print(f"Phi_LCC            = {phi_lcc:.8f}")
        print(f"LCC fraction       = {lcc_frac:.8f}")


if __name__ == "__main__":
    main()
