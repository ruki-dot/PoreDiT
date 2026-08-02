#!/usr/bin/env python3
# Evaluation Script: Euler Characteristic Density (M2)
# Source Logic: plot_minkowski_boxplots.py
# Unit: voxel^-3 (Density)

import os
import argparse
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from skimage import measure, morphology


def load_volume(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")))
        vol = np.stack([np.array(Image.open(f).convert('L')) for f in files], axis=0)
        return (vol < 128)
    else:
        vol = np.load(path)
        return vol.astype(bool)


def main(args):
    print("Evaluating Euler Characteristic Density (with cleaning)...")
    os.makedirs(args.output_dir, exist_ok=True)

    samples = sorted(glob.glob(os.path.join(args.sample_dir, "*")))
    results = []

    for p in tqdm(samples):
        name = os.path.basename(p)
        if "target" in name or "stats" in name or name.endswith(".txt"): continue
        if not os.path.isdir(p) and not p.endswith(".npy"): continue

        try:
            vol = load_volume(p)
            if vol.ndim != 3: continue

            # --- 核心逻辑: Cleaning ---
            vol_clean = morphology.remove_small_objects(vol, min_size=34, connectivity=3)
            # ---------------------------

            # --- 核心逻辑: Euler Density ---
            chi = measure.euler_number(vol_clean, connectivity=3)

            # Normalize by volume size -> Chi_v [voxel^-3]
            chi_density = chi / vol_clean.size

            results.append(f"{name}: {chi_density:.4e}")
        except Exception as e:
            print(f"Error {name}: {e}")

    out_path = os.path.join(args.output_dir, "metric_euler.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(results))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="./samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--output_dir", type=str, default="./output/metrics")
    args = parser.parse_args()
    main(args)