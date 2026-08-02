#!/usr/bin/env python3
# Evaluation Script: Porosity (M0)
# Source Logic: plot_minkowski_boxplots.py
# Preprocessing: Remove small objects (min_size=34)

import os
import argparse
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from skimage import morphology


def load_volume(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")))
        vol = np.stack([np.array(Image.open(f).convert('L')) for f in files], axis=0)
        return (vol < 128)  # Black=Pore, so <128 is True
    else:
        vol = np.load(path)
        return vol.astype(bool)


def main(args):
    print("Evaluating Porosity (with cleaning)...")
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

            # 论文逻辑：先移除小于 34 体素的孤立孔隙，再算孔隙度
            vol_clean = morphology.remove_small_objects(vol, min_size=34, connectivity=3)

            phi = vol_clean.sum() / vol_clean.size
            results.append(f"{name}: {phi:.6f}")
        except Exception as e:
            print(f"Error {name}: {e}")

    out_path = os.path.join(args.output_dir, "metric_porosity.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(results))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="./samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--output_dir", type=str, default="./output/metrics")
    args = parser.parse_args()
    main(args)