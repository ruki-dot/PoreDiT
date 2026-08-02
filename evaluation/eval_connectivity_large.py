#!/usr/bin/env python3
# Evaluation Script: Connectivity for Large Volumes (1024^3)
# Calculates the size of the largest connected pore cluster.

import os
import argparse
import numpy as np
import glob
from PIL import Image
from scipy.ndimage import label
import time


def main(args):
    print(f"Evaluating Large Scale Connectivity (Target: {args.target_size})...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Logic adapted for large tiled images (as per your original script)
    # Assuming sample_dir contains ONE huge sample as a sequence of images
    files = sorted(glob.glob(os.path.join(args.sample_dir, "*.png")))
    if not files:
        print("No images found.")
        return

    print(f"Loading {len(files)} slices...")
    # Loading logic (can be memory optimized if needed, keeping simple here)
    vol = np.stack([np.array(Image.open(f).convert('L')) for f in files], axis=0)
    vol = (vol < 128).astype(int)  # 1=Pore

    print("Labeling connected components...")
    # 26-connectivity (3x3x3 ones)
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, num_features = label(vol, structure=structure)

    print("Calculating cluster sizes...")
    sizes = np.bincount(labeled.ravel())
    # sizes[0] is background (matrix), so skip it
    pore_sizes = sizes[1:]

    max_pore = pore_sizes.max() if len(pore_sizes) > 0 else 0
    total_pore = np.sum(vol)

    ratio = max_pore / total_pore if total_pore > 0 else 0

    res_str = f"Largest Connected Cluster Ratio: {ratio:.6f}\nTotal Pore Voxels: {total_pore}\nLargest Cluster Voxels: {max_pore}"
    print(res_str)

    with open(os.path.join(args.output_dir, "metric_connectivity_large.txt"), "w") as f:
        f.write(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to your 1024 sample path
    parser.add_argument("--sample_dir", type=str,
                        default="./samples/Bentheimer/resolution_1024/phi_cond_1024_calibrated_final")
    parser.add_argument("--output_dir", type=str, default="./output/metrics")
    parser.add_argument("--target_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)