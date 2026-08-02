#!/usr/bin/env python3
# prepare_dataset.py
# Raw Binary (.raw) -> NPY Dataset Slicing Script
# Works for Bentheimer (256^3) and other porous media.

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def process_raw(args):
    # 1. Setup Paths
    raw_path = Path(args.raw_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Start] Processing Raw File: {raw_path}")
    print(f"        Output Directory:    {out_dir}")
    print(f"        Volume Dimensions:   {args.dims}")
    print(f"        Sample Size:         {args.sample_dim}^3")
    print(f"        Stride:              {args.stride}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # 2. Load Raw Data
    # Assuming uint8 binary (0/1 or 0/255) or similar.
    # Adjust dtype if your raw data is different (e.g., uint16).
    try:
        # Load as flat array
        full_volume = np.fromfile(raw_path, dtype=np.uint8)
        # Reshape to 3D
        D, H, W = args.dims
        if full_volume.size != D * H * W:
            raise ValueError(f"File size {full_volume.size} does not match dimensions {D}x{H}x{W} = {D * H * W}")
        full_volume = full_volume.reshape(D, H, W)
        print("[Data] Raw file loaded and reshaped successfully.")
    except Exception as e:
        print(f"[Error] Failed to load/reshape raw file: {e}")
        return

    # 3. Binarize (Optional but recommended)
    # Assuming pore=0, solid=1 in original, or similar.
    # Here we ensure it's 0 and 1.
    unique_vals = np.unique(full_volume)
    if len(unique_vals) > 2:
        print(f"[Warn] Volume is not binary! Unique values: {unique_vals}. Thresholding at 127.")
        full_volume = (full_volume > 127).astype(np.uint8)
    else:
        # Simplify to 0/1 if it's 0/255
        full_volume = (full_volume > 0).astype(np.uint8)

    # 4. Sliding Window Slicing
    dim = args.sample_dim
    stride = args.stride
    count = 0

    z_coords = range(0, D - dim + 1, stride)
    y_coords = range(0, H - dim + 1, stride)
    x_coords = range(0, W - dim + 1, stride)

    total = len(z_coords) * len(y_coords) * len(x_coords)

    with tqdm(total=total, desc="Slicing") as pbar:
        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    # Extract
                    sample = full_volume[z:z + dim, y:y + dim, x:x + dim]

                    # Save (Original)
                    np.save(out_dir / f"subvol_{count:04d}_orig.npy", sample)
                    count += 1

                    # Augmentation (Flip/Rotate)
                    if args.augment:
                        # Flip axis 0
                        np.save(out_dir / f"subvol_{count:04d}_aug_flip0.npy", np.flip(sample, 0))
                        count += 1
                        # Rotate 90 deg axes (1,2)
                        np.save(out_dir / f"subvol_{count:04d}_aug_rot90.npy", np.rot90(sample, k=1, axes=(1, 2)))
                        count += 1

                    pbar.update(1)

    print(f"[Done] Generated {count} samples in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Raw Rock Data to NPY")

    # Paths
    parser.add_argument("--raw_file", type=str, required=True,
                        help="Path to .raw file (e.g., ./dataset/Raw/Bentheimer.raw)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output NPY directory (e.g., ./dataset/NPY/Bentheimer)")

    # Dimensions (User must specify the size of the RAW file)
    parser.add_argument("--dims", type=int, nargs=3, required=True,
                        help="Dimensions of the raw file: Depth Height Width (e.g., 1000 1000 1000)")

    # Slicing Config
    parser.add_argument("--sample_dim", type=int, default=256, help="Size of cube to cut (default: 256)")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride (default: 128)")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation (flips/rotations)")

    args = parser.parse_args()
    process_raw(args)