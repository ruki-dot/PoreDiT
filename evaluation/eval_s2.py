#!/usr/bin/env python3
# Evaluation Script: Two-Point Correlation Function (S2)
# Method: FFT-based Autocorrelation (Raw, Unnormalized)
# Matches original logic: S2(0) == Porosity

import os
import argparse
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm


def load_volume(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")))
        if not files: return None
        vol = np.stack([np.array(Image.open(f).convert('L')) for f in files], axis=0)
        return (vol < 128).astype(np.uint8)  # 1=Pore, 0=Solid
    else:
        # NPY loading
        vol = np.load(path)
        # Handle boolean or binary
        if vol.dtype == bool:
            vol = vol.astype(np.uint8)
        else:
            # Provide a fallback threshold if it's float, though your samples are likely binary
            if vol.max() > 1:
                vol = (vol < 128).astype(np.uint8)
            else:
                vol = (vol > 0.5).astype(np.uint8)
        return vol


def calculate_s2_fft(vol):
    """
    Compute S2 using FFT autocorrelation without normalization.
    S2(r) = < I(x) * I(x+r) >
    """
    # Ensure binary 0/1
    phase = vol.astype(np.float32)

    # FFT
    K = np.fft.rfftn(phase)
    S = K * np.conj(K)
    corr = np.fft.irfftn(S, s=phase.shape)

    # Normalize only by volume size to get probability, NOT by variance/phi
    corr = corr.real / phase.size

    # Radial average (simplified for cubic grids)
    # We only take the first N lags to match typical plotting range
    # Taking a slice along one axis is a fast approximation for isotropic media
    # Or we can do full radial average. Here we do axis-aligned average to keep it robust and fast.
    # Matches the logic of "directionally averaged" if we average 3 axes.

    N = min(vol.shape) // 2
    s2_x = corr[0, 0, :N]
    s2_y = corr[0, :N, 0]
    s2_z = corr[:N, 0, 0]

    s2_curve = (s2_x + s2_y + s2_z) / 3.0
    return s2_curve


def main(args):
    print("Evaluating S2 Correlation (Raw, Unnormalized)...")
    os.makedirs(args.output_dir, exist_ok=True)

    samples = sorted(glob.glob(os.path.join(args.sample_dir, "*")))
    results = []

    for p in tqdm(samples):
        name = os.path.basename(p)
        if "target" in name or "stats" in name or name.endswith(".txt"):
            continue

        if not os.path.isdir(p) and not p.endswith(".npy"): continue

        try:
            vol = load_volume(p)
            # 二次检查维度，防止读入 1D array 报错
            if vol.ndim != 3:
                continue

            # Calculate Raw S2
            s2_curve = calculate_s2_fft(vol)

            # Record data as comma-separated string
            s2_values = ",".join([f"{x:.6f}" for x in s2_curve])
            results.append(f"{name} | S2: {s2_values}")

        except Exception as e:
            print(f"Error processing {name}: {e}")

    out_path = os.path.join(args.output_dir, "metric_s2.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(results))
    print(f"Saved S2 data to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str,
                        default="./samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--output_dir", type=str, default="./output/metrics")
    args = parser.parse_args()
    main(args)