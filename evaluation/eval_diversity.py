#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_volume_smart(path):
    try:
        # 1. 文件夹模式 (PNG序列) - 对应 load_volume_from_pngs
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.png")))
            if not files: return None
            slices = []
            for f in files:
                img = Image.open(f).convert('L')
                arr = np.array(img)
                slices.append(arr < 128)
            return np.stack(slices, axis=0).astype(bool)

        # 2. NPY 模式 - 对应 load_volume_from_npy
        else:
            arr = np.load(path)
            if arr.ndim == 4: arr = arr[0]
            threshold = 0.5 * np.max(arr)
            return (arr < threshold).astype(bool)

    except Exception as e:
        print(f"[Warn] Load failed {path}: {e}")
        return None


def compute_hamming_distance(vol1, vol2):
    # 尺寸对齐 (以防万一)
    d = min(vol1.shape[0], vol2.shape[0])
    h = min(vol1.shape[1], vol2.shape[1])
    w = min(vol1.shape[2], vol2.shape[2])

    v1 = vol1[:d, :h, :w]
    v2 = vol2[:d, :h, :w]

    # 异或运算：相同为0，不同为1
    diff = np.bitwise_xor(v1, v2)
    return np.mean(diff)


# ================= 主程序 =================

def main(args):
    print(f"Diversity Evaluation (Hamming Distance / D_NN)")
    print(f"Gen Dir: {args.sample_dir}")
    print(f"Ref Dir: {args.ref_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取真实样本列表
    if os.path.isdir(args.ref_dir):
        # 优先找npy，没有则找文件夹
        ref_files = sorted(glob.glob(os.path.join(args.ref_dir, "*.npy")))
        if not ref_files:
            ref_files = sorted([os.path.join(args.ref_dir, d) for d in os.listdir(args.ref_dir)
                                if os.path.isdir(os.path.join(args.ref_dir, d))])
    else:
        print(f"Error: Ref dir {args.ref_dir} not found.")
        return

    gen_samples = sorted(glob.glob(os.path.join(args.sample_dir, "*")))
    gen_samples = [p for p in gen_samples if os.path.isdir(p) or p.endswith(".npy")]
    gen_samples = [p for p in gen_samples if "target" not in os.path.basename(p) and "stats" not in os.path.basename(p)]

    print(f"Found {len(ref_files)} Real samples and {len(gen_samples)} Generated samples.")
    if len(ref_files) == 0: return

    print("Pre-loading reference set...")
    real_vols = []
    real_names = []
    for f in tqdm(ref_files):
        v = load_volume_smart(f)
        if v is not None:
            real_vols.append(v)
            real_names.append(os.path.basename(f))

    results = []
    min_distances = []

    print("Calculating Pairwise Distances...")
    for gen_path in tqdm(gen_samples):
        gen_vol = load_volume_smart(gen_path)
        if gen_vol is None: continue

        current_min_dist = 1.0
        nearest_real_name = ""

        for i, real_vol in enumerate(real_vols):
            dist = compute_hamming_distance(gen_vol, real_vol)
            if dist < current_min_dist:
                current_min_dist = dist
                nearest_real_name = real_names[i]

        min_distances.append(current_min_dist)
        gen_name = os.path.basename(gen_path)
        results.append(f"{gen_name},{current_min_dist:.6f},{nearest_real_name}")

    # save
    out_txt = os.path.join(args.output_dir, "metric_dnn_hamming.txt")
    with open(out_txt, "w") as f:
        f.write("Sample,Min_Hamming_Dist,Nearest_Real\n")
        f.write("\n".join(results))

    # Summary
    min_distances = np.array(min_distances)
    with open(os.path.join(args.output_dir, "metric_dnn_summary.txt"), "w") as f:
        f.write(f"Mean_Min_Dist: {min_distances.mean():.6f}\n")
        f.write(f"Abs_Min_Dist: {min_distances.min():.6f}\n")

    print(f"Done. Mean Dist: {min_distances.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str,
                        default="./samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--ref_dir", type=str,
                        default="./dataset/NPY/Bentheimer")
    parser.add_argument("--output_dir", type=str, default="./output/metrics")

    args = parser.parse_args()
    main(args)