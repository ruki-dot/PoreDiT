#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
revision_metrics_s2_minkowski_dmin.py

Corrected version.

Fixes:
1. Robust pore/matrix polarity handling for NPY and PNG.
2. If the loaded pore fraction is > 0.6, the phase is automatically inverted.
3. S2 and Minkowski results should now match the manuscript-level porosity scale.
4. D_min baseline is preserved and made polarity-consistent.

Run:
python revision_metrics_s2_minkowski_dmin.py

Optional:
python revision_metrics_s2_minkowski_dmin.py --skip_dmin
python revision_metrics_s2_minkowski_dmin.py --skip_s2 --skip_dmin
"""

import os
import csv
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from scipy import stats
from skimage import measure, morphology
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_gt_npy(gt_dir, limit=None):
    files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    if limit is not None:
        files = files[:limit]
    return files


def list_gen_png_stacks(gen_dir, limit=None):
    subdirs = sorted([
        os.path.join(gen_dir, d)
        for d in os.listdir(gen_dir)
        if os.path.isdir(os.path.join(gen_dir, d))
    ])
    subdirs = [
        p for p in subdirs
        if "target" not in os.path.basename(p).lower()
        and "stats" not in os.path.basename(p).lower()
    ]
    if limit is not None:
        subdirs = subdirs[:limit]
    return subdirs


def auto_fix_pore_polarity(vol_bool, name=""):
    """
    Input:
        vol_bool: bool array, preliminary True=pore.
    Output:
        corrected bool array, True=pore.

    For Bentheimer, pore fraction should normally be around 0.25--0.30.
    If preliminary True fraction > 0.6, it is almost certainly solid=True,
    so invert it.
    """
    vol_bool = np.asarray(vol_bool, dtype=bool)
    frac = float(vol_bool.mean())

    if frac > 0.6:
        vol_bool = ~vol_bool
        fixed_frac = float(vol_bool.mean())
        return vol_bool, frac, fixed_frac, True

    return vol_bool, frac, frac, False


def load_volume_as_pore(path, mode="morphology"):
    """
    Unified loading function. Returns:
        vol_bool, info_dict

    Convention after return:
        True = pore
        False = solid

    PNG stack:
        black pixels are pore, img < 128.

    NPY:
        Try to infer binary convention robustly.
        If values are 0/1, first use arr > 0.5 as preliminary pore,
        then auto-invert if pore fraction > 0.6.
    """
    info = {
        "Path": path,
        "Name": os.path.basename(path),
        "SourceType": "dir_png" if os.path.isdir(path) else "npy",
        "RawTrueFraction": np.nan,
        "FinalPorosity": np.nan,
        "Inverted": False,
        "Shape": "",
    }

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")))
        if not files:
            raise FileNotFoundError(f"No PNG slices found in {path}")

        vol_gray = np.stack(
            [np.array(Image.open(f).convert("L")) for f in files],
            axis=0
        )

        # In generated PNGs, black is pore.
        preliminary = (vol_gray < 128)

    else:
        arr = np.load(path)
        if arr.ndim == 4:
            arr = arr[0]

        if arr.dtype == bool:
            preliminary = arr.astype(bool)
        else:
            arr = np.asarray(arr)

            # Common binary cases.
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                preliminary = (arr > 0.5)
            else:
                # If stored as grayscale-like array, dark phase as pore.
                # Auto inversion below still protects us.
                preliminary = (arr < 128)

    vol, raw_frac, final_frac, inverted = auto_fix_pore_polarity(preliminary, name=path)

    info["RawTrueFraction"] = raw_frac
    info["FinalPorosity"] = final_frac
    info["Inverted"] = inverted
    info["Shape"] = "x".join(map(str, vol.shape))

    return vol.astype(bool), info


# ============================================================
# Statistics helpers
# ============================================================

def bootstrap_ci_mean(x, n_boot=10000, alpha=0.05, seed=42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = x[idx].mean()

    return (
        float(np.percentile(vals, 100 * alpha / 2)),
        float(np.percentile(vals, 100 * (1 - alpha / 2)))
    )


def bootstrap_ci_diff(a, b, n_boot=10000, alpha=0.05, seed=123):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        vals[i] = aa.mean() - bb.mean()

    return (
        float(np.percentile(vals, 100 * alpha / 2)),
        float(np.percentile(vals, 100 * (1 - alpha / 2)))
    )


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * s1 + (len(b) - 1) * s2) / (len(a) + len(b) - 2)

    if pooled <= 0:
        return np.nan

    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def hedges_g(a, b):
    d = cohen_d(a, b)
    if not np.isfinite(d):
        return np.nan

    df = len(a) + len(b) - 2
    if df <= 1:
        return d

    correction = 1 - 3 / (4 * df - 1)
    return float(d * correction)


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    greater = 0
    less = 0

    for x in a:
        greater += np.sum(x > b)
        less += np.sum(x < b)

    return float((greater - less) / (len(a) * len(b)))


def describe_array(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    ci_low, ci_high = bootstrap_ci_mean(x)

    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def pairwise_stats(a, b, name_a, name_b, metric):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    ks = stats.ks_2samp(a, b)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff_low, diff_high = bootstrap_ci_diff(a, b)

    return {
        "dataset_a": name_a,
        "dataset_b": name_b,
        "metric": metric,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "sd_a": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "sd_b": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        "mean_diff_a_minus_b": float(np.mean(a) - np.mean(b)),
        "mean_diff_ci95_low": diff_low,
        "mean_diff_ci95_high": diff_high,
        "ks_D": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mwu_U": float(mw.statistic),
        "mwu_p": float(mw.pvalue),
        "cohen_d": cohen_d(a, b),
        "hedges_g": hedges_g(a, b),
        "cliffs_delta": cliffs_delta(a, b),
    }


def save_descriptive_and_tests(metric_dict, out_dir, prefix):
    ensure_dir(out_dir)

    desc_rows = []
    test_rows = []

    datasets = list(metric_dict.keys())
    metrics = list(next(iter(metric_dict.values())).keys())

    for ds in datasets:
        for m in metrics:
            desc = describe_array(metric_dict[ds][m])
            desc_rows.append({
                "dataset": ds,
                "metric": m,
                **desc
            })

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a_name = datasets[i]
            b_name = datasets[j]
            for m in metrics:
                test_rows.append(
                    pairwise_stats(
                        metric_dict[a_name][m],
                        metric_dict[b_name][m],
                        a_name,
                        b_name,
                        m
                    )
                )

    desc_df = pd.DataFrame(desc_rows)
    test_df = pd.DataFrame(test_rows)

    desc_df.to_csv(
        os.path.join(out_dir, f"{prefix}_descriptive_stats.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    test_df.to_csv(
        os.path.join(out_dir, f"{prefix}_pairwise_tests.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return desc_df, test_df


# ============================================================
# Minkowski metrics
# ============================================================

def compute_minkowski_for_volume(vol):
    """
    Same metric logic as original eval scripts:
    - remove small objects, min_size=34, connectivity=3
    - porosity = clean pore voxel fraction
    - SSA = marching-cubes surface area / volume size
    - Euler density = Euler number / volume size
    """
    vol_clean = morphology.remove_small_objects(vol, min_size=34, connectivity=3)

    phi = float(vol_clean.sum() / vol_clean.size)

    vol_pad = np.pad(vol_clean, 1, mode="constant", constant_values=0)

    try:
        verts, faces, _, _ = measure.marching_cubes(vol_pad, level=0.5)
        surface_area = measure.mesh_surface_area(verts, faces)
        ssa = float(surface_area / vol_clean.size)
    except Exception:
        ssa = np.nan

    try:
        chi = measure.euler_number(vol_clean, connectivity=3)
        chi_density = float(chi / vol_clean.size)
    except Exception:
        chi_density = np.nan

    return phi, ssa, chi_density


def plot_minkowski(metrics, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    plot_items = [
        ("Porosity", "Porosity"),
        ("SpecificSurfaceArea", "Specific Surface Area"),
        ("EulerDensity", "Euler Characteristic Density"),
    ]

    for ax, (key, title) in zip(axes, plot_items):
        gt = np.asarray(metrics["GT"][key], dtype=float)
        gen = np.asarray(metrics["Gen"][key], dtype=float)

        bp = ax.boxplot(
            [gt, gen],
            labels=["GT", "Gen"],
            showmeans=True,
            patch_artist=True
        )

        bp["boxes"][0].set(facecolor="#d9d9d9", alpha=0.75)
        bp["boxes"][1].set(facecolor="#ffcccc", alpha=0.75)

        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)

        text = (
            f"GT: {np.nanmean(gt):.4g} ± {np.nanstd(gt, ddof=1):.2g}\n"
            f"Gen: {np.nanmean(gen):.4g} ± {np.nanstd(gen, ddof=1):.2g}"
        )
        ax.text(
            0.03, 0.97, text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "minkowski_boxplots_with_stats.png"), dpi=300)
    plt.close(fig)


def run_minkowski(gt_files, gen_dirs, out_dir):
    print("\n[Minkowski] Computing porosity / surface area / Euler density...")
    ensure_dir(out_dir)

    raw_rows = []
    load_rows = []

    metrics = {
        "GT": {"Porosity": [], "SpecificSurfaceArea": [], "EulerDensity": []},
        "Gen": {"Porosity": [], "SpecificSurfaceArea": [], "EulerDensity": []},
    }

    for group_name, paths in [("GT", gt_files), ("Gen", gen_dirs)]:
        for p in tqdm(paths, desc=f"Minkowski {group_name}"):
            try:
                vol, info = load_volume_as_pore(p)
                load_rows.append({"Group": group_name, **info})

                phi, ssa, chi = compute_minkowski_for_volume(vol)

                metrics[group_name]["Porosity"].append(phi)
                metrics[group_name]["SpecificSurfaceArea"].append(ssa)
                metrics[group_name]["EulerDensity"].append(chi)

                raw_rows.append({
                    "Group": group_name,
                    "Sample": os.path.basename(p),
                    "Path": p,
                    "FinalPorosityBeforeCleaning": info["FinalPorosity"],
                    "Inverted": info["Inverted"],
                    "Porosity": phi,
                    "SpecificSurfaceArea": ssa,
                    "EulerDensity": chi,
                })

            except Exception as e:
                print(f"[Minkowski warning] failed {p}: {e}")

    pd.DataFrame(load_rows).to_csv(
        os.path.join(out_dir, "minkowski_loading_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(raw_rows).to_csv(
        os.path.join(out_dir, "minkowski_raw_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    desc_df, test_df = save_descriptive_and_tests(metrics, out_dir, "minkowski")
    plot_minkowski(metrics, out_dir)

    return desc_df, test_df


# ============================================================
# S2 mismatch
# ============================================================

def calculate_s2_fft(vol):
    """
    Same as original eval_s2.py:
    S2(r) = <I(x)I(x+r)>.
    Raw autocorrelation normalized only by volume size.
    S2(0) = porosity.
    Three axis-aligned curves are averaged.
    """
    phase = vol.astype(np.float32)

    K = np.fft.rfftn(phase)
    S = K * np.conj(K)
    corr = np.fft.irfftn(S, s=phase.shape)
    corr = corr.real / phase.size

    N = min(vol.shape) // 2

    s2_x = corr[0, 0, :N]
    s2_y = corr[0, :N, 0]
    s2_z = corr[:N, 0, 0]

    s2_curve = (s2_x + s2_y + s2_z) / 3.0
    return s2_curve.astype(np.float64)


def s2_nrmse(curve, ref_curve, r_start=0, r_end=None):
    curve = np.asarray(curve, dtype=float)
    ref_curve = np.asarray(ref_curve, dtype=float)

    if r_end is None:
        r_end = min(len(curve), len(ref_curve))

    c = curve[r_start:r_end]
    r = ref_curve[r_start:r_end]

    numerator = np.sqrt(np.mean((c - r) ** 2))
    denominator = np.sqrt(np.mean(r ** 2)) + 1e-12

    return float(numerator / denominator * 100.0)


def plot_s2(gt_curves, gen_curves, out_dir):
    gt_mean = gt_curves.mean(axis=0)
    gen_mean = gen_curves.mean(axis=0)
    gt_sd = gt_curves.std(axis=0, ddof=1)
    gen_sd = gen_curves.std(axis=0, ddof=1)

    r = np.arange(len(gt_mean))

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    ax.plot(r, gt_mean, color="black", linewidth=2.2, label="GT mean")
    ax.fill_between(r, gt_mean - gt_sd, gt_mean + gt_sd, color="gray", alpha=0.25, label="GT ± SD")

    ax.plot(r, gen_mean, color="red", linewidth=2.2, label="Gen mean")
    ax.fill_between(r, gen_mean - gen_sd, gen_mean + gen_sd, color="red", alpha=0.15, label="Gen ± SD")

    ax.set_xlabel("Lag distance r [voxels]")
    ax.set_ylabel(r"$S_2(r)$")
    ax.set_title(r"Two-point Correlation Function $S_2$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    text = (
        f"GT $S_2(0)$={gt_mean[0]:.4f}\n"
        f"Gen $S_2(0)$={gen_mean[0]:.4f}\n"
        f"GT plateau≈{gt_mean[-1]:.4f}\n"
        f"Gen plateau≈{gen_mean[-1]:.4f}"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78)
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "s2_curves_gt_gen.png"), dpi=300)
    plt.close(fig)


def run_s2(gt_files, gen_dirs, out_dir, r_start=0, r_end=None):
    print("\n[S2] Computing S2 curves and mismatch...")
    ensure_dir(out_dir)

    gt_curves = []
    gen_curves = []
    gt_names = []
    gen_names = []
    load_rows = []

    for p in tqdm(gt_files, desc="S2 GT"):
        try:
            vol, info = load_volume_as_pore(p)
            load_rows.append({"Group": "GT", **info})

            curve = calculate_s2_fft(vol)
            gt_curves.append(curve)
            gt_names.append(os.path.basename(p))

        except Exception as e:
            print(f"[S2 warning] failed GT {p}: {e}")

    for p in tqdm(gen_dirs, desc="S2 Gen"):
        try:
            vol, info = load_volume_as_pore(p)
            load_rows.append({"Group": "Gen", **info})

            curve = calculate_s2_fft(vol)
            gen_curves.append(curve)
            gen_names.append(os.path.basename(p))

        except Exception as e:
            print(f"[S2 warning] failed Gen {p}: {e}")

    pd.DataFrame(load_rows).to_csv(
        os.path.join(out_dir, "s2_loading_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    gt_curves = np.asarray(gt_curves, dtype=float)
    gen_curves = np.asarray(gen_curves, dtype=float)

    min_len = min(gt_curves.shape[1], gen_curves.shape[1])
    gt_curves = gt_curves[:, :min_len]
    gen_curves = gen_curves[:, :min_len]

    if r_end is None or r_end > min_len:
        r_end = min_len

    gt_mean = gt_curves.mean(axis=0)
    gen_mean = gen_curves.mean(axis=0)

    curve_df = pd.DataFrame({
        "r": np.arange(min_len),
        "GT_mean": gt_mean,
        "GT_sd": gt_curves.std(axis=0, ddof=1),
        "Gen_mean": gen_mean,
        "Gen_sd": gen_curves.std(axis=0, ddof=1),
        "AbsDiff_mean_curves": np.abs(gt_mean - gen_mean),
    })
    curve_df.to_csv(
        os.path.join(out_dir, "s2_mean_curves.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    rows = []
    gen_mismatch = []

    for name, curve in zip(gen_names, gen_curves):
        e = s2_nrmse(curve, gt_mean, r_start=r_start, r_end=r_end)
        gen_mismatch.append(e)
        rows.append({
            "Group": "Gen_to_GTmean",
            "Sample": name,
            "S2_mismatch_NRMSE_percent": e,
        })

    gt_baseline = []

    for i, (name, curve) in enumerate(zip(gt_names, gt_curves)):
        ref = np.delete(gt_curves, i, axis=0).mean(axis=0)
        e = s2_nrmse(curve, ref, r_start=r_start, r_end=r_end)
        gt_baseline.append(e)
        rows.append({
            "Group": "GT_leave_one_out_baseline",
            "Sample": name,
            "S2_mismatch_NRMSE_percent": e,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "s2_mismatch_raw.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    metrics = {
        "GT_leave_one_out_baseline": {
            "S2_mismatch_NRMSE_percent": np.asarray(gt_baseline)
        },
        "Gen_to_GTmean": {
            "S2_mismatch_NRMSE_percent": np.asarray(gen_mismatch)
        }
    }

    desc_df, test_df = save_descriptive_and_tests(metrics, out_dir, "s2_mismatch")

    plot_s2(gt_curves, gen_curves, out_dir)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(gt_baseline, bins=25, alpha=0.55, label="GT leave-one-out baseline")
    ax.hist(gen_mismatch, bins=25, alpha=0.55, label="Gen-to-GT mean")
    ax.set_xlabel(r"$S_2$ mismatch NRMSE [%]")
    ax.set_ylabel("Count")
    ax.set_title(r"$S_2$ mismatch distribution")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    text = (
        f"GT baseline: {np.mean(gt_baseline):.3f} ± {np.std(gt_baseline, ddof=1):.3f}%\n"
        f"Gen: {np.mean(gen_mismatch):.3f} ± {np.std(gen_mismatch, ddof=1):.3f}%"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78)
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "s2_mismatch_distribution.png"), dpi=300)
    plt.close(fig)

    return desc_df, test_df


# ============================================================
# D_min baseline
# ============================================================

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack_volume_bits(vol):
    flat = np.asarray(vol, dtype=np.uint8).ravel()
    packed = np.packbits(flat)
    return packed, flat.size


def hamming_packed(packed_a, packed_b, n_bits):
    xor = np.bitwise_xor(packed_a, packed_b)
    diff = int(_POPCOUNT_TABLE[xor].sum())
    return diff / n_bits


def run_dmin(gt_files, gen_dirs, out_dir):
    print("\n[D_min] Computing Gen-to-GT and GT-to-GT baseline...")
    ensure_dir(out_dir)

    gt_packed = []
    gt_names = []
    n_bits_ref = None
    load_rows = []

    print("[D_min] Preloading and packing GT volumes...")
    for p in tqdm(gt_files, desc="Pack GT"):
        try:
            vol, info = load_volume_as_pore(p)
            load_rows.append({"Group": "GT", **info})

            packed, n_bits = pack_volume_bits(vol)

            if n_bits_ref is None:
                n_bits_ref = n_bits

            if n_bits != n_bits_ref:
                print(f"[D_min warning] skip shape mismatch: {p}")
                continue

            gt_packed.append(packed)
            gt_names.append(os.path.basename(p))

        except Exception as e:
            print(f"[D_min warning] failed GT {p}: {e}")

    pd.DataFrame(load_rows).to_csv(
        os.path.join(out_dir, "dmin_loading_check_gt.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("[D_min] Computing GT-to-GT leave-one-out baseline...")
    gt_rows = []
    gt_dmins = []

    for i in tqdm(range(len(gt_packed)), desc="GT-to-GT"):
        best = 1.0
        best_name = ""

        for j in range(len(gt_packed)):
            if i == j:
                continue

            d = hamming_packed(gt_packed[i], gt_packed[j], n_bits_ref)

            if d < best:
                best = d
                best_name = gt_names[j]

        gt_dmins.append(best)
        gt_rows.append({
            "Sample": gt_names[i],
            "D_min": best,
            "Nearest": best_name,
            "Group": "GT_to_GT_baseline",
        })

    print("[D_min] Computing Gen-to-GT nearest-neighbor distances...")
    gen_rows = []
    gen_dmins = []
    gen_load_rows = []

    for p in tqdm(gen_dirs, desc="Gen-to-GT"):
        try:
            vol, info = load_volume_as_pore(p)
            gen_load_rows.append({"Group": "Gen", **info})

            packed, n_bits = pack_volume_bits(vol)

            if n_bits != n_bits_ref:
                print(f"[D_min warning] skip shape mismatch: {p}")
                continue

            best = 1.0
            best_name = ""

            for j in range(len(gt_packed)):
                d = hamming_packed(packed, gt_packed[j], n_bits_ref)

                if d < best:
                    best = d
                    best_name = gt_names[j]

            gen_dmins.append(best)
            gen_rows.append({
                "Sample": os.path.basename(p),
                "D_min": best,
                "Nearest": best_name,
                "Group": "Gen_to_GT",
            })

        except Exception as e:
            print(f"[D_min warning] failed Gen {p}: {e}")

    pd.DataFrame(gen_load_rows).to_csv(
        os.path.join(out_dir, "dmin_loading_check_gen.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(gt_rows).to_csv(
        os.path.join(out_dir, "dmin_gt_to_gt_baseline.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(gen_rows).to_csv(
        os.path.join(out_dir, "dmin_gen_to_gt.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    metrics = {
        "GT_to_GT_baseline": {"D_min": np.asarray(gt_dmins)},
        "Gen_to_GT": {"D_min": np.asarray(gen_dmins)},
    }

    desc_df, test_df = save_descriptive_and_tests(metrics, out_dir, "dmin")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.hist(gt_dmins, bins=25, alpha=0.55, label="GT-to-GT baseline")
    ax.hist(gen_dmins, bins=25, alpha=0.55, label="Gen-to-GT")

    ax.set_xlabel(r"$D_{\min}$ Hamming distance")
    ax.set_ylabel("Count")
    ax.set_title(r"$D_{\min}$ baseline comparison")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    text = (
        f"GT baseline: {np.mean(gt_dmins):.4f} ± {np.std(gt_dmins, ddof=1):.4f}\n"
        f"Gen-to-GT: {np.mean(gen_dmins):.4f} ± {np.std(gen_dmins, ddof=1):.4f}"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78)
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dmin_distribution.png"), dpi=300)
    plt.close(fig)

    return desc_df, test_df


# ============================================================
# Report
# ============================================================

def write_summary_markdown(out_dir, sections):
    path = os.path.join(out_dir, "revision_metrics_summary.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Revision Metrics Summary\n\n")

        for name, desc_df, test_df in sections:
            f.write(f"## {name}\n\n")

            f.write("### Descriptive statistics\n\n")
            f.write(desc_df.to_csv(index=False))
            f.write("\n\n")

            f.write("### Pairwise tests\n\n")
            f.write(test_df.to_csv(index=False))
            f.write("\n\n")

    print(f"[Saved] {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_dir", type=str, default="../dataset/NPY/Bentheimer")
    parser.add_argument("--gen_dir", type=str, default="../samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--output_dir", type=str, default="../revision_experiments/reviewer_metrics_corrected")

    parser.add_argument("--limit_gt", type=int, default=None)
    parser.add_argument("--limit_gen", type=int, default=None)

    parser.add_argument("--skip_minkowski", action="store_true")
    parser.add_argument("--skip_s2", action="store_true")
    parser.add_argument("--skip_dmin", action="store_true")

    parser.add_argument("--s2_r_start", type=int, default=0)
    parser.add_argument("--s2_r_end", type=int, default=None)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    gt_files = list_gt_npy(args.gt_dir, limit=args.limit_gt)
    gen_dirs = list_gen_png_stacks(args.gen_dir, limit=args.limit_gen)

    print("=" * 80)
    print("Corrected revision metrics analysis")
    print(f"GT samples:  {len(gt_files)}")
    print(f"Gen samples: {len(gen_dirs)}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 80)

    sections = []

    if not args.skip_minkowski:
        out = os.path.join(args.output_dir, "minkowski")
        desc, tests = run_minkowski(gt_files, gen_dirs, out)
        sections.append(("Minkowski metrics", desc, tests))

    if not args.skip_s2:
        out = os.path.join(args.output_dir, "s2")
        desc, tests = run_s2(
            gt_files,
            gen_dirs,
            out,
            r_start=args.s2_r_start,
            r_end=args.s2_r_end
        )
        sections.append(("S2 mismatch", desc, tests))

    if not args.skip_dmin:
        out = os.path.join(args.output_dir, "dmin")
        desc, tests = run_dmin(gt_files, gen_dirs, out)
        sections.append(("D_min baseline", desc, tests))

    write_summary_markdown(args.output_dir, sections)

    print("=" * 80)
    print("Done.")
    print(f"All outputs saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()