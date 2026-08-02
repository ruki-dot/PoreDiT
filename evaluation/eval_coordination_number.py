#!/usr/bin/env python3
"""
Coordination-number (CN) analysis for the PoreDiT revision.

This script uses the same pore-phase convention and small-object cleaning rule
as the corrected revision metrics / Euler-density evaluation:

    skimage.morphology.remove_small_objects(vol, min_size=34, connectivity=3)

After cleaning, a pore network is extracted with PoreSpy SNOW2.  The
coordination number of pore i is defined as the number of throats incident to
that pore in the extracted network, i.e. the degree of the pore node.

Outputs are written under:
    revision_experiments/reviewer_metrics_corrected/coordination_number
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps
from PIL import Image
from scipy import stats
from skimage import morphology
from tqdm import tqdm


MIN_SMALL_OBJECT_SIZE = 34
CONNECTIVITY = 3


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def auto_fix_pore_polarity(vol_bool: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Return bool volume with True=pore and auto-invert obvious solid=True inputs."""
    vol_bool = np.asarray(vol_bool, dtype=bool)
    raw_frac = float(vol_bool.mean())
    if raw_frac > 0.6:
        fixed = ~vol_bool
        return fixed, raw_frac, float(fixed.mean()), True
    return vol_bool, raw_frac, raw_frac, False


def load_volume_as_pore(path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load GT NPY or generated PNG stack as bool pore volume.

    Convention after return:
        True = pore, False = solid.
    """
    path = Path(path)
    info = {
        "Path": str(path),
        "Name": path.name,
        "SourceType": "dir_png" if path.is_dir() else "npy",
        "RawTrueFraction": np.nan,
        "FinalPorosity": np.nan,
        "Inverted": False,
        "Shape": "",
    }

    if path.is_dir():
        files = sorted(path.glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No PNG slices found in {path}")
        vol_gray = np.stack(
            [np.array(Image.open(f).convert("L")) for f in files],
            axis=0,
        )
        preliminary = vol_gray < 128
    else:
        arr = np.load(path)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype == bool:
            preliminary = arr.astype(bool)
        else:
            arr = np.asarray(arr)
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                preliminary = arr > 0.5
            else:
                preliminary = arr < 128

    vol, raw_frac, final_frac, inverted = auto_fix_pore_polarity(preliminary)
    info["RawTrueFraction"] = raw_frac
    info["FinalPorosity"] = final_frac
    info["Inverted"] = inverted
    info["Shape"] = "x".join(map(str, vol.shape))
    return vol.astype(bool), info


def list_gt_npy(gt_dir: str | Path, limit: int | None = None) -> list[Path]:
    files = sorted(Path(gt_dir).glob("*.npy"))
    return files[:limit] if limit else files


def list_gen_png_stacks(gen_dir: str | Path, limit: int | None = None) -> list[Path]:
    dirs = sorted([p for p in Path(gen_dir).iterdir() if p.is_dir()])
    return dirs[:limit] if limit else dirs


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, len(x), size=len(x))].mean()
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 123):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        diffs[i] = a[rng.integers(0, len(a), size=len(a))].mean() - b[rng.integers(0, len(b), size=len(b))].mean()
    return float(np.percentile(diffs, 100 * alpha / 2)), float(np.percentile(diffs, 100 * (1 - alpha / 2)))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    d = (np.mean(a) - np.mean(b)) / pooled
    correction = 1 - (3 / (4 * (len(a) + len(b)) - 9))
    return float(d * correction)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for val in a:
        gt += np.sum(val > b)
        lt += np.sum(val < b)
    return float((gt - lt) / (len(a) * len(b)))


def describe_by_group(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby("Group"):
        for metric in metrics:
            x = sub[metric].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            ci_low, ci_high = bootstrap_ci_mean(x)
            rows.append({
                "dataset": group,
                "metric": metric,
                "n": len(x),
                "mean": float(np.mean(x)) if len(x) else np.nan,
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "median": float(np.median(x)) if len(x) else np.nan,
                "q1": float(np.percentile(x, 25)) if len(x) else np.nan,
                "q3": float(np.percentile(x, 75)) if len(x) else np.nan,
                "min": float(np.min(x)) if len(x) else np.nan,
                "max": float(np.max(x)) if len(x) else np.nan,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    return pd.DataFrame(rows)


def pairwise_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        gt = df.loc[df["Group"] == "GT", metric].to_numpy(dtype=float)
        gen = df.loc[df["Group"] == "Gen", metric].to_numpy(dtype=float)
        gt = gt[np.isfinite(gt)]
        gen = gen[np.isfinite(gen)]
        ci_low, ci_high = bootstrap_ci_diff(gt, gen)
        ks_stat, ks_p = stats.ks_2samp(gt, gen)
        mw_stat, mw_p = stats.mannwhitneyu(gt, gen, alternative="two-sided")
        rows.append({
            "comparison": "GT_vs_Gen",
            "metric": metric,
            "n_GT": len(gt),
            "n_Gen": len(gen),
            "GT_mean": float(np.mean(gt)) if len(gt) else np.nan,
            "Gen_mean": float(np.mean(gen)) if len(gen) else np.nan,
            "mean_diff_GT_minus_Gen": float(np.mean(gt) - np.mean(gen)) if len(gt) and len(gen) else np.nan,
            "diff_ci95_low": ci_low,
            "diff_ci95_high": ci_high,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "mwu_stat": float(mw_stat),
            "mwu_p": float(mw_p),
            "hedges_g_GT_minus_Gen": hedges_g(gt, gen),
            "cliffs_delta_GT_minus_Gen": cliffs_delta(gt, gen),
        })
    return pd.DataFrame(rows)


def compute_cn_for_volume(path: Path, group: str) -> tuple[dict, pd.DataFrame, dict]:
    t0 = time.time()
    vol, load_info = load_volume_as_pore(path)
    vol_clean = morphology.remove_small_objects(
        vol,
        min_size=MIN_SMALL_OBJECT_SIZE,
        connectivity=CONNECTIVITY,
    )

    snow = ps.networks.snow2(vol_clean, voxel_size=1)
    net = snow.network if hasattr(snow, "network") else snow
    conns = np.asarray(net["throat.conns"], dtype=int)
    n_pores = int(len(net["pore.coords"]))
    n_throats = int(len(conns))

    if n_pores == 0:
        deg = np.array([], dtype=int)
    elif n_throats == 0:
        deg = np.zeros(n_pores, dtype=int)
    else:
        deg = np.bincount(conns.ravel(), minlength=n_pores).astype(int)

    connected = deg[deg > 0]
    row = {
        "Group": group,
        "Sample": path.name,
        "Path": str(path),
        "Shape": load_info["Shape"],
        "RawTrueFraction": load_info["RawTrueFraction"],
        "FinalPorosity": load_info["FinalPorosity"],
        "Inverted": load_info["Inverted"],
        "CleanPorosity": float(vol_clean.mean()),
        "SmallObjectMinSize": MIN_SMALL_OBJECT_SIZE,
        "Connectivity": CONNECTIVITY,
        "NumPores": n_pores,
        "NumThroats": n_throats,
        "MeanCN_AllPores": float(deg.mean()) if len(deg) else np.nan,
        "MeanCN_ConnectedPores": float(connected.mean()) if len(connected) else np.nan,
        "MedianCN_AllPores": float(np.median(deg)) if len(deg) else np.nan,
        "MaxCN": int(deg.max()) if len(deg) else 0,
        "FracCN0": float(np.mean(deg == 0)) if len(deg) else np.nan,
        "FracCN1": float(np.mean(deg == 1)) if len(deg) else np.nan,
        "FracCN2": float(np.mean(deg == 2)) if len(deg) else np.nan,
        "FracCN_ge3": float(np.mean(deg >= 3)) if len(deg) else np.nan,
        "FracCN_ge4": float(np.mean(deg >= 4)) if len(deg) else np.nan,
        "RuntimeSeconds": float(time.time() - t0),
    }

    pore_rows = pd.DataFrame({
        "Group": group,
        "Sample": path.name,
        "PoreIndex": np.arange(n_pores, dtype=int),
        "CN": deg,
    })
    return row, pore_rows, load_info


def plot_cn_distribution(pore_df: pd.DataFrame, out_dir: Path) -> None:
    max_bin = 10
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {"GT": "black", "Gen": "#d62728"}
    width = 0.36
    xs = np.arange(max_bin + 1)
    for offset, group in [(-width / 2, "GT"), (width / 2, "Gen")]:
        vals = pore_df.loc[pore_df["Group"] == group, "CN"].to_numpy(dtype=int)
        clipped = np.clip(vals, 0, max_bin)
        counts = np.bincount(clipped, minlength=max_bin + 1).astype(float)
        probs = counts / counts.sum()
        ax.bar(xs + offset, probs, width=width, color=colors[group], alpha=0.72, label=group)
    labels = [str(i) for i in range(max_bin)] + [f">={max_bin}"]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Coordination number")
    ax.set_ylabel("Probability")
    ax.set_title("Pooled coordination-number distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "coordination_number_distribution.png", dpi=300)
    plt.close(fig)


def plot_sample_metrics(sample_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("MeanCN_ConnectedPores", "Mean CN\n(CN > 0)"),
        ("FracCN1", "Fraction\nCN = 1"),
        ("FracCN_ge3", "Fraction\nCN >= 3"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 4.2))
    for ax, (metric, ylabel) in zip(axes, metrics):
        data = [
            sample_df.loc[sample_df["Group"] == "GT", metric].to_numpy(dtype=float),
            sample_df.loc[sample_df["Group"] == "Gen", metric].to_numpy(dtype=float),
        ]
        bp = ax.boxplot(data, labels=["GT", "Gen"], showmeans=True, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["lightgray", "#f5b7b1"]):
            patch.set_facecolor(color)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "coordination_number_sample_metrics.png", dpi=300)
    plt.close(fig)


def write_markdown(out_dir: Path, desc_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    with open(out_dir / "coordination_number_report.md", "w", encoding="utf-8") as f:
        f.write("# Coordination Number Analysis\n\n")
        f.write("Definition: CN is the graph degree of each pore node in the PoreSpy SNOW2-extracted pore network.\n\n")
        f.write(f"Cleaning: remove_small_objects(min_size={MIN_SMALL_OBJECT_SIZE}, connectivity={CONNECTIVITY}).\n\n")
        f.write("## Descriptive statistics\n\n")
        f.write(desc_df.to_csv(index=False))
        f.write("\n\n## Pairwise tests\n\n")
        f.write(test_df.to_csv(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="../dataset/NPY/Bentheimer")
    parser.add_argument("--gen_dir", default="../samples/Bentheimer/resolution_256/phi_cond_samples")
    parser.add_argument("--output_dir", default="../revision_experiments/reviewer_metrics_corrected/coordination_number")
    parser.add_argument("--limit_gt", type=int, default=None)
    parser.add_argument("--limit_gen", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    gt_files = list_gt_npy(args.gt_dir, args.limit_gt)
    gen_dirs = list_gen_png_stacks(args.gen_dir, args.limit_gen)
    targets = [("GT", p) for p in gt_files] + [("Gen", p) for p in gen_dirs]

    sample_csv = out_dir / "coordination_number_sample_metrics.csv"
    pore_csv = out_dir / "coordination_number_pore_values.csv"
    load_csv = out_dir / "coordination_number_loading_check.csv"

    sample_rows = []
    pore_frames = []
    load_rows = []
    completed = set()
    if args.resume and sample_csv.exists():
        old = pd.read_csv(sample_csv)
        sample_rows = old.to_dict("records")
        completed = set(zip(old["Group"].astype(str), old["Path"].astype(str)))
        if pore_csv.exists():
            pore_frames.append(pd.read_csv(pore_csv))
        if load_csv.exists():
            load_rows = pd.read_csv(load_csv).to_dict("records")

    print("=" * 80)
    print("Coordination-number analysis")
    print(f"GT samples:  {len(gt_files)}")
    print(f"Gen samples: {len(gen_dirs)}")
    print(f"Output dir:  {out_dir}")
    print(f"Resume:      {args.resume}; already completed: {len(completed)}")
    print("=" * 80)

    for group, path in tqdm(targets, desc="CN samples"):
        key = (group, str(path))
        if key in completed:
            continue
        try:
            row, pore_df, load_info = compute_cn_for_volume(path, group)
            sample_rows.append(row)
            pore_frames.append(pore_df)
            load_rows.append({"Group": group, **load_info})
        except Exception as exc:
            sample_rows.append({
                "Group": group,
                "Sample": path.name,
                "Path": str(path),
                "Status": f"FAILED: {type(exc).__name__}: {exc}",
            })

        pd.DataFrame(sample_rows).to_csv(sample_csv, index=False, encoding="utf-8-sig")
        if pore_frames:
            pd.concat(pore_frames, ignore_index=True).to_csv(pore_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(load_rows).to_csv(load_csv, index=False, encoding="utf-8-sig")

    sample_df = pd.read_csv(sample_csv)
    pore_df = pd.read_csv(pore_csv)

    metrics = [
        "NumPores",
        "NumThroats",
        "MeanCN_AllPores",
        "MeanCN_ConnectedPores",
        "MedianCN_AllPores",
        "FracCN0",
        "FracCN1",
        "FracCN2",
        "FracCN_ge3",
        "FracCN_ge4",
    ]
    desc_df = describe_by_group(sample_df, metrics)
    test_df = pairwise_tests(sample_df, metrics)
    desc_df.to_csv(out_dir / "coordination_number_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "coordination_number_pairwise_tests.csv", index=False, encoding="utf-8-sig")

    plot_cn_distribution(pore_df, out_dir)
    plot_sample_metrics(sample_df, out_dir)
    write_markdown(out_dir, desc_df, test_df)

    metadata = {
        "method": "PoreSpy SNOW2 pore-network extraction",
        "cn_definition": "number of throats incident to each extracted pore node",
        "cleaning": {
            "function": "skimage.morphology.remove_small_objects",
            "min_size": MIN_SMALL_OBJECT_SIZE,
            "connectivity": CONNECTIVITY,
        },
        "gt_dir": str(args.gt_dir),
        "gen_dir": str(args.gen_dir),
    }
    with open(out_dir / "coordination_number_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[Done] Coordination-number analysis complete.")


if __name__ == "__main__":
    main()
