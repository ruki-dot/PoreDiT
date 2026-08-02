#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_256_permeability.py

读取 256^3 渗透率表格，输出 reviewer 要的统计结果：
  - mean ± SD
  - mean 的 95% bootstrap CI
  - Gen - GT 的 mean difference 与 95% bootstrap CI
  - two-sample KS test
  - Mann-Whitney U test
  - Cohen's d / Hedges' g / Cliff's delta

支持两类输入：
1) 旧单方向表格：
   Type,Phi,Perm_Darcy
   GT / Gen
2) 新三方向表格：
   Dataset,Phi_effective_LCC,Kz_Darcy,Ky_Darcy,Kx_Darcy,K_mean_arithmetic_Darcy...

本地旧表格示例：
python stats_256_permeability.py ^
  --input "C:\\Users\\黄意卓\\Desktop\\rock\\code\\perm_results_live.csv" ^
  --output_dir "C:\\Users\\黄意卓\\Desktop\\rock\\code\\revision_experiments\\stats_256_old"

服务器新三方向结果示例：
python stats_256_permeability.py \
  --input /root/Xzk/RockProject/new/code/revision_experiments/permeability_xyz/permeability_xyz_merged_256.csv \
  --output_dir /root/Xzk/RockProject/new/code/revision_experiments/stats_256_xyz \
  --metrics Phi_effective_LCC Kz_Darcy Ky_Darcy Kx_Darcy K_mean_arithmetic_Darcy
"""

import csv
import math
import argparse
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def infer_group(row):
    for col in ["Type", "type", "Group", "group"]:
        if col in row:
            v = row[col].strip().lower()
            if v in ("gt", "groundtruth", "ground_truth", "real"):
                return "GT"
            if v in ("gen", "generated", "ours"):
                return "Gen"

    ds = row.get("Dataset", row.get("dataset", "")).lower()
    if "gt" in ds:
        return "GT"
    if "gen" in ds:
        return "Gen"
    return None


def as_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def collect_metric(rows, metric):
    vals = {"GT": [], "Gen": []}
    for r in rows:
        g = infer_group(r)
        if g not in vals or metric not in r:
            continue
        v = as_float(r[metric])
        if v is not None:
            vals[g].append(v)
    return {k: np.asarray(v, dtype=float) for k, v in vals.items()}


def bootstrap_ci_mean(x, n_boot=10000, alpha=0.05, seed=123):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, size=n)].mean()
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def bootstrap_ci_mean_diff(gen, gt, n_boot=10000, alpha=0.05, seed=456):
    gen = np.asarray(gen, dtype=float)
    gt = np.asarray(gt, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    ng, nt = len(gen), len(gt)
    for i in range(n_boot):
        g = gen[rng.integers(0, ng, size=ng)]
        t = gt[rng.integers(0, nt, size=nt)]
        diffs[i] = g.mean() - t.mean()
    return np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def cohen_d(gen, gt):
    n1, n0 = len(gen), len(gt)
    if n1 < 2 or n0 < 2:
        return np.nan
    v1 = np.var(gen, ddof=1)
    v0 = np.var(gt, ddof=1)
    pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
    if pooled <= 0:
        return np.nan
    return float((np.mean(gen) - np.mean(gt)) / np.sqrt(pooled))


def hedges_g(d, n1, n0):
    if not math.isfinite(d):
        return np.nan
    df = n1 + n0 - 2
    if df <= 1:
        return d
    return float(d * (1 - 3 / (4 * df - 1)))


def cliffs_delta(gen, gt):
    greater = 0
    less = 0
    for g in gen:
        greater += np.sum(g > gt)
        less += np.sum(g < gt)
    return float((greater - less) / (len(gen) * len(gt)))


def ks_test(gen, gt):
    if scipy_stats is None:
        return np.nan, np.nan
    r = scipy_stats.ks_2samp(gen, gt, alternative="two-sided", mode="auto")
    return float(r.statistic), float(r.pvalue)


def mwu_test(gen, gt):
    if scipy_stats is None:
        return np.nan, np.nan
    r = scipy_stats.mannwhitneyu(gen, gt, alternative="two-sided", method="auto")
    return float(r.statistic), float(r.pvalue)


def desc(arr, n_boot):
    ci = bootstrap_ci_mean(arr, n_boot=n_boot)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_mean_low": float(ci[0]),
        "ci95_mean_high": float(ci[1]),
    }


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.4e}"
    return f"{x:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input)
    if not rows:
        raise RuntimeError("empty input CSV")

    cols = list(rows[0].keys())
    if args.metrics:
        metrics = args.metrics
    else:
        preferred = [
            "Phi", "Perm_Darcy",
            "Phi_raw", "Phi_effective_LCC",
            "Kz_Darcy", "Ky_Darcy", "Kx_Darcy",
            "K_mean_arithmetic_Darcy", "K_mean_geometric_Darcy",
        ]
        metrics = [m for m in preferred if m in cols]

    descriptive_rows = []
    test_rows = []

    for metric in metrics:
        vals = collect_metric(rows, metric)
        gt = vals["GT"]
        gen = vals["Gen"]
        if len(gt) == 0 or len(gen) == 0:
            print(f"[Skip] {metric}: missing GT or Gen")
            continue

        dgt = desc(gt, args.n_boot)
        dge = desc(gen, args.n_boot)
        descriptive_rows.append({"metric": metric, "group": "GT", **dgt})
        descriptive_rows.append({"metric": metric, "group": "Gen", **dge})

        diff = float(gen.mean() - gt.mean())
        ci_diff = bootstrap_ci_mean_diff(gen, gt, n_boot=args.n_boot)
        ks_D, ks_p = ks_test(gen, gt)
        U, U_p = mwu_test(gen, gt)
        d = cohen_d(gen, gt)
        g = hedges_g(d, len(gen), len(gt))
        cd = cliffs_delta(gen, gt)

        test_rows.append({
            "metric": metric,
            "n_GT": len(gt),
            "n_Gen": len(gen),
            "GT_mean": float(gt.mean()),
            "Gen_mean": float(gen.mean()),
            "mean_diff_Gen_minus_GT": diff,
            "relative_diff_percent": float(diff / gt.mean() * 100.0) if gt.mean() != 0 else np.nan,
            "ci95_mean_diff_low": float(ci_diff[0]),
            "ci95_mean_diff_high": float(ci_diff[1]),
            "KS_D": ks_D,
            "KS_p": ks_p,
            "MWU_U": U,
            "MWU_p": U_p,
            "Cohen_d": d,
            "Hedges_g": g,
            "Cliffs_delta": cd,
        })

    desc_fields = [
        "metric", "group", "n", "mean", "sd", "median", "q1", "q3",
        "min", "max", "ci95_mean_low", "ci95_mean_high",
    ]
    test_fields = [
        "metric", "n_GT", "n_Gen", "GT_mean", "Gen_mean",
        "mean_diff_Gen_minus_GT", "relative_diff_percent",
        "ci95_mean_diff_low", "ci95_mean_diff_high",
        "KS_D", "KS_p", "MWU_U", "MWU_p",
        "Cohen_d", "Hedges_g", "Cliffs_delta",
    ]

    write_csv(out_dir / "descriptive_stats.csv", descriptive_rows, desc_fields)
    write_csv(out_dir / "pairwise_tests_GT_vs_Gen.csv", test_rows, test_fields)

    lookup = {(r["metric"], r["group"]): r for r in descriptive_rows}
    with open(out_dir / "statistical_report.md", "w", encoding="utf-8") as f:
        f.write("# Statistical report: GT vs Gen\n\n")
        if scipy_stats is None:
            f.write("> Warning: scipy not installed, KS and Mann-Whitney were not computed.\n\n")
        f.write("| Metric | GT mean ± SD | Gen mean ± SD | Δmean | 95% CI(Δ) | KS D / p | MWU p | Cohen d | Cliff δ |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in test_rows:
            m = r["metric"]
            gt_r = lookup[(m, "GT")]
            ge_r = lookup[(m, "Gen")]
            f.write(
                f"| {m} | {fmt(gt_r['mean'])} ± {fmt(gt_r['sd'])} | "
                f"{fmt(ge_r['mean'])} ± {fmt(ge_r['sd'])} | "
                f"{fmt(r['mean_diff_Gen_minus_GT'])} | "
                f"[{fmt(r['ci95_mean_diff_low'])}, {fmt(r['ci95_mean_diff_high'])}] | "
                f"{fmt(r['KS_D'])} / {fmt(r['KS_p'])} | "
                f"{fmt(r['MWU_p'])} | {fmt(r['Cohen_d'])} | {fmt(r['Cliffs_delta'])} |\n"
            )

    print(f"[Done] results saved under {out_dir}")


if __name__ == "__main__":
    main()
