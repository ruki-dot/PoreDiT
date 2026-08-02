#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_permeability_xyz.py

Supplementary permeability experiment for revision.

Main features:
1. Compute permeability in three directions: Kz, Ky, Kx.
2. Use batch=3 inside each GPU worker to calculate three directions together.
3. Use multiple GPUs in parallel, default GPU ids: 0,1,2,3.
4. Support:
   - GT 256^3 NPY samples
   - generated 256^3 PNG-stack samples
   - generated 512^3 PNG-stack split into 256^3 blocks
   - generated 1024^3 PNG-stack split into 256^3 blocks
5. Save raw results first, then optionally generate plots.
6. Support --resume.

Axis convention:
Loaded volume shape is vol[z, y, x].
Kz = force along axis 0.
Ky = force along axis 1.
Kx = force along axis 2.
"""

import os
import csv
import glob
import time
import math
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label


# ================= Global physical parameters =================

LATTICE_VISCOSITY = 0.166667
BODY_FORCE = 1e-5
MAX_ITER = 4000
WARMUP_STEPS = 100
AVG_LAST_STEPS = 200


# ================= Data loading =================

def read_png_gray(path):
    return np.array(Image.open(path).convert("L"))


def load_png_stack_as_pore_bool(folder):
    """
    Load one PNG/TIF stack.
    Returns True = pore.

    Logic follows the original permeability code:
    first use img > 127, then auto-flip if true fraction > 0.6.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.tif")) +
        glob.glob(os.path.join(folder, "*.tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No PNG/TIF slices found in {folder}")

    img0 = read_png_gray(files[0])
    vol = np.zeros((len(files), img0.shape[0], img0.shape[1]), dtype=bool)

    for i, f in enumerate(files):
        img = read_png_gray(f)
        vol[i] = (img > 127)

    if vol.mean() > 0.6:
        vol = ~vol

    return vol.astype(bool)


def load_npy_as_pore_bool(path):
    """
    Load one NPY volume.
    Returns True = pore.

    Logic follows the original permeability code.
    """
    arr = np.load(path)

    if arr.ndim == 4:
        arr = arr[0]

    if arr.dtype in [np.float32, np.float64]:
        if arr.max() > 1.0:
            arr = arr / arr.max()
        arr = (arr > 0.5)
    else:
        if arr.max() > 1:
            arr = (arr > 127)
        else:
            arr = (arr > 0)

    if arr.mean() > 0.6:
        arr = ~arr

    return arr.astype(bool)


def load_large_png_block(folder, bz, by, bx, block_size=256):
    """
    Load one 256^3 block from a large PNG-stack volume.
    Large sample is saved as slices: vol[z, y, x].
    """
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"No PNG slices found in {folder}")

    first = read_png_gray(files[0])
    z_total = len(files)
    y_total, x_total = first.shape

    z0, z1 = bz * block_size, (bz + 1) * block_size
    y0, y1 = by * block_size, (by + 1) * block_size
    x0, x1 = bx * block_size, (bx + 1) * block_size

    if z1 > z_total or y1 > y_total or x1 > x_total:
        raise ValueError(
            f"Block index out of range: bz={bz}, by={by}, bx={bx}, "
            f"shape=({z_total},{y_total},{x_total})"
        )

    vol = np.zeros((block_size, block_size, block_size), dtype=bool)

    for local_i, f in enumerate(files[z0:z1]):
        img = read_png_gray(f)
        block_img = img[y0:y1, x0:x1]
        vol[local_i] = (block_img > 127)

    if vol.mean() > 0.6:
        vol = ~vol

    return vol.astype(bool)


def list_gt_samples(gt_dir, limit=None):
    files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    if limit is not None:
        files = files[:limit]
    return files


def list_gen256_samples(gen256_dir, limit=None):
    subdirs = sorted([
        os.path.join(gen256_dir, d)
        for d in os.listdir(gen256_dir)
        if os.path.isdir(os.path.join(gen256_dir, d))
    ])
    subdirs = [
        p for p in subdirs
        if not os.path.basename(p).startswith("stats")
        and "target" not in os.path.basename(p).lower()
    ]
    if limit is not None:
        subdirs = subdirs[:limit]
    return subdirs


def get_large_shape(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"No PNG slices found in {folder}")
    first = read_png_gray(files[0])
    return len(files), first.shape[0], first.shape[1]


def make_large_block_tasks(folder, dataset_name, block_size=256):
    z_total, y_total, x_total = get_large_shape(folder)

    if z_total % block_size != 0 or y_total % block_size != 0 or x_total % block_size != 0:
        raise ValueError(
            f"Large volume shape ({z_total},{y_total},{x_total}) is not divisible by {block_size}"
        )

    nz = z_total // block_size
    ny = y_total // block_size
    nx = x_total // block_size

    tasks = []
    for bz in range(nz):
        for by in range(ny):
            for bx in range(nx):
                block_name = f"block_z{bz}_y{by}_x{bx}"
                tasks.append({
                    "kind": "large_block",
                    "dataset": dataset_name,
                    "type": "Gen",
                    "sample": f"{dataset_name}_volume",
                    "block": block_name,
                    "source_path": folder,
                    "bz": bz,
                    "by": by,
                    "bx": bx,
                    "block_size": block_size,
                })
    return tasks


# ================= LBM core =================

def get_largest_connected_component(vol_bool):
    if vol_bool.sum() == 0:
        return vol_bool

    labeled, num = label(vol_bool, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return vol_bool

    counts = np.bincount(labeled.ravel())
    if len(counts) <= 1:
        return vol_bool

    largest_idx = int(np.argmax(counts[1:]) + 1)
    return (labeled == largest_idx)


class LBMSolverXYZBatch:
    """
    Batch=3 D3Q19 BGK LBM solver.

    Batch index:
        0 -> Kz, force along z axis
        1 -> Ky, force along y axis
        2 -> Kx, force along x axis
    """

    def __init__(self, vol_bool, device):
        self.device = device

        self.phi_raw = float(vol_bool.sum() / vol_bool.size)

        self.vol_clean = get_largest_connected_component(vol_bool)
        self.phi_effective = float(self.vol_clean.sum() / self.vol_clean.size)

        self.mask_pore = torch.from_numpy(self.vol_clean).to(self.device, dtype=torch.bool)
        self.mask_solid = ~self.mask_pore

        self.nz, self.ny, self.nx = self.mask_pore.shape

        self.c = torch.tensor([
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
        ], device=self.device, dtype=torch.int64)

        w_np = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float32)
        self.w = torch.from_numpy(w_np).to(self.device)

        self.inv_idx = torch.tensor(
            [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15],
            device=self.device,
            dtype=torch.long
        )

        # f shape: [batch=3, q=19, z, y, x]
        self.f = torch.zeros((3, 19, self.nz, self.ny, self.nx), device=self.device, dtype=torch.float32)
        for i in range(19):
            self.f[:, i] = self.w[i]

        self.tau = 3.0 * LATTICE_VISCOSITY + 0.5
        self.omega = 1.0 / self.tau

    @torch.no_grad()
    def step(self):
        rho = torch.sum(self.f, dim=1)  # [3,z,y,x]

        u0 = torch.zeros_like(rho)  # z velocity
        u1 = torch.zeros_like(rho)  # y velocity
        u2 = torch.zeros_like(rho)  # x velocity

        for i in range(19):
            val = self.f[:, i]
            c0, c1, c2 = self.c[i].tolist()
            if c0 != 0:
                u0 += val * c0
            if c1 != 0:
                u1 += val * c1
            if c2 != 0:
                u2 += val * c2

        inv_rho = 1.0 / (rho + 1e-10)

        u0 *= inv_rho
        u1 *= inv_rho
        u2 *= inv_rho

        # Batch 0: force along z
        u0[0] += self.tau * BODY_FORCE * inv_rho[0]

        # Batch 1: force along y
        u1[1] += self.tau * BODY_FORCE * inv_rho[1]

        # Batch 2: force along x
        u2[2] += self.tau * BODY_FORCE * inv_rho[2]

        usq = u0 ** 2 + u1 ** 2 + u2 ** 2

        f_next = torch.empty_like(self.f)

        for i in range(19):
            c0, c1, c2 = self.c[i].tolist()
            cu = c0 * u0 + c1 * u1 + c2 * u2
            feq = self.w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq)
            f_next[:, i] = (1.0 - self.omega) * self.f[:, i] + self.omega * feq

        # Streaming
        for i in range(19):
            shift = self.c[i].tolist()
            if shift != [0, 0, 0]:
                f_next[:, i] = torch.roll(f_next[:, i], shifts=shift, dims=(1, 2, 3))

        # Bounce-back
        f_solid = f_next[:, :, self.mask_solid]  # [3,19,N_solid]
        f_next[:, :, self.mask_solid] = f_solid[:, self.inv_idx, :]

        self.f = f_next

        # Directional mean velocity over pore voxels
        mask = self.mask_pore
        uz_mean = torch.abs(u0[0][mask]).mean()
        uy_mean = torch.abs(u1[1][mask]).mean()
        ux_mean = torch.abs(u2[2][mask]).mean()

        return torch.stack([uz_mean, uy_mean, ux_mean])

    @torch.no_grad()
    def run(self, steps, warmup_steps, avg_last_steps, log_prefix="", step_log_interval=500):
        for _ in range(warmup_steps):
            self.step()

        u_sum = torch.zeros(3, device=self.device)
        sample_count = 0

        avg_start = max(0, steps - avg_last_steps)

        for i in range(steps):
            u = self.step()

            if torch.isnan(u).any():
                return np.zeros(3, dtype=float), self.phi_raw, self.phi_effective

            if i >= avg_start:
                u_sum += u
                sample_count += 1

            if step_log_interval and step_log_interval > 0:
                if (i + 1) % step_log_interval == 0 or (i + 1) == steps:
                    vals = u.detach().cpu().numpy()
                    print(
                        f"{log_prefix} LBM step {i + 1}/{steps} "
                        f"u=[Kz:{vals[0]:.3e}, Ky:{vals[1]:.3e}, Kx:{vals[2]:.3e}]",
                        flush=True
                    )

        u_avg = (u_sum / max(1, sample_count)).detach().cpu().numpy()
        return u_avg, self.phi_raw, self.phi_effective


def permeability_from_velocity(u_pore, phi_eff, resolution_um):
    if u_pore <= 0 or phi_eff <= 0:
        return 0.0

    k_lb = (u_pore * phi_eff) * LATTICE_VISCOSITY / BODY_FORCE
    k_phys = k_lb * (resolution_um ** 2)
    k_darcy = k_phys / 0.986923
    return float(k_darcy)


def compute_xyz_permeability(vol, device, args, log_prefix=""):
    solver = LBMSolverXYZBatch(vol, device=device)

    u_xyz, phi_raw, phi_eff = solver.run(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        avg_last_steps=args.avg_last_steps,
        log_prefix=log_prefix,
        step_log_interval=args.step_log_interval
    )

    kz = permeability_from_velocity(u_xyz[0], phi_eff, args.resolution)
    ky = permeability_from_velocity(u_xyz[1], phi_eff, args.resolution)
    kx = permeability_from_velocity(u_xyz[2], phi_eff, args.resolution)

    arr = np.array([kz, ky, kx], dtype=float)

    result = {
        "Phi_raw": phi_raw,
        "Phi_effective_LCC": phi_eff,
        "Kz_Darcy": kz,
        "Ky_Darcy": ky,
        "Kx_Darcy": kx,
        "K_mean_arithmetic_Darcy": float(arr.mean()),
        "K_mean_geometric_Darcy": float(np.exp(np.mean(np.log(np.maximum(arr, 1e-12))))),
    }

    del solver
    torch.cuda.empty_cache()

    return result


# ================= CSV utilities =================

CSV_FIELDS = [
    "Type",
    "Dataset",
    "Sample",
    "Block",
    "SourcePath",
    "ShapeZ",
    "ShapeY",
    "ShapeX",
    "Phi_raw",
    "Phi_effective_LCC",
    "Kz_Darcy",
    "Ky_Darcy",
    "Kx_Darcy",
    "K_mean_arithmetic_Darcy",
    "K_mean_geometric_Darcy",
    "GPU",
    "Elapsed_s",
]


def append_row_locked(csv_path, row, lock):
    with lock:
        csv_path = Path(csv_path)
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def load_existing_keys(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()

    keys = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            keys.add((r.get("Dataset", ""), r.get("Sample", ""), r.get("Block", "")))
    return keys


def summarize_csv(csv_path, out_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    numeric_cols = [
        "Phi_raw",
        "Phi_effective_LCC",
        "Kz_Darcy",
        "Ky_Darcy",
        "Kx_Darcy",
        "K_mean_arithmetic_Darcy",
        "K_mean_geometric_Darcy",
    ]

    groups = {}
    for r in rows:
        ds = r["Dataset"]
        groups.setdefault(ds, []).append(r)

    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write("Dataset,Metric,n,mean,sd,median,min,max\n")

        for ds, rs in groups.items():
            for col in numeric_cols:
                vals = []
                for r in rs:
                    try:
                        v = float(r[col])
                        if math.isfinite(v):
                            vals.append(v)
                    except Exception:
                        pass

                if not vals:
                    continue

                arr = np.array(vals, dtype=float)
                sd = arr.std(ddof=1) if len(arr) > 1 else 0.0

                f.write(
                    f"{ds},{col},{len(arr)},"
                    f"{arr.mean():.8g},{sd:.8g},{np.median(arr):.8g},"
                    f"{arr.min():.8g},{arr.max():.8g}\n"
                )


def merge_256_csvs(out_dir):
    out_dir = Path(out_dir)
    gt_csv = out_dir / "permeability_xyz_gt_256.csv"
    gen_csv = out_dir / "permeability_xyz_gen_256.csv"
    merged_csv = out_dir / "permeability_xyz_merged_256.csv"

    rows = []

    for p in [gt_csv, gen_csv]:
        if not p.exists():
            continue
        with p.open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))

    if not rows:
        return

    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    summarize_csv(merged_csv, out_dir / "summary_merged_256.csv")
    print(f"[Saved] {merged_csv}", flush=True)


# ================= Task building =================

def build_tasks_for_mode(mode, args):
    tasks = []

    if mode == "gt":
        samples = list_gt_samples(args.gt_dir, limit=args.limit_gt)
        for p in samples:
            tasks.append({
                "kind": "whole_npy",
                "type": "GT",
                "dataset": "GT_256",
                "sample": os.path.basename(p),
                "block": "whole",
                "source_path": p,
            })

    elif mode == "gen256":
        samples = list_gen256_samples(args.gen256_dir, limit=args.limit_gen256)
        for p in samples:
            tasks.append({
                "kind": "whole_png_stack",
                "type": "Gen",
                "dataset": "Gen_256",
                "sample": os.path.basename(p),
                "block": "whole",
                "source_path": p,
            })

    elif mode == "gen512":
        tasks = make_large_block_tasks(
            folder=args.gen512_dir,
            dataset_name="Gen_512_block256",
            block_size=args.block_size
        )

    elif mode == "gen1024":
        tasks = make_large_block_tasks(
            folder=args.gen1024_dir,
            dataset_name="Gen_1024_block256",
            block_size=args.block_size
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return tasks


def csv_path_for_mode(mode, out_dir):
    out_dir = Path(out_dir)

    if mode == "gt":
        return out_dir / "permeability_xyz_gt_256.csv"
    if mode == "gen256":
        return out_dir / "permeability_xyz_gen_256.csv"
    if mode == "gen512":
        return out_dir / "permeability_xyz_gen_512_blocks256.csv"
    if mode == "gen1024":
        return out_dir / "permeability_xyz_gen_1024_blocks256.csv"

    raise ValueError(mode)


def load_task_volume(task):
    kind = task["kind"]

    if kind == "whole_npy":
        return load_npy_as_pore_bool(task["source_path"])

    if kind == "whole_png_stack":
        return load_png_stack_as_pore_bool(task["source_path"])

    if kind == "large_block":
        return load_large_png_block(
            folder=task["source_path"],
            bz=task["bz"],
            by=task["by"],
            bx=task["bx"],
            block_size=task["block_size"]
        )

    raise ValueError(f"Unknown task kind: {kind}")


# ================= Multiprocessing workers =================

def split_tasks_round_robin(tasks, n):
    chunks = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        chunks[i % n].append(t)
    return chunks


def worker_process(gpu_id, tasks, csv_path, lock, args):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[GPU {gpu_id}] Worker started on {gpu_name}, tasks={len(tasks)}", flush=True)
    else:
        device = torch.device("cpu")
        print(f"[CPU] Worker started, tasks={len(tasks)}", flush=True)

    total = len(tasks)

    for idx, task in enumerate(tasks, start=1):
        key_str = f"{task['dataset']} | {task['sample']} | {task['block']}"
        prefix = f"[GPU {gpu_id}] [{idx}/{total}] {key_str}"

        try:
            print(f"{prefix} START", flush=True)
            t0 = time.time()

            vol = load_task_volume(task)

            if vol.ndim != 3:
                print(f"{prefix} SKIP non-3D volume shape={vol.shape}", flush=True)
                continue

            result = compute_xyz_permeability(
                vol=vol,
                device=device,
                args=args,
                log_prefix=prefix
            )

            elapsed = time.time() - t0

            row = {
                "Type": task.get("type", ""),
                "Dataset": task["dataset"],
                "Sample": task["sample"],
                "Block": task["block"],
                "SourcePath": task["source_path"],
                "ShapeZ": vol.shape[0],
                "ShapeY": vol.shape[1],
                "ShapeX": vol.shape[2],
                "GPU": gpu_id,
                "Elapsed_s": f"{elapsed:.3f}",
                **result
            }

            append_row_locked(csv_path, row, lock)

            print(
                f"{prefix} DONE elapsed={elapsed:.1f}s "
                f"Kz={result['Kz_Darcy']:.4f}, "
                f"Ky={result['Ky_Darcy']:.4f}, "
                f"Kx={result['Kx_Darcy']:.4f}, "
                f"Kmean={result['K_mean_arithmetic_Darcy']:.4f}",
                flush=True
            )

            del vol
            torch.cuda.empty_cache()

        except Exception as e:
            elapsed = 0.0
            try:
                elapsed = time.time() - t0
            except Exception:
                pass

            print(f"{prefix} ERROR after {elapsed:.1f}s: {repr(e)}", flush=True)

            row = {
                "Type": task.get("type", ""),
                "Dataset": task["dataset"],
                "Sample": task["sample"],
                "Block": task["block"],
                "SourcePath": task["source_path"],
                "ShapeZ": "",
                "ShapeY": "",
                "ShapeX": "",
                "Phi_raw": "",
                "Phi_effective_LCC": "",
                "Kz_Darcy": 0.0,
                "Ky_Darcy": 0.0,
                "Kx_Darcy": 0.0,
                "K_mean_arithmetic_Darcy": 0.0,
                "K_mean_geometric_Darcy": 0.0,
                "GPU": gpu_id,
                "Elapsed_s": f"{elapsed:.3f}",
            }
            append_row_locked(csv_path, row, lock)

    print(f"[GPU {gpu_id}] Worker finished.", flush=True)


def run_mode(mode, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_path_for_mode(mode, out_dir)

    tasks = build_tasks_for_mode(mode, args)

    if args.resume:
        existing = load_existing_keys(csv_path)
        before = len(tasks)
        tasks = [
            t for t in tasks
            if (t["dataset"], t["sample"], t["block"]) not in existing
        ]
        print(f"[{mode}] Resume: {before - len(tasks)} existing rows skipped.", flush=True)

    print(f"[{mode}] Tasks to run: {len(tasks)}", flush=True)

    if not tasks:
        summarize_csv(csv_path, out_dir / f"summary_{csv_path.stem}.csv")
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if not torch.cuda.is_available():
        gpu_ids = [-1]

    chunks = split_tasks_round_robin(tasks, len(gpu_ids))

    manager = mp.Manager()
    lock = manager.Lock()

    processes = []

    for gpu_id, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue

        p = mp.Process(
            target=worker_process,
            args=(gpu_id, chunk, str(csv_path), lock, args)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    summarize_csv(csv_path, out_dir / f"summary_{csv_path.stem}.csv")
    print(f"[{mode}] Finished. CSV saved to {csv_path}", flush=True)


# ================= Plotting =================

def make_plots(out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plot skipped] matplotlib unavailable: {e}", flush=True)
        return

    out_dir = Path(out_dir)

    csv_files = [
        out_dir / "permeability_xyz_merged_256.csv",
        out_dir / "permeability_xyz_gen_512_blocks256.csv",
        out_dir / "permeability_xyz_gen_1024_blocks256.csv",
    ]

    for csv_path in csv_files:
        if not csv_path.exists():
            continue

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        for kcol in ["Kz_Darcy", "Ky_Darcy", "Kx_Darcy", "K_mean_arithmetic_Darcy"]:
            plt.figure(figsize=(7, 5))

            groups = {}
            for r in rows:
                typ = r.get("Type", r.get("Dataset", "Unknown"))
                groups.setdefault(typ, []).append(r)

            for typ, rs in groups.items():
                phi = []
                kval = []
                for r in rs:
                    try:
                        phi.append(float(r["Phi_effective_LCC"]))
                        kval.append(float(r[kcol]))
                    except Exception:
                        pass

                if phi:
                    marker = "o" if typ == "GT" else "x"
                    plt.scatter(phi, kval, alpha=0.75, label=typ, marker=marker)

            plt.xlabel("Effective connected porosity")
            plt.ylabel(kcol)
            plt.yscale("log")
            plt.title(f"{csv_path.stem}: {kcol}")
            plt.grid(True, which="both", linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()

            out_png = out_dir / f"{csv_path.stem}_{kcol}.png"
            plt.savefig(out_png, dpi=300)
            plt.close()

            print(f"[Saved plot] {out_png}", flush=True)


# ================= Main =================

def parse_gpu_ids(s):
    if isinstance(s, list):
        return s
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["all", "gt", "gen256", "gen512", "gen1024"],
        default="all"
    )

    # Use 4 cards by default, leave the other 4 cards free.
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids, e.g. 0,1,2,3"
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--resolution", type=float, default=2.25)
    parser.add_argument("--block_size", type=int, default=256)

    parser.add_argument("--steps", type=int, default=MAX_ITER)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--avg_last_steps", type=int, default=AVG_LAST_STEPS)

    # Set to 0 if you want less console output.
    parser.add_argument("--step_log_interval", type=int, default=500)

    # Debug limits.
    parser.add_argument("--limit_gt", type=int, default=None)
    parser.add_argument("--limit_gen256", type=int, default=None)

    # New machine paths.
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/home/hyz/code/dataset/NPY/Bentheimer"
    )
    parser.add_argument(
        "--gen256_dir",
        type=str,
        default="/home/hyz/code/samples/Bentheimer/resolution_256/phi_cond_samples"
    )
    parser.add_argument(
        "--gen512_dir",
        type=str,
        default="/home/hyz/code/samples/Bentheimer/resolution_512/phi_cond_512_calibrated"
    )
    parser.add_argument(
        "--gen1024_dir",
        type=str,
        default="/home/hyz/code/samples/Bentheimer/resolution_1024/phi_cond_1024_calibrated_final"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/hyz/code/revision_experiments/permeability_xyz"
    )

    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("Permeability XYZ Evaluation", flush=True)
    print(f"Mode:       {args.mode}", flush=True)
    print(f"GPU ids:    {args.gpu_ids}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Steps:      {args.steps}", flush=True)
    print("=" * 80, flush=True)

    modes = ["gt", "gen256", "gen512", "gen1024"] if args.mode == "all" else [args.mode]

    for m in modes:
        run_mode(m, args)

    merge_256_csvs(args.output_dir)

    if args.plot:
        make_plots(args.output_dir)

    print("[All done]", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()