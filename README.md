# PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction Using 3D Swin Transformers

This repository contains the PyTorch implementation for **PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction Using 3D Swin Transformers**.

PoreDiT reconstructs 3D digital rock pore structures with a 3D Swin-Transformer diffusion backbone. Large domains such as 512^3 and 1024^3 are reconstructed by tiled/sliding-window inference with Global Coherent Noise and overlap-weighted fusion, rather than by single-pass end-to-end gigavoxel generation.

## Repository Scope

This GitHub repository is intended for source code, preprocessing scripts, evaluation scripts, and reproducibility instructions. Large binary artifacts are intentionally excluded from normal Git history:

- `dataset/`: raw and processed micro-CT volumes
- `checkpoints/`: pretrained `.pth` weights and checkpoint archives
- `samples/`: generated `.npy` volumes and image exports
- `output/`: local training/evaluation outputs
- `code.zip` and review-only `.docx` files

The complete peer-review reproduction package, including source code, model weights, and data, is available via the Figshare private review link used in the manuscript: <https://figshare.com/s/e61d63b6c3d431635898>. The public GitHub repository listed in the manuscript is <https://github.com/ruki-dot/PoreDiT>.

## Project Structure

```text
code/
|-- src/                         # Training and sampling scripts
|   |-- train_bentheimer.py
|   |-- train_ketton.py
|   |-- sample_bentheimer_256.py
|   |-- sample_bentheimer_tiled.py
|   `-- sample_ketton.py
|-- preprocessing/
|   `-- prepare_dataset.py
|-- evaluation/                  # Morphological, statistical, and physics metrics
|   |-- eval_porosity.py
|   |-- eval_surface_area.py
|   |-- eval_euler.py
|   |-- eval_s2.py
|   |-- eval_permeability.py
|   |-- eval_permeability_xyz.py
|   |-- eval_connectivity_large.py
|   |-- eval_coordination_number.py
|   `-- eval_diversity.py
|-- revision_experiments/         # Additional scripts/results used during revision
|-- dataset/                      # Local data only; not committed
|-- checkpoints/                  # Local pretrained weights only; not committed
|-- samples/                      # Generated samples only; not committed
|-- output/                       # Training/evaluation outputs only; not committed
|-- requirements.txt
`-- README.md
```

## Environment Setup

We recommend Anaconda or Miniconda.

```bash
conda create -n poredit python=3.8
conda activate poredit
pip install -r requirements.txt
```

Core dependencies are listed in `requirements.txt`:

```text
torch>=1.10.0
torchvision>=0.11.0
numpy
pandas
scipy
matplotlib
seaborn
scikit-image
tqdm
Pillow
```

## Data Preparation

The project supports either processed binary `.npy` files or custom raw micro-CT volumes. Large data files are not committed to this repository. Download the reproduction package from Figshare or prepare your own data locally.

Expected local layout:

```text
dataset/
|-- Raw/
|   `-- Bentheimer_2d25um_binary.raw
`-- NPY/
    |-- Bentheimer/
    `-- Ketton/
```

Example preprocessing command:

```bash
python preprocessing/prepare_dataset.py \
  --raw_file "./dataset/Raw/Bentheimer_2d25um_binary.raw" \
  --output_dir "./dataset/NPY/Bentheimer" \
  --dims 1000 1000 1000 \
  --sample_dim 256 \
  --stride 128 \
  --augment
```

Binary convention: grain = 0, pore = 1.

## Training

### Bentheimer Sandstone

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,5 \
accelerate launch --num_processes=2 \
  src/train_bentheimer.py \
  --data_dir "./dataset/NPY/Bentheimer" \
  --output_dir "./output/bentheimer_train_official" \
  --num_train_epochs 300 \
  --train_batch_size 1 \
  --learning_rate 3e-5 \
  --mixed_precision fp16 \
  --eval_every 5 \
  --s2_lags 8 16 32 64 96 128 \
  --cond_loss_weight 1.5 \
  --grad_loss_weight 0.008 \
  --cond_warmup_epochs 3 \
  --cfg_scale 1.0
```

### Ketton Limestone

```bash
CUDA_VISIBLE_DEVICES=3 accelerate launch \
  --num_processes=1 \
  --main_process_port=29503 \
  src/train_ketton.py \
  --data_dir "./dataset/NPY/Ketton" \
  --output_dir "./output/ketton_train_official" \
  --patch_size 8 \
  --window_size 12 \
  --train_resolution 192 \
  --train_depth 192 \
  --pore_value 1 \
  --num_train_epochs 300 \
  --learning_rate 3e-5 \
  --fft_loss_weight 0.5 \
  --grad_loss_weight 0.2 \
  --ssim_loss_weight 0.2 \
  --style_loss_weight 10.0 \
  --s2_lags 2 4 8 16 32 64 128 \
  --seed 42
```

## Sampling

Place pretrained weights locally before sampling:

```text
checkpoints/Bentheimer/epoch_0300.pth
checkpoints/Ketton/epoch_0300.pth
```

### Bentheimer 256^3 Conditional Generation

```bash
CUDA_VISIBLE_DEVICES=0 python src/sample_bentheimer_256.py \
  --ckpt_path "./checkpoints/Bentheimer/epoch_0300.pth" \
  --cond_stats_path "./checkpoints/Bentheimer/cond_stats.json" \
  --output_dir "./samples/Bentheimer/resolution_256" \
  --num_samples 100 \
  --depth 256 \
  --resolution 256 \
  --cfg_scale 1.0 \
  --seed 1234
```

### Bentheimer Large-Scale Tiled Reconstruction

512^3 example:

```bash
CUDA_VISIBLE_DEVICES=1 python src/sample_bentheimer_tiled.py \
  --ckpt_path "./checkpoints/Bentheimer/epoch_0300.pth" \
  --dataset_dir "./dataset/NPY/Bentheimer" \
  --out_dir "./samples/Bentheimer/resolution_512" \
  --full_size 512 \
  --timesteps 1000
```

1024^3 example:

```bash
CUDA_VISIBLE_DEVICES=1 python src/sample_bentheimer_tiled.py \
  --ckpt_path "./checkpoints/Bentheimer/epoch_0300.pth" \
  --dataset_dir "./dataset/NPY/Bentheimer" \
  --out_dir "./samples/Bentheimer/resolution_1024" \
  --full_size 1024 \
  --timesteps 1000
```

### Ketton 192^3 Generation

```bash
CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python src/sample_ketton.py \
  --data_dir "./dataset/NPY/Ketton" \
  --output_dir "./samples/Ketton/resolution_192" \
  --ckpt_path "./checkpoints/Ketton/epoch_0300.pth" \
  --patch_size 8 \
  --window_size 12 \
  --train_depth 192 \
  --train_resolution 192 \
  --s2_lags 2 4 8 16 32 64 128 \
  --timesteps 1000 \
  --num_samples 100 \
  --mixed_precision fp16
```

## Evaluation

Porosity:

```bash
python evaluation/eval_porosity.py --sample_dir "./samples/Bentheimer/resolution_256/phi_cond_samples"
```

Specific surface area:

```bash
python evaluation/eval_surface_area.py
```

Euler characteristic:

```bash
python evaluation/eval_euler.py
```

Two-point correlation function:

```bash
python evaluation/eval_s2.py
```

Absolute permeability:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_permeability.py \
  --sample_dir "./samples/Bentheimer/resolution_256/phi_cond_samples" \
  --output_dir "./output/metrics" \
  --gpu 0
```

Directional permeability:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_permeability_xyz.py
```

Connectivity for large generated domains:

```bash
python evaluation/eval_connectivity_large.py \
  --sample_dir "./samples/Bentheimer/resolution_1024/phi_cond_1024_calibrated_final"
```

Diversity and nearest-neighbor novelty:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_diversity.py
```

## GitHub File-Size Notes

Normal GitHub repositories warn at files larger than 50 MiB and block files larger than 100 MiB. This project contains several artifacts above that limit, including raw volumes and pretrained weights. Keep those files outside normal Git history and distribute them through Figshare, Zenodo, GitHub Releases, or Git LFS only if the storage quota is acceptable.

The included `.gitignore` excludes the current `code.zip`, review-only `.docx`, `dataset/`, `checkpoints/`, `samples/`, and generated outputs so that the repository can be initialized and pushed as a code-only repository.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{HUANG2026PoreDiT,
  title={PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction Using 3D Swin Transformers},
  author={Huang Yizhuo and Sun Baoquan and Huang Haibo},
  journal={Results in Engineering},
  year={2026},
  note={Submitted}
}
```
