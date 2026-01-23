# The source code and datasets for the paper 'PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction Using 3D Swin Transformers' will be released in this repository upon the acceptance of the manuscript.

# PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction Using 3D Swin Transformers

This repository contains the official PyTorch implementation of the paper: **PoreDiT: "A Scalable Generative Model for Large-Scale Digital Rock
Reconstruction Using 3D Swin Transformers"**.

## 📝 Abstract

Digital rock physics (DRP) relies heavily on high-resolution 3D pore structure reconstruction. Traditional methods often struggle with the trade-off between resolution and field of view (FOV). We propose a **Dual-Scale Diffusion Model** that utilizes a Coarse-to-Fine generation strategy. This method can generate high-fidelity porous media structures at arbitrary scales (e.g., 256³, 512³, 1024³) while maintaining morphological consistency and physical properties (permeability, porosity).

## 📂 Project Structure

Please organize your directory as follows:

```text
code/
├── checkpoints/          # Pre-trained models
│   ├── Bentheimer/
│   │   └── epoch_0300.pth
│   └── Ketton/
│       └── epoch_0300.pth
├── dataset/              # Data root
│   ├── raw/              # Original raw images
│   └── NPY/              # Processed .npy files (Bentheimer & Ketton)
├── samples/              # Generated results
├── src/                  # Source code for training and sampling
├── processing/           # Data preprocessing scripts
├── evaluation/           # Evaluation metrics (Permeability & Diversity)
└── requirements.txt      # Environment dependencies
```

## 🛠️ Environment Setup
We recommend using Anaconda or Miniconda to manage the environment.
1.Create a new environment:
```bash
conda create -n rock_diffusion python=3.8
conda activate rock_diffusion
```
2.Install dependencies: Create a file named  `requirements.txt` with the content below, then run  `pip install -r requirements.txt` to install the dependencies.

requirements.txt content:
```bash
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

## 💾 Data Preparation

The project supports two modes for data preparation: using the provided pre-processed datasets or processing your own raw micro-CT images.

### Option A: Use Provided Pre-processed Data (Ready to Use)
We have already processed and placed the binary dataset files in the `dataset/NPY/` directory. 
* **Bentheimer Sandstone:** Available in `dataset/NPY/Bentheimer`
* **Ketton Limestone:** Available in `dataset/NPY/Ketton`

These files are pre-converted to binary format (0 for grain, 1 for pore) and are ready for training or evaluation immediately. No further action is required.

### Option B: Process Custom Raw Data
If you wish to train on your own Digital Rock Physics (DRP) data, please follow these steps:

1.  **Prepare Raw Data:** Place your raw micro-CT volume files into `dataset/raw/`. 
    * *Note: We recommend using raw volumes with dimensions approximating $1000^3$ voxels for optimal sub-volume extraction.*

2.  **Run Preprocessing:**
    Run the script to convert raw images into binary `.npy` volumes and perform data splitting.
    ```bash
    # Data Preprocessing (Example for Bentheimer)
    python preprocessing/prepare_dataset.py \
      --raw_file "./dataset/Raw/Bentheimer_2d25um_binary.raw" \
      --output_dir "./dataset/NPY/Bentheimer" \
      --dims 1000 1000 1000 \
      --sample_dim 256 \
      --stride 128 \
      --augment
    ```
    The script will automatically process the raw files and populate the `dataset/NPY/` directory.

## 🚀 Training

We demonstrate the training and sampling procedures using the representative samples discussed in our paper. Users may also employ their own datasets; for this purpose, we recommend using the `src/train_ketton.py` script as a baseline due to its enhanced generalization capabilities. The instructions below focus on reproducing the specific generated samples showcased in the manuscript.

To train the models using the provided configurations:

**Train Model (Bentheimer Example):**
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

**Train Model (Ketton Example - Recommended for Custom Data)**:
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
The checkpoints will be automatically saved to the checkpoints/ directory upon completion.

## ⚡ Sampling
The following commands allow you to reproduce the generated samples and reconstruction results presented in the paper using the pre-trained models.

### 1.Bentheimer Sandstone (Standard 256³ Generation)
Generates samples with specific target porosity conditions.
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

### 2.Bentheimer Large-Scale (Tiled Generation 512³ / 1024³)
Demonstrates the scalability of the model using tiled sampling strategies.
```bash
CUDA_VISIBLE_DEVICES=1 python src/sample_bentheimer_tiled.py \
  --ckpt_path "./checkpoints/Bentheimer/epoch_0300.pth" \
  --dataset_dir "./dataset/NPY/Bentheimer" \
  --out_dir "./samples/Bentheimer/resolution_512" \
  --full_size 512 \
  --timesteps 1000
```
```bash
CUDA_VISIBLE_DEVICES=1 python src/sample_bentheimer_tiled.py \
  --ckpt_path "./checkpoints/Bentheimer/epoch_0300.pth" \
  --dataset_dir "./dataset/NPY/Bentheimer" \
  --out_dir "./samples/Bentheimer/resolution_1024" \
  --full_size 1024 \
  --timesteps 1000
```

### 3.Ketton Limestone (192³ Generation)
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

## 📊 Evaluation & Physics Metrics
We provide standalone scripts to reproduce the metric evaluations reported in the paper. These scripts operate on the generated samples in `samples/` and compare them against the training data in `dataset/`.

### 1. Porosity Verification
Verifies if the generated samples match the target porosity distribution.
```bash
python evaluation/eval_porosity.py --sample_dir "./samples/Bentheimer/resolution_256/phi_cond_samples"
```

### 2. Specific Surface Area (SSA)
Calculates the specific surface area, a critical morphological metric for reaction kinetics.
```bash
python evaluation/eval_surface_area.py
```
### 3. Euler Characteristic (Topology)
Computes the Euler characteristic density to evaluate the topological connectivity and complexity of the pore network.
```bash
python evaluation/eval_euler.py
```

### 4.Two-Point Correlation Function ($S_2$)
Computes the two-point correlation function $S_2(r)$ to assess the spatial structure and statistical equivalence of the generated media.
```bash
python evaluation/eval_s2.py
```

### 5.Absolute Permeability (LBM)

Calculates the absolute permeability ($K$) using a D3Q19 Lattice Boltzmann Method (LBM) solver implemented in PyTorch (GPU-accelerated).

**Logic:** Uses Darcy's Law implementation with D3Q19 BGK collision model.

**Target:** samples/Bentheimer/resolution_256/phi_cond_samples
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_permeability.py \
  --sample_dir "./samples/Bentheimer/resolution_256/phi_cond_samples" \
  --output_dir "./output/metrics" \
  --gpu 0
```

The script automatically reads samples from the default sample directory and outputs `metric_permeability.txt`.

### 6.Diversity & Novelty Analysis
Evaluates the generative diversity and checks for data leakage (memorization) by calculating the Distance to Nearest Neighbor (D_NN) via Hamming distance.

**Logic**: Compares every generated sub-volume against the entire training dataset to find the closest real sample.
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_diversity.py
```

### 7.Connectivity Analysis
Analyzes the connected porosity ratio to ensure the generated pore networks are percolating (essential for fluid transport).
```bash
python evaluation/eval_connectivity_large.py \
  --sample_dir "./samples/Bentheimer/resolution_1024/phi_cond_1024_calibrated_final"
```

## 📥 Pre-trained Checkpoints
We provide pre-trained weights for reproducibility. Ensure they are placed at:

`checkpoints/Bentheimer/epoch_0300.pth`

`checkpoints/Ketton/epoch_0300.pth`

## 📝 Citation
If you use this code or dataset, please cite our paper:
```
@article{HUANG2026PoreDiT,
  title={PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction Using 3D Swin Transformers},
  author={Huang Yizhuo and Baoquan Sun and Haibo Huang},
  journal={Computational Materials Science},
  year={2026},
  note={Submitted}
}
```

## ✉️ Contact
For any questions regarding the code or the paper, please open an issue or contact [23307130266@m.fudan.edu.cn].
