# DDPM Image Generation - Inference Guide

PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation on CIFAR-10. This repository supports both unconditional generation and three class-conditional methods with pre-trained models ready for inference.

## Quick Start

### 1. Environment Setup

**Local Setup:**
```bash
git clone <repo-url>
cd DDPM-Conditional
conda create -n nodeenv python=3.13 -y
conda activate nodeenv
pip install -r requirements.txt
```

**HPC/SLURM Setup:**
```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate nodeenv
```

### 2. Generate Images

**Unconditional Generation:**
```bash
python generate_unconditioned.py \
    --checkpoint checkpoint/cifar10.pth \
    --output_dir result_for_eval/uncond \
    --num_images 200
```

**Conditional Generation:**
```bash
python generate_conditioned.py \
    --method <METHOD> \
    --checkpoint <CHECKPOINT_PATH> \
    --output_dir <OUTPUT_DIR> \
    --images_per_class 20
```

## Generation Methods

This repository implements unconditional and three different class-conditional approaches:

### 0. Unconditional Generation
Pure DDPM generation without class labels.
- **Model**: `UNet` (unconditional)
- **Checkpoint**: `checkpoint/cifar10.pth`
- **Usage**:
```bash
python generate_unconditioned.py \
    --checkpoint checkpoint/cifar10.pth \
    --output_dir result_for_eval/uncond \
    --num_images 200
```

## Conditional Methods

### 1. Time Embedding Conditioning
Class information embedded at the timestep level.
- **Model**: `UNet_emb_at_time`
- **Checkpoint**: `checkpoint/cifar10_cond.pth`
- **Usage**:
```bash
python generate_conditioned.py \
    --method time \
    --checkpoint checkpoint/cifar10_cond.pth \
    --output_dir result_for_eval/time \
    --images_per_class 20
```

### 2. Input Channel Conditioning
Class information concatenated at the input channel level.
- **Model**: `UNet_emb_at_input`
- **Checkpoint**: `checkpoint/cifar10_cond_at_input.pth`
- **Usage**:
```bash
python generate_conditioned.py \
    --method input \
    --checkpoint checkpoint/cifar10_cond_at_input.pth \
    --output_dir result_for_eval/input \
    --images_per_class 20
```

### 3. Classifier-Free Guidance (CFG)
Enhanced quality through guided generation with adjustable guidance scale.
- **Model**: `UNet_emb_at_time` + CFG sampler
- **Checkpoint**: `checkpoint/cifar10_cfg.pth`
- **Guidance Scale**: Controls quality vs diversity trade-off
  - `0.0`: Pure unconditional (ignores class labels)
  - `1.0`: Pure conditional (standard DDPM, no guidance)
  - `3.0-7.0`: Typical CFG range (recommended)
  - `10.0+`: Very strong guidance (may cause artifacts)

**Usage**:
```bash
# Moderate guidance (recommended)
python generate_conditioned.py \
    --method cfg \
    --checkpoint checkpoint/cifar10_cfg.pth \
    --output_dir result_for_eval/cfg_w3.0 \
    --images_per_class 20 \
    --guidance_scale 3.0

# Strong guidance for higher quality
python generate_conditioned.py \
    --method cfg \
    --checkpoint checkpoint/cifar10_cfg.pth \
    --output_dir result_for_eval/cfg_w7.0 \
    --images_per_class 20 \
    --guidance_scale 7.0
```

## SLURM Batch Generation

For HPC environments, use the provided SLURM scripts:

### Unconditional Generation

**Edit `generate_eval_uncond.sh`:**
```bash
CHECKPOINT="checkpoint/cifar10.pth"
NUM_IMAGES=200            # Total number of images
OUTPUT_DIR="result_for_eval/uncond"
DEVICE="cuda"
```

**Submit job:**
```bash
sbatch generate_eval_uncond.sh
```

**Monitor:**
```bash
tail -f logs/ddpm_gen_uncond_<JOBID>.out
```

### Conditional Generation

**Edit `generate_eval.sh`:**
```bash
METHOD="cfg"              # Options: "time", "input", "cfg"
IMAGES_PER_CLASS=20       # Number of images per class
GUIDANCE_SCALE=3.0        # Only for cfg method
DEVICE="cuda"
```

**Submit job:**
```bash
sbatch generate_eval.sh
```

**Monitor:**
```bash
tail -f logs/ddpm_gen_eval_<JOBID>.out
```

The scripts automatically:
- Select the correct checkpoint based on METHOD (for conditional)
- Set output directory
- Activate conda environment (`nodeenv`)
- Check GPU status
- Generate images
- Record timing and job statistics

## CIFAR-10 Classes

Conditional generation produces images for these 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Output Format

**Unconditional**: Images saved as `image{index}.png` (e.g., `image1.png`, `image2.png`, ...)

**Conditional**: Images saved as `{class_name}{index}.png` (e.g., `airplane1.png`, `cat5.png`, ...)

**Example output structure:**
```
result_for_eval/
├── uncond/
│   ├── image1.png
│   ├── image2.png
│   ├── ...
│   └── image200.png
├── time/
│   ├── airplane1.png
│   ├── airplane2.png
│   ├── ...
│   ├── truck20.png
├── input/
│   └── ...
├── cfg_w3.0/
│   └── ...
└── cfg_w7.0/
    └── ...
```

## Command-Line Arguments

### Unconditional Generation (`generate_unconditioned.py`)

**Required:**
- `--checkpoint`: Path to model checkpoint file
- `--output_dir`: Output directory for generated images

**Optional:**
- `--num_images`: Total number of images to generate (default: 200)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

### Conditional Generation (`generate_conditioned.py`)

**Required:**
- `--method`: Conditioning method (`time`, `input`, or `cfg`)
- `--checkpoint`: Path to model checkpoint file
- `--output_dir`: Output directory for generated images

**Optional:**
- `--images_per_class`: Number of images per class (default: 20)
- `--guidance_scale`: CFG guidance scale, only for `--method cfg` (default: 3.0)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

## Available Checkpoints

| Checkpoint | Method | Size | Description |
|------------|--------|------|-------------|
| `cifar10.pth` | unconditional | 429 MB | Pure DDPM (no class conditioning) |
| `cifar10_cond.pth` | time | 416 MB | Time embedding conditioning |
| `cifar10_cond_at_input.pth` | input | 410 MB | Input channel conditioning |
| `cifar10_cfg.pth` | cfg | 416 MB | Classifier-Free Guidance |

## Advanced Usage

### CPU Generation

**Unconditional:**
```bash
python generate_unconditioned.py \
    --checkpoint checkpoint/cifar10.pth \
    --output_dir results/cpu_gen \
    --num_images 50 \
    --device cpu
```

**Conditional:**
```bash
python generate_conditioned.py \
    --method cfg \
    --checkpoint checkpoint/cifar10_cfg.pth \
    --output_dir results/cpu_gen \
    --images_per_class 5 \
    --device cpu
```

### Compare Different CFG Scales
```bash
# Generate with multiple guidance scales to compare
for scale in 1.0 3.0 5.0 7.0 10.0; do
    python generate_conditioned.py \
        --method cfg \
        --checkpoint checkpoint/cifar10_cfg.pth \
        --output_dir result_for_eval/cfg_w${scale} \
        --images_per_class 10 \
        --guidance_scale ${scale}
done
```

## Model Architecture

All models use a U-Net backbone with:
- **Resolution**: 32×32 (CIFAR-10)
- **Channels**: 3 (RGB)
- **Diffusion Steps**: 1000 (T=1000)
- **Beta Schedule**: Linear from 1e-4 to 0.02

### Architecture Variants

1. **UNet** (unconditional): Pure DDPM with only timestep embeddings
2. **UNet_emb_at_time**: Time and class embeddings combined at timestep level
3. **UNet_emb_at_input**: Class embedding concatenated with input channels
4. **CFG-trained UNet_emb_at_time**: Trained with unconditional dropout for guidance

## Repository Structure

```
DDPM-Conditional/
├── generate_unconditioned.py    # Unconditional generation script
├── generate_conditioned.py      # Conditional generation script
├── generate_eval_uncond.sh      # SLURM script for unconditional generation
├── generate_eval.sh             # SLURM script for conditional generation
├── checkpoint/                  # Pre-trained model checkpoints
│   ├── cifar10.pth             # Unconditional model
│   ├── cifar10_cond.pth        # Time embedding
│   ├── cifar10_cond_at_input.pth  # Input channel
│   └── cifar10_cfg.pth         # CFG
├── model_uncond/
│   └── UNet.py                 # Unconditional architecture
├── model/
│   ├── UNet_emb_at_time.py     # Time embedding architecture
│   └── UNet_emb_at_input.py    # Input channel architecture
├── utils_uncond/
│   └── engine.py               # Unconditional DDPM sampler
├── utils/
│   ├── engine.py               # Conditional DDPM sampler
│   └── engine_cfg.py           # CFG-enabled sampler
├── training/                   # Training scripts (separate documentation)
└── result_for_eval/            # Generated images output
```

## Troubleshooting

### CUDA Out of Memory
Use CPU instead:
```bash
# Unconditional
python generate_unconditioned.py --checkpoint checkpoint/cifar10.pth \
    --output_dir results --num_images 10 --device cpu

# Conditional
python generate_conditioned.py --method cfg --checkpoint checkpoint/cifar10_cfg.pth \
    --output_dir results --images_per_class 1 --device cpu
```

### Checkpoint Not Found
Ensure checkpoint files are in the `checkpoint/` directory and paths are correct.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install torch torchvision tqdm pillow pyyaml
```

### Module Import Errors
If you get import errors for `model_uncond` or `utils_uncond`, ensure you're running from the project root directory:
```bash
cd /home/msai/zhenyong001/DDPM-Conditional
python generate_unconditioned.py --checkpoint checkpoint/cifar10.pth --output_dir results --num_images 10
```

## Credits

- Original DDPM paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Base implementation inspired by [Alokia/diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch)
- Extended with conditional generation and CFG support

## Model Training

This repository supports training both standard conditional DDPM models and Classifier-Free Guidance (CFG) models. Training scripts are located in the root directory.

### Training Scripts Overview

| Script | Config File | Purpose | Output |
|--------|-------------|---------|--------|
| `train.py` | `config.yml` | Standard conditional training (time/input strategies) | `checkpoint/{run_name}.pth` |
| `train_cfg.py` | `config_cfg.yml` | CFG training with unconditional dropout | `checkpoint/{run_name}.pth` |

### Standard Conditional Training (`train.py`)

Trains conditional DDPM models using either time embedding or input channel conditioning strategies.

**Configuration (`config.yml`):**

Key parameters to configure:

```yaml
# Run name for checkpoint file naming
run_name: "cifar10_cond_at_time_cfg"

# Conditioning Strategy - IMPORTANT!
# Controls which UNet architecture to use:
#   "time":  UNet_emb_at_time (class embedding at timestep level)
#   "input": UNet_emb_at_input (class embedding at input channel level)
conditioning_strategy: "time"  # Default: "time"

# Model parameters
Model:
  in_channels: 3              # RGB channels
  out_channels: 3
  model_channels: 128         # Base channel count
  attention_resolutions: [2]  # Resolutions to apply attention
  num_res_blocks: 2           # Residual blocks per level
  dropout: 0.1                # Dropout probability
  channel_mult: [1, 2, 2, 2]  # Channel multipliers per level
  num_classes: 10             # CIFAR-10 has 10 classes

# Dataset parameters
Dataset:
  dataset: "cifar"            # Options: "mnist", "cifar", "custom"
  batch_size: 256
  image_size: [32, 32]        # CIFAR-10 size
  data_path: "./data"
  download: True              # Auto-download CIFAR-10

# Diffusion parameters
Trainer:
  T: 1000                     # Diffusion timesteps
  beta: [0.0001, 0.02]        # Linear beta schedule

# Training parameters
device: "cuda:0"
epochs: 300                   # Total training epochs
lr: 0.0003                    # Learning rate (AdamW)

# Checkpoint settings
Callback:
  filepath: "./checkpoint/{run_name}.pth"
  save_freq: 1                # Save every N epochs

# Resume training
consume: False                # Set to True to resume
consume_path: "./checkpoint/{run_name}.pth"
```

**Local Training:**
```bash
# Edit config.yml to set conditioning_strategy and other parameters
python train.py
```

**HPC/SLURM Training:**
```bash
# Create a SLURM script (e.g., train_ddpm.sh)
sbatch train_ddpm.sh
```

**Example SLURM script:**
```bash
#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=ddpm_train
#SBATCH --output=logs/ddpm_train_%j.out
#SBATCH --error=logs/ddpm_train_%j.err
#SBATCH --time=06:00:00

mkdir -p logs checkpoint

# HPC setup
module load anaconda
eval "$(conda shell.bash hook)"
conda activate nodeenv

# Run training
python train.py 2>&1 | tee -a logs/training_detailed.log
```

**Switching Conditioning Strategies:**

To train with **time embedding** (recommended):
```yaml
conditioning_strategy: "time"
run_name: "cifar10_time_emb"
```

To train with **input channel** conditioning:
```yaml
conditioning_strategy: "input"
run_name: "cifar10_input_emb"
```

### CFG Training (`train_cfg.py`)

Trains DDPM models with Classifier-Free Guidance support. Always uses `UNet_emb_at_time` architecture.

**Configuration (`config_cfg.yml`):**

Key CFG-specific parameters:

```yaml
# Run name
run_name: "cifar10_cfg"

# CFG-Specific Parameters

# cfg_dropout_prob: Probability of dropping class conditioning during training
# This enables the model to learn both conditional and unconditional generation
# Typical values:
#   0.1 (10%):  Standard CFG training, good balance
#   0.15 (15%): More unconditional training, stronger guidance
#   0.2 (20%):  High dropout, may weaken pure conditional performance
cfg_dropout_prob: 0.1

# guidance_scale: Default guidance scale for generation
# This is just a default - can be overridden during generation
# Formula: ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
# Values:
#   0.0:     Pure unconditional (ignores class labels)
#   1.0:     Pure conditional (standard DDPM, no guidance)
#   3.0-5.0: Moderate guidance (recommended)
#   5.0-7.0: Strong guidance (better quality, less diversity)
#   7.0+:    Very strong guidance (may cause artifacts)
guidance_scale: 3.0

# Model, Dataset, Trainer, and training parameters
# (Same as config.yml, but conditioning_strategy is NOT used - always uses time embedding)
```

**Local Training:**
```bash
# Edit config_cfg.yml to set cfg_dropout_prob and other parameters
python train_cfg.py
```

**HPC/SLURM Training:**
```bash
# Create a SLURM script (e.g., train_ddpm_cfg.sh)
sbatch train_ddpm_cfg.sh
```

**Example SLURM script for CFG:**
```bash
#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=ddpm_train_cfg
#SBATCH --output=logs/ddpm_train_cfg_%j.out
#SBATCH --error=logs/ddpm_train_cfg_%j.err
#SBATCH --time=06:00:00

mkdir -p logs checkpoint

# HPC setup
module load anaconda
eval "$(conda shell.bash hook)"
conda activate nodeenv

# Run CFG training
python train_cfg.py 2>&1 | tee -a logs/training_detailed_cfg.log
```

### Resuming Training

Both training scripts support resuming from checkpoints.

**In `config.yml` or `config_cfg.yml`:**
```yaml
consume: True
consume_path: "./checkpoint/{run_name}.pth"
```

The checkpoint contains:
- Model state dict
- Optimizer state dict
- Training epoch
- Full configuration
- Checkpoint callback state

### Monitoring Training

**Check job status:**
```bash
squeue -u zhenyong001
```

**Monitor training output:**
```bash
# Standard training
tail -f logs/training_detailed.log

# CFG training
tail -f logs/training_detailed_cfg.log
```

**Check job efficiency:**
```bash
seff <JOB_ID>
```

### Configuration File Differences

| Parameter | `config.yml` | `config_cfg.yml` | Notes |
|-----------|--------------|------------------|-------|
| `conditioning_strategy` | ✅ Required | ❌ Not used | Controls UNet variant for `train.py` only |
| `cfg_dropout_prob` | ❌ Not used | ✅ Required | CFG-specific: unconditional dropout probability |
| `guidance_scale` | ❌ Not used | ✅ Required | CFG-specific: default guidance scale for generation |
| `Model`, `Dataset`, `Trainer` | ✅ | ✅ | Shared parameters |

**Key Differences:**
- `train.py` uses `conditioning_strategy` to select between `UNet_emb_at_time` and `UNet_emb_at_input`
- `train_cfg.py` always uses `UNet_emb_at_time` and requires CFG-specific parameters
- Both use similar model architecture and training hyperparameters

### Training Tips

1. **Start with time embedding**: `conditioning_strategy: "time"` generally performs better
2. **CFG dropout**: Start with `cfg_dropout_prob: 0.1` (10%) for CFG training
3. **Batch size**: Reduce if you encounter OOM errors (256 → 128 → 64)
4. **Learning rate**: 0.0003 works well for CIFAR-10; may need tuning for other datasets
5. **Epochs**: 300 epochs is recommended for CIFAR-10; convergence typically around epoch 200-250
6. **Checkpoints**: Save frequently (`save_freq: 1`) to prevent data loss

### Dataset Compatibility Warning

⚠️ **IMPORTANT**: While this codebase is adapted from [Alokia/diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch), our modifications and testing are **specifically based on CIFAR-10**. There is **no guarantee** that the code will work perfectly with other datasets (MNIST, custom datasets) supported by the original Alokia implementation.

If using datasets other than CIFAR-10:
- Adjust `image_size` in config files
- Modify `num_classes` for different class counts
- Test thoroughly and adjust hyperparameters as needed
- Input channel conditioning may require architecture adjustments for non-RGB images

## License

This project is for research and educational purposes.
