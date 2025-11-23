# DDPM-Conditional: Conditional Diffusion Models for CIFAR-10

PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for CIFAR-10 image generation. This repository supports unconditional generation and three class-conditional methods: time embedding, input channel conditioning, and Classifier-Free Guidance (CFG).

## Environment Setup

```bash
git clone <repo-url>
cd DDPM-Conditional
conda create -n nodeenv python=3.13 -y
conda activate nodeenv
pip install -r requirements.txt
```

## Model Training

This repository provides training scripts for conditional DDPM models. Training scripts are located in the root directory.

### Training Scripts Overview

| Script | Config File | Purpose |
|--------|-------------|---------|
| `train.py` | `config.yml` | Standard conditional training (time/input strategies) |
| `train_cfg.py` | `config_cfg.yml` | CFG training with unconditional dropout |

### Standard Conditional Training (`train.py`)

Trains conditional DDPM models using either time embedding or input channel conditioning strategies.

**Usage:**
```bash
# Edit config.yml first, then run:
python train.py
```

**Important Configuration Parameters in `config.yml`:**

Open `config.yml` and configure these key parameters:

- **`run_name`**: Name for your checkpoint file (e.g., `"cifar10_time_emb"`)
- **`conditioning_strategy`**: Choose `"time"` (recommended) or `"input"`
  - `"time"`: Uses `UNet_emb_at_time` - class embedding at timestep level
  - `"input"`: Uses `UNet_emb_at_input` - class embedding at input channel level
- **`epochs`**: Total training epochs (default: 300, convergence around 200-250)
- **`batch_size`**: Training batch size (default: 256, reduce if OOM)
- **`lr`**: Learning rate (default: 0.0003)
- **`device`**: Training device (e.g., `"cuda:0"` or `"cpu"`)
- **`consume`**: Set to `True` to resume from checkpoint, `False` for fresh training
- **Model parameters**: `model_channels`, `num_res_blocks`, `dropout`, etc.
- **Dataset parameters**: `dataset` (e.g., `"cifar"`), `data_path`, `download`

The checkpoint will be saved to `checkpoint/{run_name}.pth`.

### CFG Training (`train_cfg.py`)

Trains DDPM models with Classifier-Free Guidance support. Always uses `UNet_emb_at_time` architecture.

**Usage:**
```bash
# Edit config_cfg.yml first, then run:
python train_cfg.py
```

**Important Configuration Parameters in `config_cfg.yml`:**

Open `config_cfg.yml` and configure these key CFG-specific parameters:

- **`run_name`**: Name for your checkpoint file (e.g., `"cifar10_cfg"`)
- **`cfg_dropout_prob`**: Probability of dropping class conditioning during training (default: 0.1)
  - `0.1` (10%): Standard CFG training, good balance
  - `0.15` (15%): More unconditional training, stronger guidance capability
  - `0.2` (20%): High dropout, may weaken pure conditional performance
- **`guidance_scale`**: Default guidance scale for generation (default: 3.0)
  - This is just a default value; can be overridden during generation
  - `1.0`: Standard conditional DDPM (no guidance)
  - `3.0-5.0`: Moderate guidance (recommended)
  - `5.0-7.0`: Strong guidance (better quality, less diversity)
- **Other parameters**: Same as `config.yml` (epochs, batch_size, lr, device, etc.)

**Note**: `conditioning_strategy` is NOT used in CFG training - it always uses time embedding.

### Resuming Training

To resume training from a checkpoint, set `consume: True` in your config file and ensure `consume_path` points to your checkpoint file.

### Dataset Compatibility Warning

⚠️ **IMPORTANT**: While this codebase is adapted from [Alokia/diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch), our modifications and testing are **specifically based on CIFAR-10**. There is **no guarantee** that the code will work perfectly with other datasets (MNIST, custom datasets) supported by the original Alokia implementation.

## Pre-trained Checkpoints

Pre-trained model checkpoints can be downloaded from:
**[Download Checkpoints (OneDrive)](https://entuedu-my.sharepoint.com/:f:/g/personal/zhenyong001_e_ntu_edu_sg/IgAAlnKx7_qTSaI3_YrjpbhwARCGR3s8653Z3zEf-OdJPe4?e=gtdssQ)**

Place downloaded checkpoints in the `checkpoint/` directory.

| Checkpoint | Method | Size | Description |
|------------|--------|------|-------------|
| `cifar10.pth` | unconditional | 429 MB | Pure DDPM (no class conditioning) |
| `cifar10_cond.pth` | time | 416 MB | Time embedding conditioning |
| `cifar10_cond_at_input.pth` | input | 410 MB | Input channel conditioning |
| `cifar10_cfg.pth` | cfg | 416 MB | Classifier-Free Guidance |

**Note on Unconditional Training**: While unconditional generation is supported with the pre-trained `cifar10.pth` checkpoint, this repository does **not provide training scripts for unconditional models**. For training unconditional CIFAR-10 DDPM from scratch, please refer to the original [Alokia/diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch) implementation.

## Image Generation

### CIFAR-10 Classes

Conditional generation produces images for these 10 classes:
- 0: airplane
- 1: automobile
- 2: bird
- 3: cat
- 4: deer
- 5: dog
- 6: frog
- 7: horse
- 8: ship
- 9: truck

### Unconditional Generation

Pure DDPM generation without class labels.

```bash
python generate_unconditioned.py \
    --checkpoint checkpoint/cifar10.pth \
    --output_dir result_for_eval/uncond \
    --num_images 200
```

**Parameters:**
- `--checkpoint`: Path to unconditional checkpoint
- `--output_dir`: Output directory for generated images
- `--num_images`: Total number of images to generate (default: 200)
- `--device`: `cuda` or `cpu` (default: `cuda`)

**Output Format**: Images saved as `image{index}.png` (e.g., `image1.png`, `image2.png`, ...)

### Conditional Generation

#### 1. Time Embedding Conditioning

Class information embedded at the timestep level.

```bash
python generate_conditioned.py \
    --method time \
    --checkpoint checkpoint/cifar10_cond.pth \
    --output_dir result_for_eval/time \
    --images_per_class 20
```

#### 2. Input Channel Conditioning

Class information concatenated at the input channel level.

```bash
python generate_conditioned.py \
    --method input \
    --checkpoint checkpoint/cifar10_cond_at_input.pth \
    --output_dir result_for_eval/input \
    --images_per_class 20
```

#### 3. Classifier-Free Guidance (CFG)

Enhanced quality through guided generation with adjustable guidance scale.

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

**Guidance Scale Values:**
- `0.0`: Pure unconditional (ignores class labels)
- `1.0`: Pure conditional (standard DDPM, no guidance)
- `3.0-7.0`: Typical CFG range (recommended)
- `10.0+`: Very strong guidance (may cause artifacts)

**Parameters:**
- `--method`: Conditioning method (`time`, `input`, or `cfg`)
- `--checkpoint`: Path to model checkpoint
- `--output_dir`: Output directory for generated images
- `--images_per_class`: Number of images per class (default: 20)
- `--guidance_scale`: CFG guidance scale, only for `--method cfg` (default: 3.0)
- `--device`: `cuda` or `cpu` (default: `cuda`)

**Output Format**: Images saved as `{class_name}{index}.png` (e.g., `airplane1.png`, `cat5.png`, ...)

## Sample Generated Images

For testing the evaluation notebooks, sample generated images can be downloaded from:
**[Download Sample Images (OneDrive)](https://entuedu-my.sharepoint.com/:f:/g/personal/zhenyong001_e_ntu_edu_sg/IgByeF_iG1jHRoe9Myq4vO3lAQbvLFrboYQCcrjbpNGnlBk?e=LO5w4B)**

⚠️ **Important Note on Sample Images**: These are **small sample sets** provided for testing the evaluation pipeline, **NOT** the full 10k images used to calculate the metrics in our project report. For **reliable and publishable metrics**, you should generate and evaluate **10,000 to 50,000 images** for CIFAR-10. Using fewer images will result in high variance and unreliable metric estimates.

## Evaluation Metrics

The `evaluation/` folder contains Jupyter notebooks for computing generation quality metrics.

### Available Metrics

1. **Inception Score (IS)** - Measures quality and diversity
2. **Fréchet Inception Distance (FID)** - Measures similarity to real data distribution
3. **Classification Accuracy** - Tests if generated images are recognizable
4. **Logit Score** - Measures confidence of classifier predictions

### Notebooks

- **`evaluation/accuracy_logit_IS.ipynb`**: Computes Inception Score, Classification Accuracy, and Logit Score
- **`evaluation/fid.ipynb`**: Computes Fréchet Inception Distance

### Usage

**Important**: Before running the notebooks, you must **edit the directory paths** inside the notebook code to point to your generated images and real CIFAR-10 data.

### Recommendations for Reliable Metrics

- **FID**: Generate at least 10,000 images (50,000 recommended for publication-quality results)
- **IS**: Generate at least 10,000 images
- **Accuracy/Logit**: Generate at least 5,000 images (1,000 minimum for rough estimates)
- Use the same number of images across all methods for fair comparison

## Repository Structure

```
DDPM-Conditional/
├── train.py                     # Standard conditional training script
├── train_cfg.py                 # CFG training script
├── config.yml                   # Standard training configuration
├── config_cfg.yml               # CFG training configuration
├── generate_unconditioned.py    # Unconditional generation script
├── generate_conditioned.py      # Conditional generation script
├── checkpoint/                  # Pre-trained model checkpoints
│   ├── cifar10.pth             # Unconditional model
│   ├── cifar10_cond.pth        # Time embedding
│   ├── cifar10_cond_at_input.pth  # Input channel
│   └── cifar10_cfg.pth         # CFG
├── model/
│   ├── UNet_emb_at_time.py     # Time embedding architecture
│   └── UNet_emb_at_input.py    # Input channel architecture
├── model_uncond/
│   └── UNet.py                 # Unconditional architecture
├── utils/
│   ├── engine.py               # Conditional DDPM trainer/sampler
│   ├── engine_cfg.py           # CFG-enabled trainer/sampler
│   └── tools.py, tools_cfg.py  # Training utilities
├── utils_uncond/
│   └── engine.py               # Unconditional DDPM sampler
├── dataset/                    # Dataset loaders
│   ├── __init__.py             # Dataset factory
│   ├── CIFAR.py, MNIST.py      # Dataset implementations
│   └── Custom.py
├── evaluation/                 # Evaluation metrics notebooks
│   ├── accuracy_logit_IS.ipynb # IS, Accuracy, Logit Score
│   └── fid.ipynb               # Fréchet Inception Distance
├── training_loss_curve/        # Training loss visualization
│   └── plot_loss_curve.ipynb   # Plot training loss curves
└── result_for_eval/            # Generated images output
```

## Credits

- Original DDPM paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Base implementation inspired by [Alokia/diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch)
- Extended with conditional generation and CFG support

## License

This project is for research and educational purposes.
