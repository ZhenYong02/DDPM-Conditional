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

## Training Documentation

Training documentation and scripts are located in the `training/` directory. See separate training guide for model training workflows.

## License

This project is for research and educational purposes.
