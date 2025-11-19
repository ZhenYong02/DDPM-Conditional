# DDPM Image Generation - Inference Guide

PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for class-conditional image generation on CIFAR-10. This repository supports three conditioning methods with pre-trained models ready for inference.

## Quick Start

### 1. Environment Setup

**Local Setup:**
```bash
git clone <repo-url>
cd diffusion-DDPM-pytorch
conda create -n ddpm python=3.10 -y
conda activate ddpm
pip install -r requirements.txt
```

**HPC/SLURM Setup:**
```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate nodeenv
```

### 2. Generate Images

Use the unified generation script that supports all three conditioning methods:

```bash
python generate_conditioned.py \
    --method <METHOD> \
    --checkpoint <CHECKPOINT_PATH> \
    --output_dir <OUTPUT_DIR> \
    --images_per_class 20
```

## Conditioning Methods

This repository implements three different conditioning approaches:

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

For HPC environments, use the provided SLURM script:

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

The script automatically:
- Selects the correct checkpoint based on METHOD
- Sets output directory (`result_for_eval/{method}` or `result_for_eval/cfg_w{scale}`)
- Activates conda environment
- Checks GPU status
- Generates images for all 10 CIFAR-10 classes
- Records timing and job statistics

## CIFAR-10 Classes

Generated images will be saved for these classes:
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

Images are saved with the naming convention: `{class_name}{index}.png`

**Example output structure:**
```
result_for_eval/
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

### Required Arguments
- `--method`: Conditioning method (`time`, `input`, or `cfg`)
- `--checkpoint`: Path to model checkpoint file
- `--output_dir`: Output directory for generated images

### Optional Arguments
- `--images_per_class`: Number of images per class (default: 20)
- `--guidance_scale`: CFG guidance scale, only for `--method cfg` (default: 3.0)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

## Available Checkpoints

| Checkpoint | Method | Size | Description |
|------------|--------|------|-------------|
| `cifar10_cond.pth` | time | 416 MB | Time embedding conditioning |
| `cifar10_cond_at_input.pth` | input | 410 MB | Input channel conditioning |
| `cifar10_cfg.pth` | cfg | 416 MB | Classifier-Free Guidance |

## Advanced Usage

### CPU Generation
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
- **Classes**: 10 (CIFAR-10 categories)

### Architecture Variants

1. **UNet_emb_at_time**: Time and class embeddings combined at timestep level
2. **UNet_emb_at_input**: Class embedding concatenated with input channels
3. **CFG-trained UNet_emb_at_time**: Trained with unconditional dropout for guidance

## Repository Structure

```
diffusion-DDPM-pytorch/
├── generate_conditioned.py      # Unified generation script
├── generate_eval.sh             # SLURM batch generation script
├── checkpoint/                  # Pre-trained model checkpoints
│   ├── cifar10_cond.pth
│   ├── cifar10_cond_at_input.pth
│   └── cifar10_cfg.pth
├── model/
│   ├── UNet_emb_at_time.py     # Time embedding architecture
│   └── UNet_emb_at_input.py    # Input channel architecture
├── utils/
│   ├── engine.py               # Standard DDPM sampler
│   └── engine_cfg.py           # CFG-enabled sampler
├── training/                   # Training scripts (separate documentation)
└── result_for_eval/            # Generated images output
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or use CPU:
```bash
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

## Credits

- Original DDPM paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Base implementation inspired by [Alokia/diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch)
- Extended with conditional generation and CFG support

## Training Documentation

Training documentation and scripts are located in the `training/` directory. See separate training guide for model training workflows.

## License

This project is for research and educational purposes.
