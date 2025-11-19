#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=ddpm_gen_eval
#SBATCH --output=logs/ddpm_gen_eval_%j.out
#SBATCH --error=logs/ddpm_gen_eval_%j.err
#SBATCH --time=06:00:00

###############################################################################
# Unified SLURM Script for Generating Evaluation Images
#
# This script generates evaluation images using different conditioning methods:
#   - time: Conditioning embedded at timestep (UNet_emb_at_time)
#   - input: Conditioning concatenated at input channels (UNet_emb_at_input)
#   - cfg: Classifier-Free Guidance (UNet_emb_at_time + CFG sampler)
#
# Usage:
#   1. Edit the configuration variables below
#   2. Submit: sbatch generate_eval.sh
#   3. Monitor: tail -f logs/ddpm_gen_eval_<JOBID>.out
#
###############################################################################

# ==================== CONFIGURATION ====================
# REQUIRED: Choose conditioning method
METHOD="input"  # Options: "time", "input", "cfg"

# Number of images per class (default: 20)
IMAGES_PER_CLASS=20

# CFG guidance scale (only used when METHOD="cfg")
# 0.0=unconditional, 1.0=conditional (no guidance), 3.0-7.0=typical CFG
GUIDANCE_SCALE=12.0

# Device (cuda or cpu)
DEVICE="cuda"

# =======================================================

# Validate METHOD
if [[ ! "$METHOD" =~ ^(time|input|cfg)$ ]]; then
    echo "Error: METHOD must be 'time', 'input', or 'cfg'"
    exit 1
fi

# Set checkpoint based on method
if [ "$METHOD" = "cfg" ]; then
    CHECKPOINT="checkpoint/cifar10_cfg.pth"
elif [ "$METHOD" = "time" ]; then
    CHECKPOINT="checkpoint/cifar10_cond.pth"
elif [ "$METHOD" = "input" ]; then
    CHECKPOINT="checkpoint/cifar10_cond_at_input.pth"
fi

# Set output directory based on method
if [ "$METHOD" = "cfg" ]; then
    OUTPUT_DIR="result_for_eval/cfg_w$GUIDANCE_SCALE"
else
    OUTPUT_DIR="result_for_eval/$METHOD"
fi

# Print job information
echo "=================================================="
echo "SLURM Job: DDPM Evaluation Image Generation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Method: $METHOD"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Images per class: $IMAGES_PER_CLASS"
echo "  Total images: $((IMAGES_PER_CLASS * 10))"
if [ "$METHOD" = "cfg" ]; then
    echo "  CFG guidance scale: $GUIDANCE_SCALE"
fi
echo "  Device: $DEVICE"
echo "=================================================="
echo ""

# Setup environment
echo "Setting up environment..."
module load anaconda
eval "$(conda shell.bash hook)"
conda activate nodeenv

# Print Python and PyTorch info
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Change to project directory
cd /home/msai/zhenyong001/diffusion-DDPM-pytorch

# Print GPU information
if [ "$DEVICE" = "cuda" ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Python command
echo "Starting image generation..."
echo "=================================================="
echo ""

PYTHON_CMD="python -u generate_conditioned.py \
    --method $METHOD \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --images_per_class $IMAGES_PER_CLASS \
    --device $DEVICE"

# Add guidance_scale for CFG method
if [ "$METHOD" = "cfg" ]; then
    PYTHON_CMD="$PYTHON_CMD --guidance_scale $GUIDANCE_SCALE"
fi

# Run generation
$PYTHON_CMD

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Generation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

# Print job statistics
if command -v seff &> /dev/null; then
    echo ""
    echo "Job efficiency statistics:"
    seff $SLURM_JOB_ID
fi

exit $EXIT_CODE
