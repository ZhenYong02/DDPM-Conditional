#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=ddpm_gen_uncond
#SBATCH --output=logs/ddpm_gen_uncond_%j.out
#SBATCH --error=logs/ddpm_gen_uncond_%j.err
#SBATCH --time=06:00:00

###############################################################################
# SLURM Script for Generating Unconditional Evaluation Images
#
# This script generates evaluation images using an unconditional DDPM model
# (no class conditioning).
#
# Usage:
#   1. Edit the configuration variables below
#   2. Submit: sbatch generate_eval_uncond.sh
#   3. Monitor: tail -f logs/ddpm_gen_uncond_<JOBID>.out
#
###############################################################################

# ==================== CONFIGURATION ====================
# Checkpoint path
CHECKPOINT="checkpoint/cifar10.pth"

# Total number of images to generate
NUM_IMAGES=200

# Output directory
OUTPUT_DIR="result_for_eval/uncond"

# Device (cuda or cpu)
DEVICE="cuda"

# =======================================================

# Print job information
echo "=================================================="
echo "SLURM Job: DDPM Unconditional Image Generation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of images: $NUM_IMAGES"
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
cd /home/msai/zhenyong001/DDPM-Conditional

# Print GPU information
if [ "$DEVICE" = "cuda" ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Python command
echo "Starting unconditional image generation..."
echo "=================================================="
echo ""

PYTHON_CMD="python -u generate_unconditioned.py \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --num_images $NUM_IMAGES \
    --device $DEVICE"

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
