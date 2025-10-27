#!/bin/bash

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=ddpm_generate
#SBATCH --output=logs/ddpm_generate_%j.out
#SBATCH --error=logs/ddpm_generate_%j.err
#SBATCH --time=01:00:00

# Create logs and results directories
mkdir -p logs results

# Use absolute python path to bypass conda activation issues
PYTHON_PATH="/home/msai/zhenyong001/.conda/envs/nodeenv/bin/python"

# Print job information
echo "=== DDPM Generation Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================"

# Check GPU availability
nvidia-smi

# Print Python and PyTorch versions
$PYTHON_PATH -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check available checkpoints
echo "Available checkpoints:"
ls -la checkpoint/*.pth 2>/dev/null || echo "No checkpoints found"

# Set checkpoint path (modify this to your desired checkpoint)
CHECKPOINT_PATH="checkpoint/cifar10_cond.pth"

# Generate timestamp for unique filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting image generation with checkpoint: $CHECKPOINT_PATH"
echo "Generation started at: $(date)"

# Generate 8 images with denoising step visualization
# CIFAR-10 classes: airplane automobile bird cat deer dog frog horse ship truck
$PYTHON_PATH generate.py \
    -cp "$CHECKPOINT_PATH" \
    -bs 8 \
    --class_name automobile \
    --interval 50 \
    -sp "results/ddpm_denoising_steps_${TIMESTAMP}.png" \
    --show

echo "Denoising steps visualization saved to: results/ddpm_denoising_steps_${TIMESTAMP}.png"

# Also generate final results only (8 images in 2x4 grid)
echo "Generating final results..."
$PYTHON_PATH generate.py \
    -cp "$CHECKPOINT_PATH" \
    -bs 8 \
    --class_name automobile \
    --result_only \
    --nrow 4 \
    -sp "results/ddpm_final_${TIMESTAMP}.png" \
    --show

echo "Final results saved to: results/ddpm_final_${TIMESTAMP}.png"
echo "Job completed successfully!"
