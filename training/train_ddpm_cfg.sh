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

###############################################################################
# SLURM Script for Training DDPM with Classifier-Free Guidance (CFG)
#
# This script trains a DDPM model with CFG support by:
# 1. Using conditional dropout during training (configured in config_cfg.yml)
# 2. Enabling the model to learn both conditional and unconditional generation
# 3. Allowing CFG-guided sampling at generation time
#
# Usage:
#   1. Configure CFG parameters in config_cfg.yml:
#      - cfg_dropout_prob: Probability of dropping conditioning (0.1-0.2)
#      - guidance_scale: Default guidance scale for generation (3.0-7.0)
#   2. Submit: sbatch train_ddpm_cfg.sh
#   3. Monitor: tail -f logs/ddpm_train_cfg_<JOBID>.out
#   4. Check detailed logs: tail -f logs/training_detailed_cfg.log
#
###############################################################################

# Create logs and checkpoint directories
mkdir -p logs checkpoint

# Use absolute python path to bypass conda activation issues
PYTHON_PATH="/home/msai/zhenyong001/.conda/envs/nodeenv/bin/python"

# Verify environment and test imports
echo "Python path: $PYTHON_PATH" | tee -a logs/training_status_cfg.log
$PYTHON_PATH -c "import torch, torchvision; print('PyTorch and torchvision imported successfully')" | tee -a logs/training_status_cfg.log

# Print job information to both stdout and a status file
echo "=== DDPM CFG Training Job Started ===" | tee logs/training_status_cfg.log
echo "Job ID: $SLURM_JOB_ID" | tee -a logs/training_status_cfg.log
echo "Node: $SLURMD_NODENAME" | tee -a logs/training_status_cfg.log
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a logs/training_status_cfg.log
echo "Start time: $(date)" | tee -a logs/training_status_cfg.log
echo "Working directory: $(pwd)" | tee -a logs/training_status_cfg.log
echo "Training Mode: Classifier-Free Guidance (CFG)" | tee -a logs/training_status_cfg.log
echo "================================" | tee -a logs/training_status_cfg.log

# Check GPU availability
nvidia-smi | tee -a logs/training_status_cfg.log

# Print Python and PyTorch versions
$PYTHON_PATH -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" | tee -a logs/training_status_cfg.log

# Create a progress monitoring script for CFG training
cat > logs/monitor_progress_cfg.py << 'EOF'
#!/usr/bin/env python3
import os
import glob
from datetime import datetime

def monitor_training():
    print(f"=== DDPM CFG Training Progress Monitor ===")
    print(f"Check time: {datetime.now()}")

    # Check if training is running
    if os.path.exists("logs/training_status_cfg.log"):
        print("\n--- Job Status ---")
        with open("logs/training_status_cfg.log", "r") as f:
            print(f.read())

    # Check for checkpoints
    checkpoints = glob.glob("checkpoint/*_cfg.pth")
    if checkpoints:
        print(f"\n--- CFG Checkpoints Found ---")
        for cp in sorted(checkpoints):
            stat = os.stat(cp)
            print(f"{cp} - Modified: {datetime.fromtimestamp(stat.st_mtime)}")
    else:
        print("\n--- No CFG checkpoints found yet ---")

    # Check recent log output
    log_files = glob.glob("logs/ddpm_train_cfg_*.out")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"\n--- Recent Training Output (last 20 lines) ---")
        print(f"Log file: {latest_log}")
        with open(latest_log, "r") as f:
            lines = f.readlines()
            print("".join(lines[-20:]))

    print(f"\n=== End Monitor ===\n")

if __name__ == "__main__":
    monitor_training()
EOF

chmod +x logs/monitor_progress_cfg.py

# Print CFG-specific information
echo "" | tee -a logs/training_status_cfg.log
echo "=== CFG Configuration ===" | tee -a logs/training_status_cfg.log
echo "Reading configuration from config_cfg.yml..." | tee -a logs/training_status_cfg.log
$PYTHON_PATH -c "
from utils.tools_cfg import load_yaml
config = load_yaml('config_cfg.yml')
print(f\"CFG Dropout Probability: {config.get('cfg_dropout_prob', 0.1):.2%}\")
print(f\"Default Guidance Scale: {config.get('guidance_scale', 3.0)}\")
print(f\"Model: UNet_emb_at_time\")
print(f\"Dataset: {config['Dataset']['dataset']}\")
print(f\"Epochs: {config['epochs']}\")
" | tee -a logs/training_status_cfg.log
echo "========================" | tee -a logs/training_status_cfg.log
echo "" | tee -a logs/training_status_cfg.log

# Start training with progress logging
echo "Starting DDPM CFG training..." | tee -a logs/training_status_cfg.log
echo "Training started at: $(date)" | tee -a logs/training_status_cfg.log

# Run CFG training and capture both stdout and stderr
$PYTHON_PATH train_cfg.py 2>&1 | tee -a logs/training_detailed_cfg.log

# Print completion info
echo "Training completed at: $(date)" | tee -a logs/training_status_cfg.log
echo "Job finished successfully!" | tee -a logs/training_status_cfg.log
