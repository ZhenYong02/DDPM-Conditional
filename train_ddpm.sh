#!/bin/bash

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --job-name=ddmp_train
#SBATCH --output=logs/ddpm_train_%j.out
#SBATCH --error=logs/ddpm_train_%j.err
#SBATCH --time=06:00:00

# Create logs and checkpoint directories
mkdir -p logs checkpoint

# Use absolute python path to bypass conda activation issues
PYTHON_PATH="/home/msai/zhenyong001/.conda/envs/nodeenv/bin/python"

# Verify environment and test imports
echo "Python path: $PYTHON_PATH" | tee -a logs/training_status.log
$PYTHON_PATH -c "import torch, torchvision; print('PyTorch and torchvision imported successfully')" | tee -a logs/training_status.log

# Print job information to both stdout and a status file
echo "=== DDPM Training Job Started ===" | tee logs/training_status.log
echo "Job ID: $SLURM_JOB_ID" | tee -a logs/training_status.log
echo "Node: $SLURMD_NODENAME" | tee -a logs/training_status.log
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a logs/training_status.log
echo "Start time: $(date)" | tee -a logs/training_status.log
echo "Working directory: $(pwd)" | tee -a logs/training_status.log
echo "================================" | tee -a logs/training_status.log

# Check GPU availability
nvidia-smi | tee -a logs/training_status.log

# Print Python and PyTorch versions
$PYTHON_PATH -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" | tee -a logs/training_status.log

# Create a progress monitoring script
cat > logs/monitor_progress.py << 'EOF'
#!/usr/bin/env python3
import os
import glob
from datetime import datetime

def monitor_training():
    print(f"=== DDPM Training Progress Monitor ===")
    print(f"Check time: {datetime.now()}")

    # Check if training is running
    if os.path.exists("logs/training_status.log"):
        print("\n--- Job Status ---")
        with open("logs/training_status.log", "r") as f:
            print(f.read())

    # Check for checkpoints
    checkpoints = glob.glob("checkpoint/*.pth")
    if checkpoints:
        print(f"\n--- Checkpoints Found ---")
        for cp in sorted(checkpoints):
            stat = os.stat(cp)
            print(f"{cp} - Modified: {datetime.fromtimestamp(stat.st_mtime)}")
    else:
        print("\n--- No checkpoints found yet ---")

    # Check recent log output
    log_files = glob.glob("logs/ddpm_train_*.out")
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

chmod +x logs/monitor_progress.py

# Start training with progress logging
echo "Starting DDMP training..." | tee -a logs/training_status.log
echo "Training started at: $(date)" | tee -a logs/training_status.log

# Run training and capture both stdout and stderr
$PYTHON_PATH train.py 2>&1 | tee -a logs/training_detailed.log

# Print completion info
echo "Training completed at: $(date)" | tee -a logs/training_status.log
echo "Job finished successfully!" | tee -a logs/training_status.log