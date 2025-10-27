#!/bin/bash

echo "=== DDPM Training Status Check ==="
echo "Time: $(date)"
echo

# Check if any DDPM jobs are running
echo "--- SLURM Job Status ---"
squeue -u $USER --name=ddmp_train || echo "No DDPM training jobs found in queue"
echo

# Run the monitoring script if it exists
if [ -f "logs/monitor_progress.py" ]; then
    python logs/monitor_progress.py
else
    echo "Monitor script not found. Training may not have started yet."
fi

# Quick checkpoint check
echo "--- Quick Checkpoint Check ---"
if ls checkpoint/*.pth 1> /dev/null 2>&1; then
    ls -lth checkpoint/*.pth | head -3
else
    echo "No checkpoints found yet"
fi
echo

echo "--- GPU Usage (if job is running) ---"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv 2>/dev/null || echo "No GPU access or job not running"
echo

echo "=== Commands for More Details ==="
echo "Monitor live: tail -f logs/ddpm_train_<JOB_ID>.out"
echo "Detailed logs: tail -f logs/training_detailed.log"
echo "Job efficiency: seff <JOB_ID>"
echo "Job details: scontrol show jobid <JOB_ID>"