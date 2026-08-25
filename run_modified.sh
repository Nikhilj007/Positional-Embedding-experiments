#!/bin/bash
#SBATCH --job-name=modified_vit
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/modified_vit_%j.out
#SBATCH --error=logs/modified_vit_%j.err

# ============================================================
# SLURM job script: Train Modified ViT (2D RoPE) on CIFAR-100
# ============================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "============================================"

# Create log directory
mkdir -p logs

# Activate conda environment
source /opt/ohpc/apps/conda/etc/profile.d/conda.sh
conda activate hicom_bw

# Navigate to project directory
cd /csehome/r25cs0001/projects/GenAi/Assignment1

# Print environment info
echo ""
echo "Python: $(python3 --version)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
echo ""

# Create data and results directories
mkdir -p data results

# Run training
echo "Starting Modified ViT (2D RoPE) training..."
python3 modified_vit/train.py \
    --epochs 200 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --dropout 0.1 \
    --seed 42 \
    --data_dir ./data \
    --save_dir ./results \
    --num_workers 4

echo ""
echo "Modified ViT training completed at $(date)"
