#!/bin/bash
#SBATCH --job-name=compare_vit
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

# ============================================================
# SLURM job script: Generate comparison plots and PDF report
# Run this AFTER both training jobs complete.
# ============================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Generating comparison plots and PDF report"
echo "Date:   $(date)"
echo "============================================"

mkdir -p logs

source /opt/ohpc/apps/conda/etc/profile.d/conda.sh
conda activate hicom_bw

cd /csehome/r25cs0001/projects/GenAi/Assignment1

# Generate loss curves
echo "Generating comparison plots..."
python3 compare.py --results_dir ./results

# Generate PDF report
echo "Generating PDF report..."
python3 generate_report.py --results_dir ./results

echo ""
echo "All done! Check results/ directory for outputs."
echo "  - results/loss_curves.png"
echo "  - results/accuracy_curves.png"
echo "  - results/report.pdf"
