#!/bin/bash
#SBATCH --job-name=eval_retrieval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

echo "Job started: $(date)"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate rag

cd ~/nalla/demo

python -u scripts/eval_retrieval.py "$@"

echo "Job finished: $(date)"