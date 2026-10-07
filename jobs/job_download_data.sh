#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu-short

echo "Job started: $(date)"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate rag

cd ~/nalla/demo

echo "Using email: $SEC_EMAIL"
python -u scripts/download_data.py --email "$SEC_EMAIL"

echo "Job finished: $(date)"