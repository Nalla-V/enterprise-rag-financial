#!/bin/bash
#SBATCH --job-name=fcheck_fact
#SBATCH --output=logs/check_%j.out
#SBATCH --error=logs/check_%j.err
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

python -u scripts/check_facts.py

echo "Job finished: $(date)"