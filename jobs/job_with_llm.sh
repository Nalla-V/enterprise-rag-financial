#!/bin/bash
# Runs any Python script WITH a local LLM server available on the same GPU node.
#   1. starts Qwen 2.5-7B as an OpenAI-compatible server (vLLM, 'llm' env) in the background
#   2. waits until it answers
#   3. runs your script in the 'rag' env, pointed at http://localhost:8000/v1
#   4. stops the server
# Usage:  sbatch jobs/job_with_llm.sh scripts/test_llm.py
#         sbatch jobs/job_with_llm.sh scripts/run_agent.py --question "..."
#SBATCH --job-name=with_llm
#SBATCH --output=logs/with_llm_%j.out
#SBATCH --error=logs/with_llm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

echo "Job started: $(date)"
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh

cd ~/nalla/demo
export HF_HOME=${HF_HOME:-$HOME/nalla/hf_cache}
export VLLM_USE_FLASHINFER_SAMPLER=0          # issue 18: no nvcc on the compute nodes
MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_ENV=$HOME/.conda/envs/llm

# ---- 1. LLM server. 0.55 of the GPU: leaves room for bge-m3 + reranker in the same job
$LLM_ENV/bin/vllm serve $MODEL --port 8000 --max-model-len 16384 \
    --gpu-memory-utilization 0.55 > logs/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# ---- 2. wait until it answers (max 15 min)
for i in $(seq 1 90); do
    if curl -s localhost:8000/v1/models > /dev/null; then echo "LLM server ready after $((i*10))s"; break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died - see logs/vllm_$SLURM_JOB_ID.log"; tail -30 logs/vllm_$SLURM_JOB_ID.log; exit 1; fi
    sleep 10
done

# ---- 3. your script, in the rag env
conda activate base
conda activate rag
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=EMPTY LLM_MODEL=$MODEL
python -u "$@"

echo "Job finished: $(date)"