#!/bin/bash
#SBATCH -p aichem
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -J llama-int4
#SBATCH -o /mnt/tank/scratch/igolovkin/triton_quantization/logs/%j.out

set -euo pipefail
cd /mnt/tank/scratch/igolovkin/triton_quantization
mkdir -p logs results
export HF_HOME=/mnt/tank/scratch/igolovkin/triton_quantization/.cache/huggingface
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/tank/scratch/igolovkin/triton_quantization:/mnt/tank/scratch/igolovkin/triton_quantization/.deps:/mnt/tank/scratch/igolovkin/.venv/lib/python3.10/site-packages"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
/usr/bin/python3.10 scripts/run_llama_eval.py
