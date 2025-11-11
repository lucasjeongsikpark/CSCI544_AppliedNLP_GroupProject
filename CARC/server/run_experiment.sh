#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=a100

mkdir -p ./logs

# ==============================
# Parse Input Arguments
# ==============================
JOB_NAME=${1:-gemma-api}
MODEL_ID=${2:-google/gemma-2-2b}
PORT=${3:-8083}

echo "🔹 Job Name: $JOB_NAME"
echo "🔹 Model ID: $MODEL_ID"
echo "🔹 Port: $PORT"

module purge
module load python/3.10

# ==============================
# Virtual Environment Setup
# ==============================
if [ ! -d ~/env_runner ]; then
  python -m venv ~/env_runner
  source ~/env_runner/bin/activate
  pip install --upgrade pip
  pip install torch transformers pydantic ollama pandas
else
  source ~/env_runner/bin/activate
fi

# ==============================
# Python App
# ==============================
echo "Launching Experiment"
python -m framework_runner.main
echo "Woohoo"
