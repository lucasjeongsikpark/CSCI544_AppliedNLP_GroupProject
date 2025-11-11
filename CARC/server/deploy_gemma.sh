#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
if [ ! -d ~/env_llama3 ]; then
  python -m venv ~/env_llama3
  source ~/env_llama3/bin/activate
  pip install --upgrade pip
  pip install torch transformers fastapi uvicorn accelerate sentencepiece pydantic
else
  source ~/env_llama3/bin/activate
fi

# ==============================
# Python App
# ==============================
python <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "google/gemma-2-2b"
cache_dir = os.getenv("HF_HOME")
token = os.getenv("HF_TOKEN")

print(f"Downloading {model_id} into cache: {cache_dir}")
tok = AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    cache_dir=cache_dir,
    torch_dtype="auto",
    device_map="auto"
)
print("✅ Download complete and cached.")
EOF
