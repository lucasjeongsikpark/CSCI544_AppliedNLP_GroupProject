#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=v100|a100|a40|l40s

mkdir -p ./logs

# ==============================
# Parse Input Arguments
# ==============================
JOB_NAME=${1:-mistral_api}
MODEL_ID=${2:-mistralai/Mistral-7B-Instruct-v0.1}
PORT=${3:-8083}

echo "🔹 Job Name: $JOB_NAME"
echo "🔹 User Email: $USER_EMAIL"
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
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

model_id = "${MODEL_ID}"
print(f"🔹 Loading {model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
app = FastAPI()
class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = 2048
    temperature: float | None = 0.7
    top_p: float | None = 0.9
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    input_ids = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in req.messages],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=req.max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    response_tokens = outputs[0][input_ids.shape[-1]:]
    assistant_reply = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    return JSONResponse({
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant_reply}
        }]
    })
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=${PORT})
EOF