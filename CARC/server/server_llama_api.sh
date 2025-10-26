
#!/bin/bash
#SBATCH --job-name=llama3_api
#SBATCH --output=logs/llama3_api_%j.out
#SBATCH --error=logs/llama3_api_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jeongsik@usc.edu
module purge
module load python/3.10
# --- Virtual env ---
if [ ! -d ~/env_llama3 ]; then
  python -m venv ~/env_llama3
  source ~/env_llama3/bin/activate
  pip install --upgrade pip
  pip install torch transformers fastapi uvicorn accelerate sentencepiece pydantic
else
  source ~/env_llama3/bin/activate
fi
# --- Run FastAPI server ---
python <<'EOF'
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import JSONResponse
import uvicorn, os
HF_TOKEN = os.getenv("HF_TOKEN", None)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
print(f"🔹 Loading {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=HF_TOKEN
)
app = FastAPI()

class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = 256
    temperature: float | None = 0.7
@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = ""
    for m in req.messages:
        if m.role == "system":
            prompt += f"[System]: {m.content}\n"
        elif m.role == "user":
            prompt += f"[User]: {m.content}\n"
        elif m.role == "assistant":
            prompt += f"[Assistant]: {m.content}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse({
        "object": "chat.completion",
        "choices": [{"index":0,"message":{"role":"assistant","content":text}}]
    })
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
EOF