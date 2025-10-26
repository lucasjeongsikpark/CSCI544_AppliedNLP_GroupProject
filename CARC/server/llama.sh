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

python <<'EOF'
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# ==============================
# Load Model & Tokenizer
# ==============================
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"🔹 Loading {model_id} ...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ==============================
# FastAPI Server
# ==============================
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
    """
    OpenAI-compatible chat completion endpoint.
    """

    # ✅ Convert to chat template format
    input_ids = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in req.messages],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # ✅ Generate model output
    outputs = model.generate(
        input_ids,
        max_new_tokens=req.max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    # ✅ Extract only new tokens (assistant response)
    response_tokens = outputs[0][input_ids.shape[-1]:]
    assistant_reply = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # ✅ Return in OpenAI-style JSON
    return JSONResponse({
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant_reply}
        }]
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF
