from openai import OpenAI

# ==============================
# MODEL CONFIGURATION
# ==============================
BASE_URL = "http://localhost:8082/v1"   # modify the port if needed
MODEL_NAME = "Qwen/Qwen3-8B"  # modify the model if needed

# ==============================
# LLM CALL
# ==============================
llm = OpenAI(base_url=BASE_URL, api_key="EMPTY")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize LangGraph in one sentence."}
]

response = llm.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

print(response.choices[0].message.content)
