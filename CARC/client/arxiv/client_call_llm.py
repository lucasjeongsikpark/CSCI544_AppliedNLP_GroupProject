from openai import OpenAI

llm = OpenAI(base_url="http://localhost:8083/v1", api_key="EMPTY") #modify the port if needed

resp = llm.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1", #modify the model name if needed
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize LangGraph in one sentence."}
    ],
)
print(resp.choices[0].message.content)

