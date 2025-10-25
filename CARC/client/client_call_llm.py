from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize LangGraph in one sentence."}
    ],
)
print(resp.choices[0].message.content)

