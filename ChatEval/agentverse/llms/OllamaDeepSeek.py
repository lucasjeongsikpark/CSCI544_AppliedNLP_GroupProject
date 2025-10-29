import os
import requests
from typing import List
from pydantic import BaseModel, Field
from agentverse.llms.base import LLMResult, BaseChatModel, BaseModelArgs
from agentverse.message import Message
from agentverse.llms import llm_registry

class OllamaDeepSeekAPIArgs(BaseModelArgs):
    base_url: str = Field(default=os.environ.get("LOCAL_API_BASE_URL", "http://localhost:11434/api"))
    model: str = Field(default=os.environ.get("LOCAL_API_MODEL", "deepseek-r1:1.5b"))
    stream: bool = Field(default=False)   

@llm_registry.register("OllamaDeepSeek")
class OllamaDeepSeekAPIChat(BaseChatModel):
    args: OllamaDeepSeekAPIArgs = Field(default_factory=OllamaDeepSeekAPIArgs)

    def _construct_messages(self, prompt: str, chat_memory: List[Message], final_prompt: str):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for item in chat_memory:
            role = "assistant" if item.sender == "assistant" else "user"
            messages.append({"role": role, "content": item.content})
        messages.append({"role": "user", "content": prompt})
        if final_prompt:
            messages.append({"role": "user", "content": final_prompt})
        return messages

    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        msgs = self._construct_messages(prompt, chat_memory, final_prompt)
        payload = {
            "model": self.args.model,
            "stream": self.args.stream,
            "messages": msgs,
        }

        resp = requests.post(f"{self.args.base_url}/chat", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error: {resp.text}")

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        return self.generate_response(prompt, chat_memory, final_prompt)
