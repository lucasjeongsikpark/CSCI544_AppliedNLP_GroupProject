# agentverse/llms/local_api.py
import os
import requests
from typing import List
from pydantic import BaseModel, Field
from agentverse.llms.base import LLMResult, BaseChatModel, BaseModelArgs
from agentverse.message import Message
from agentverse.llms import llm_registry

class QwenAPIArgs(BaseModelArgs):
    base_url: str = Field(default=os.environ.get("LOCAL_API_BASE_URL", "http://localhost:8082/v1"))
    model: str = Field(default=os.environ.get("LOCAL_API_MODEL", "Qwen/Qwen3-8B"))
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)

@llm_registry.register("Qwen")
class QwenAPIChat(BaseChatModel):
    args: QwenAPIArgs = Field(default_factory=QwenAPIArgs)

    def _construct_messages(self, prompt: str, chat_memory: List[Message], final_prompt: str):
        messages = [{"role": "user", "content": prompt}]
        for item in chat_memory:
            messages.append({"role": "assistant", "content": f"{item.sender}: {item.content}"})
        messages.append({"role": "user", "content": final_prompt})
        return messages

    def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        msgs = self._construct_messages(prompt, chat_memory, final_prompt)
        payload = {
            "model": self.args.model,
            "messages": msgs,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        resp = requests.post(f"{self.args.base_url}/chat/completions", json=payload)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)

    async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
        """비동기 환경에서도 동기 메서드를 재사용하도록 간단히 구현."""
        return self.generate_response(prompt, chat_memory, final_prompt)
