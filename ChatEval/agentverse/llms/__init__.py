from agentverse.registry import Registry

llm_registry = Registry(name="LLMRegistry")

from .base import BaseLLM, BaseChatModel, BaseCompletionModel, LLMResult
from .openai import OpenAIChat, OpenAICompletion
from .local_api import LocalAPIChat  # local_api.py의 클래스를 임포트하여 등록
from .Mistral import MistralAPIChat
from .Qwen import QwenAPIChat 