# Evaluator LLM integration for debint_impl.py
# This module provides a utility to call Ollama for evaluation, using the same logic as in Debatable-Intelligence/src/model.py

import requests
import time

class OllamaEvaluator:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def _simple_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max(1, int(max_tokens)),
                "temperature": float(temperature),
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            print(f"Error querying Ollama model '{self.model_name}': {e}")
            raise

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        retries = 2
        backoff = 5
        last_err = None
        for _ in range(retries + 1):
            try:
                response_text = self._simple_generate(prompt, max_tokens, temperature)
                return response_text
            except Exception as e:
                last_err = e
                time.sleep(backoff)
        raise last_err
