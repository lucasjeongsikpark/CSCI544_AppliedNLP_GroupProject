import re
import time
from .base import Framework, DataOutput
from .ollama_eval import OllamaEvaluator

class SingleAgentFramework(Framework):
    def __init__(self, name, model_name, prompt_template_path, max_tokens=512, temperature=0.2, base_url="http://localhost:11434"):
        super().__init__(name)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self.prompt_template = self._load_prompt(prompt_template_path)
        self.evaluator = OllamaEvaluator(model_name, base_url)

    def _load_prompt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _fill_prompt(self, data, response_field):
        return self.prompt_template.format(
            DOCUMENT=data.get("document", ""),
            INPUT=data.get("input", ""),
            OUTPUT=data.get("output", ""),
            RESPONSE=data.get(response_field, "")
        )

    def _parse_metrics(self, text, aspects):
        # Extract metrics block
        metrics = {k: None for k in aspects}
        overall_score = None
        scratchpad = ""
        m = re.search(r"<metrics>(.*?)</metrics>", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                for aspect in aspects:
                    if aspect in line.lower():
                        val = re.findall(r"(\d)", line)
                        if val:
                            metrics[aspect] = int(val[0])
        m2 = re.search(r"<overall_score>\s*(\d)\s*</overall_score>", text)
        if m2:
            overall_score = int(m2.group(1))
        m3 = re.search(r"<scratchpad>(.*?)</scratchpad>", text, re.DOTALL)
        if m3:
            scratchpad = m3.group(1).strip()
        return metrics, overall_score, scratchpad

    def run(self, data: dict, aspects: list[str]) -> DataOutput:
        results = {}
        scores = {}
        attempt_counts = {}
        # Always evaluate in this order for consistency
        for field in ["llama_output", "distill_llama_output"]:
            prompt = self._fill_prompt(data, field)
            retries = 0
            max_retries = 3
            while True:
                start = time.time()
                response = self.evaluator.generate(prompt, self.max_tokens, self.temperature)
                elapsed = time.time() - start
                metrics, overall_score, scratchpad = self._parse_metrics(response, aspects)
                retries += 1
                # Check for perfectly structured output: all metrics present, overall_score present, scratchpad present
                metrics_ok = all(v is not None for v in metrics.values())
                overall_ok = overall_score is not None
                scratchpad_ok = bool(scratchpad)
                if metrics_ok and overall_ok and scratchpad_ok:
                    break
                if retries >= max_retries:
                    break
            results[field] = {
                "llm_response": response,
                "overall_score": overall_score,
                "scratchpad": scratchpad,
                "elapsed_time": elapsed
            }
            scores[field] = metrics
            attempt_counts[field] = retries
        # Ensure score1 is always llama_output and score2 is always distill_llama_output
        score1 = scores.get("llama_output", {})
        score2 = scores.get("distill_llama_output", {})
        attempts = max(attempt_counts.values()) if attempt_counts else 0
        return DataOutput(chat_logs=results, score1=score1, score2=score2, attempts=attempts)

# For dynamic instantiation from main.py
