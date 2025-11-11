def _is_valid_metrics(metrics: dict, aspects: list[str]) -> bool:
    if not isinstance(metrics, dict):
        return False
    for aspect in aspects:
        val = metrics.get(aspect.lower())
        if not (isinstance(val, int) and 1 <= val <= 5):
            return False
    return True
import os
import json
from .base import Framework, DataOutput
from .ollama_eval import OllamaEvaluator

# Inline helpers (extracted from dataset_experiment.py) to avoid heavy imports.
def parse_response(response: str):
    metrics = {}
    text = response or ''
    if '<metrics>' in text and '</metrics>' in text:
        mstart = text.find('<metrics>') + len('<metrics>')
        mend = text.find('</metrics>')
        block = text[mstart:mend].strip()
        for line in block.splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip().lower()
            v = v.strip().replace('[', '').replace(']', '')
            if v.isdigit():
                iv = int(v)
                if iv in (1,2,3,4,5):
                    metrics[k] = iv
    overall = -1
    if '<overall_score>' in text and '</overall_score>' in text:
        s = text.find('<overall_score>') + len('<overall_score>')
        e = text.find('</overall_score>')
        raw = text[s:e].strip().replace('[','').replace(']','')
        if raw.isdigit():
            iv = int(raw)
            if iv in (1,2,3,4,5):
                overall = iv
    elif '<score>' in text and '</score>' in text:
        s = text.find('<score>') + len('<score>')
        e = text.find('</score>')
        raw = text[s:e].strip().replace('[','').replace(']','')
        if raw.isdigit():
            iv = int(raw)
            if iv in (1,2,3,4,5):
                overall = iv
    return overall, metrics

def get_reasoning(response: str) -> str:
    if not response:
        return ''
    reasoning_start = response.find('<scratchpad>') + len('<scratchpad>')
    reasoning_end = response.find('</scratchpad>')
    if reasoning_start != -1 and reasoning_end != -1:
        return response[reasoning_start:reasoning_end].strip()
    return ''


# Helper to load config with CLI/file/dataset_type logic
_CONFIG_CLI_FILE = None
_CONFIG_CLI_DATASET_TYPE = None
def set_config_cli_args(config_file=None, dataset_type=None):
    global _CONFIG_CLI_FILE, _CONFIG_CLI_DATASET_TYPE
    _CONFIG_CLI_FILE = config_file or ''
    _CONFIG_CLI_DATASET_TYPE = dataset_type or ''

def _find_config():
    # 1. If CLI --config_file is set, use it
    if _CONFIG_CLI_FILE:
        if os.path.exists(_CONFIG_CLI_FILE):
            with open(_CONFIG_CLI_FILE) as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file specified but not found: {_CONFIG_CLI_FILE}")
    # 2. If dataset_type is set, use config_<dataset_type>.json if present
    if _CONFIG_CLI_DATASET_TYPE:
        fname = f"config_{_CONFIG_CLI_DATASET_TYPE.lower()}.json"
        for d in [os.getcwd(), os.path.dirname(os.getcwd())]:
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    return json.load(f)
    # 3. Fallback to config.json
    for d in [os.getcwd(), os.path.dirname(os.getcwd())]:
        fpath = os.path.join(d, "config.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                return json.load(f)
    raise FileNotFoundError("No config file found. Searched CLI, dataset_type, and config.json in cwd or parent.")

class DebIntFramework(Framework):
    @staticmethod
    def set_config_cli_args(config_file=None, dataset_type=None):
        set_config_cli_args(config_file, dataset_type)
    def __init__(self, name="DEBINT"):
        super().__init__(name)
        self._config = None
        self._prompt_template = None
        self._evaluator = None

    def _ensure_loaded(self):
        if self._config is None:
            self._config = _find_config()
        if self._prompt_template is None:
            prompt_path = self._config["experiments"][0]["prompt_path"]
            # Try relative to config file, then cwd
            config_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(prompt_path):
                # Try relative to Debatable-Intelligence/src/
                src_dir = os.path.join(config_dir, "..", "Debatable-Intelligence", "prompts")
                abs_path = os.path.join(src_dir, os.path.basename(prompt_path))
                if os.path.exists(abs_path):
                    prompt_path = abs_path
            with open(prompt_path, "r") as f:
                self._prompt_template = f.read()
        if self._evaluator is None:
            model_name = self._config["models"]["ollama"][0]
            base_url = self._config.get("ollama_base_url", "http://localhost:11434")
            self._evaluator = OllamaEvaluator(model_name, base_url)
        self._max_tokens = int(self._config.get("max_tokens", 2048))
        self._temperature = float(self._config.get("temperature", 0.01))

    def run(self, data: dict, aspects: list[str]) -> DataOutput:
        self._ensure_loaded()
        chat_logs: dict = {}
        score1: dict[str, int] = {}
        score2: dict[str, int] = {}
        attempts_llama = 0
        attempts_distill = 0
        max_retries = 3
        for answer_key in ["llama_output", "distill_llama_output"]:
            answer_attempts = 0
            extracted = {}
            response_text = ""
            reasoning = ""
            for attempt in range(1, max_retries + 1):
                answer_attempts = attempt
                prompt = self._prompt_template
                prompt = prompt.replace("{INPUT}", str(data.get("input", "")))
                prompt = prompt.replace("{OUTPUT}", str(data.get("output", "")))
                prompt = prompt.replace("{RESPONSE}", str(data.get(answer_key, "")))
                response_text = self._evaluator.generate(prompt, self._max_tokens, self._temperature)
                # DEBUG: Print the raw <metrics> block for diagnosis
                if '<metrics>' in response_text and '</metrics>' in response_text:
                    mstart = response_text.find('<metrics>') + len('<metrics>')
                    mend = response_text.find('</metrics>')
                    block = response_text[mstart:mend].strip()
                    print(f"[DEBUG] Raw <metrics> block for {answer_key}:\n{block}\n---")
                overall, metrics = parse_response(response_text)
                reasoning = get_reasoning(response_text)
                if _is_valid_metrics(metrics, aspects):
                    for aspect in aspects:
                        val = metrics.get(aspect.lower())
                        if isinstance(val, int):
                            extracted[aspect.capitalize()] = val
                    break  # Valid, stop retrying
                # else: retry
            if not extracted and overall != -1:
                extracted['Overall'] = overall
            chat_logs.setdefault(answer_key, {})
            chat_logs[answer_key] = {
                'response': response_text,
                'reasoning': reasoning
            }
            if answer_key == 'llama_output':
                score1 = extracted
                attempts_llama = answer_attempts
            else:
                score2 = extracted
                attempts_distill = answer_attempts
        total_attempts = max(attempts_llama, attempts_distill)
        return DataOutput(chat_logs=chat_logs, score1=score1, score2=score2, attempts=total_attempts)

# Register the framework instance

debint_framework = DebIntFramework(name="DEBINT")
