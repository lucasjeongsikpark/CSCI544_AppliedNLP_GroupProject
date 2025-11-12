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
        self._evaluators = []  # list of (model_name, evaluator)

    def _ensure_loaded(self):
        if self._config is None:
            self._config = _find_config()
        if self._prompt_template is None:
            prompt_path = self._config["experiments"][0]["prompt_path"]
            config_dir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.isabs(prompt_path):
                src_dir = os.path.join(config_dir, "..", "Debatable-Intelligence", "prompts")
                abs_path = os.path.join(src_dir, os.path.basename(prompt_path))
                if os.path.exists(abs_path):
                    prompt_path = abs_path
            with open(prompt_path, "r") as f:
                self._prompt_template = f.read()
        if not self._evaluators:
            base_url = self._config.get("ollama_base_url", "http://localhost:11434")
            model_list = self._config.get("models", {}).get("ollama", [])
            if not model_list:
                raise ValueError("No models specified under models.ollama in config.")
            for m in model_list:
                self._evaluators.append((m, OllamaEvaluator(m, base_url)))
        self._max_tokens = int(self._config.get("max_tokens", 2048))
        self._temperature = float(self._config.get("temperature", 0.01))

    def run(self, data: dict, aspects: list[str]) -> DataOutput:
        self._ensure_loaded()
        max_retries = 3
        # Structure: chat_logs[model_name][answer_key] = {...}
        chat_logs: dict = {}
        per_model_scores_llama = []  # list of dicts per model
        per_model_scores_distill = []
        attempts_tracker = []
        for model_name, evaluator in self._evaluators:
            chat_logs.setdefault(model_name, {})
            model_attempts = 0
            model_scores_llama = {}
            model_scores_distill = {}
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
                    response_text = evaluator.generate(prompt, self._max_tokens, self._temperature)
                    overall, metrics = parse_response(response_text)
                    reasoning = get_reasoning(response_text)
                    if _is_valid_metrics(metrics, aspects):
                        for aspect in aspects:
                            val = metrics.get(aspect.lower())
                            if isinstance(val, int):
                                extracted[aspect.lower()] = val
                        break
                if not extracted and overall != -1:
                    extracted['overall'] = overall
                chat_logs[model_name][answer_key] = {
                    'response': response_text,
                    'reasoning': reasoning,
                    'attempts': answer_attempts,
                    'metrics': extracted
                }
                model_attempts = max(model_attempts, answer_attempts)
                if answer_key == 'llama_output':
                    model_scores_llama = extracted
                else:
                    model_scores_distill = extracted
            per_model_scores_llama.append(model_scores_llama)
            per_model_scores_distill.append(model_scores_distill)
            attempts_tracker.append(model_attempts)
        # Aggregate scores by averaging (ignoring missing aspects) across models
        def aggregate(scores_list):
            agg = {}
            for aspect in aspects:
                vals = [s.get(aspect.lower()) for s in scores_list if isinstance(s.get(aspect.lower()), int)]
                if vals:
                    agg[aspect] = round(sum(vals)/len(vals))
            return agg
        score1 = aggregate(per_model_scores_llama)
        score2 = aggregate(per_model_scores_distill)
        total_attempts = max(attempts_tracker) if attempts_tracker else 0
        return DataOutput(chat_logs=chat_logs, score1=score1, score2=score2, attempts=total_attempts)

# Register the framework instance

debint_framework = DebIntFramework(name="DEBINT")
