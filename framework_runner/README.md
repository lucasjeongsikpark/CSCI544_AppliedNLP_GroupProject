# Framework Runner

A lightweight orchestration layer to evaluate datasets row-by-row using pluggable scoring frameworks (e.g. debate-style evaluation or rubric parsing).

## Key Concepts

- **Framework**: Implements a `run(data: dict, aspects: list[str]) -> DataOutput` method. Encapsulates how a single example is evaluated. Some frameworks (e.g. DEBINT) now support multi-model evaluation per row.
- **DataOutput**: Standard result schema with `chat_logs`, `score1`, `score2`, `attempts`. A `LoggedOutput` wrapper adds `elapsed_time`.
- **Runner**: Streams a dataset (JSON or CSV) and invokes the chosen framework for each row. Outputs results as NDJSON/JSONL allowing incremental progress & resume.

## Features

- Supports `.json` (array of objects) and `.csv` datasets.
- Incremental append-only output (`.jsonl` / `.ndjson`).
- Auto-resume by inspecting existing output or `<output>.progress` file.
- Manual resume via `--start_from`.
- Flexible aspect sourcing: command line list, prompt file parsing (extracts metric names), or dataset-type defaults.
- Pluggable frameworks: current examples are `DEBATE` (`debate_impl.py`), `DEBINT` (`debint_impl.py`), and `SINGLE_AGENT` (`single_agent_impl.py`).
- Remote HuggingFace (CARC) model support via SSH tunnel (see below) alongside local Ollama.

## Directory Structure
```
framework_runner/
  base.py          # DataOutput / LoggedOutput dataclasses + Framework ABC
  runner.py        # FrameworkRunner class (dataset streaming, resume, persistence)
  debate_impl.py   # DebateFramework example (multi-iteration evaluator)
  debint_impl.py   # DebIntFramework (rubric parsing of existing answers)
  main.py          # CLI entry point
  README.md        # This file
```

## Output Schema (Single-Model Frameworks)
```json
{
  "chat_log": {
    "llama_output": {"response": "...", "reasoning": "..."},
    "distill_llama_output": {"response": "...", "reasoning": "..."}
  },
  "score1": {"Correctness": 4, "Reasoning": 5, "Completeness": 4, "Accuracy": 3},
  "score2": {"Correctness": 5, "Reasoning": 3, "Completeness": 2, "Accuracy": 4},
  "attempts": 2,
  "elapsed_time": 27.43
}
```
Notes (single-model):
- `score1` always corresponds to `llama_output`, `score2` to `distill_llama_output`.
- `attempts` counts total per-answer evaluation attempts (for frameworks with retry logic).
- `elapsed_time` is the duration in seconds for that single row evaluation.

## Output Schema (Multi-Model DEBINT)
When multiple models (Ollama and/or remote HF) are listed in the config (e.g.:
```json
"models": {
  "ollama": ["gemma2:2b", "deepseek-r1:1.5b"],
  "huggingface_remote": [
    {"name": "meta-llama/Meta-Llama-3-8B-Instruct", "endpoint": "http://localhost:8000/generate"}
  ]
}
```
DEBINT evaluates every model for both answer fields. Structure per JSONL line (example below shows only Ollama entries for brevity):

```json
{
  "chat_log": {
    "gemma2:2b": {
      "llama_output": {
        "response": "...",
        "reasoning": "...",
        "attempts": 1,
        "metrics": {"correctness": 4, "reasoning": 5, "completeness": 4, "accuracy": 3}
      },
      "distill_llama_output": {
        "response": "...",
        "reasoning": "...",
        "attempts": 1,
        "metrics": {"correctness": 5, "reasoning": 4, "completeness": 3, "accuracy": 4}
      }
    },
    "deepseek-r1:1.5b": {
      "llama_output": {"response": "...", "reasoning": "...", "attempts": 2, "metrics": {"correctness": 3, "reasoning": 5, "completeness": 4, "accuracy": 4}},
      "distill_llama_output": {"response": "...", "reasoning": "...", "attempts": 1, "metrics": {"correctness": 4, "reasoning": 4, "completeness": 3, "accuracy": 5}}
    }
  },
  "score1": {"correctness": 4, "reasoning": 5, "completeness": 4, "accuracy": 4},
  "score2": {"correctness": 5, "reasoning": 4, "completeness": 3, "accuracy": 5},
  "attempts": 2,
  "elapsed_time": 31.9
}
```

Notes (multi-model DEBINT):
- `chat_log` nests per-model dictionaries containing both answer keys.
- Individual per-model metrics remain lowercase; aggregated `score1`/`score2` are aspect-wise averages (rounded) across models for the respective answer field.
- `attempts` is the max attempts used among all model/answer evaluations.
- Use per-model entries in `chat_log` if you need raw, un-averaged scores.
- Remote HF models appear under their full model identifiers.

### Why Aggregation?
Storing aggregated scores in `score1`/`score2` preserves compatibility with downstream consumers expecting a single metrics object per answer field, while still exposing full granularity under `chat_log`.

## Installing Dependencies
Ensure the project dependencies (e.g. pandas) are available:
```bash
pip install -r Debatable-Intelligence/requirements.txt
```
(Or activate your existing environment.)

## Running via CLI
Minimal example (math dataset, debate framework):
```bash
python -m framework_runner.main \
  --framework debate \
  --dataset_path CSCI544_AppliedNLP_GroupProject/datasets/data/math_cleaned_250.json \
  --output_file CSCI544_AppliedNLP_GroupProject/results/debate_math.jsonl \
  --dataset_type math
```

Using explicit aspects override:
```bash
python -m framework_runner.main \
  --framework debint \
  --dataset_path CSCI544_AppliedNLP_GroupProject/datasets/data/math_cleaned_250.json \
  --output_file CSCI544_AppliedNLP_GroupProject/results/debint_math.jsonl \
  --aspects correctness,reasoning,completeness,accuracy
```

Parsing aspects from a prompt file (looks for lines beginning with `- <metric>:` after a `Metrics:` marker):
```bash
python -m framework_runner.main \
  --framework debint \
  --dataset_path CSCI544_AppliedNLP_GroupProject/datasets/data/math_cleaned_250.json \
  --output_file CSCI544_AppliedNLP_GroupProject/results/debint_math.jsonl \
  --aspects_file Debatable-Intelligence/prompts/eval_math_response.txt
```

Resume after interruption (auto):
```bash
python -m framework_runner.main \
  --framework debate \
  --dataset_path CSCI544_AppliedNLP_GroupProject/datasets/data/openQA_cleaned_250.json \
  --output_file CSCI544_AppliedNLP_GroupProject/results/debate_openqa.jsonl
```
The runner will detect existing progress and continue where it left off. To force a manual start index:
```bash
python -m framework_runner.main \
  --framework debate \
  --dataset_path CSCI544_AppliedNLP_GroupProject/datasets/data/openQA_cleaned_250.json \
  --output_file CSCI544_AppliedNLP_GroupProject/results/debate_openqa.jsonl \
  --start_from 120
```
Disable auto resume:
```bash
python -m framework_runner.main ... --no_auto_resume
```

## Adding a New Framework
1. Create a new `XYZFramework(Framework)` class in `framework_runner/xyz_impl.py`.
2. Implement `run(self, data: dict, aspects: list[str]) -> DataOutput`.
3. Return `DataOutput(chat_logs=..., score1=..., score2=..., attempts=...)`.
4. Import and register in `main.py` (add to `FRAMEWORKS`).

## Design Notes
- Runner avoids loading entire outputs into memory (streaming writes).
- Progress tracking uses both `<output>.progress` and output line count fallback.
- Aspect names are normalized to lowercase when parsed; output capitalizes first letter for readability.
- `debint_impl.py` dynamically loads evaluators (Ollama + remote HF) based on the config; if none are provided a clear error is raised.

## Troubleshooting
- Import errors for local modules: ensure you invoke with `python -m framework_runner.main` from project root.
- Remote HF endpoint returns empty: check that it uses one of accepted keys (`completion`, `generated_text`, `text`, `response`).
- Timeouts: verify SSH tunnel and remote server health; consider increasing `hf_remote_default_endpoint` or server timeout.
- Auth failures: supply `hf_remote_auth_header` or per-model `auth_header`.

### Remote HuggingFace (CARC) Usage
1. Start an inference server on the GPU node exposing either:
  - Simple generate endpoint: `/generate` returning `{ "completion": "..." }` (or `generated_text/text/response`).
  - OpenAI-style chat endpoint: `/v1/chat/completions` returning `{ "choices": [{ "message": { "content": "..."}}]}`.
2. Create SSH tunnel locally:
  ```bash
  ssh -L 8000:localhost:8000 <user>@<cluster-host>
  ```
3. Add to config (example simple + chat mix):
  ```json
  "models": {
    "ollama": ["gemma2:2b"],
    "huggingface_remote": [
       {"name": "meta-llama/Meta-Llama-3-8B-Instruct", "endpoint": "http://localhost:8000/generate"},
       {"name": "Qwen/Qwen3-8B", "endpoint": "http://localhost:8082/v1/chat/completions"}
    ]
  },
  "hf_remote_default_endpoint": "http://localhost:8000/generate"
  ```
4. Run: `python -m framework_runner.main --framework debint ...` — aggregated scores include all models (local Ollama + remote chat + remote generate).
5. Optional global auth header:
  ```json
  "hf_remote_auth_header": "Bearer YOUR_TOKEN"
  ```
6. Per-model auth override:
  ```json
  {"name": "mistralai/Mistral-7B-Instruct-v0.2", "auth_header": "Bearer OTHER"}
  ```

- If output lines seem truncated, verify no concurrent process is writing to the same file.
- To re-run from scratch, delete both the output `.jsonl` and `.progress` files.

## Future Enhancements (Ideas)
- Parallel row processing with rate limit controls.
- Unified metrics registry with schema versioning in each JSON line.
- Aggregated summary statistics written after completion.
- Optional compression of large chat logs.

## License
Refer to root project license.
