# Framework Runner

A lightweight orchestration layer to evaluate datasets row-by-row using pluggable scoring frameworks (e.g. debate-style evaluation or rubric parsing).

## Key Concepts

- **Framework**: Implements a `run(data: dict, aspects: list[str]) -> DataOutput` method. Encapsulates how a single example is evaluated.
- **DataOutput**: Standard result schema with `chat_logs`, `score1`, `score2`, `attempts`. A `LoggedOutput` wrapper adds `elapsed_time`.
- **Runner**: Streams a dataset (JSON or CSV) and invokes the chosen framework for each row. Outputs results as NDJSON/JSONL allowing incremental progress & resume.

## Features

- Supports `.json` (array of objects) and `.csv` datasets.
- Incremental append-only output (`.jsonl` / `.ndjson`).
- Auto-resume by inspecting existing output or `<output>.progress` file.
- Manual resume via `--start_from`.
- Flexible aspect sourcing: command line list, prompt file parsing (extracts metric names), or dataset-type defaults.
- Pluggable frameworks: current examples are `DEBATE` (`debate_impl.py`) and `DEBINT` (`debint_impl.py`).

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

## Output Schema (Each JSONL line)
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
Notes:
- `score1` always corresponds to `llama_output`, `score2` to `distill_llama_output`.
- `attempts` counts total per-answer evaluation attempts (for frameworks with retry logic).
- `elapsed_time` is the duration in seconds for that single row evaluation.

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
- `debint_impl.py` dynamically imports parsing helpers, isolating scoring from generation logic.

## Troubleshooting
- Import errors for local modules: ensure you invoke with `python -m framework_runner.main` from project root.
- If output lines seem truncated, verify no concurrent process is writing to the same file.
- To re-run from scratch, delete both the output `.jsonl` and `.progress` files.

## Future Enhancements (Ideas)
- Parallel row processing with rate limit controls.
- Unified metrics registry with schema versioning in each JSON line.
- Aggregated summary statistics written after completion.
- Optional compression of large chat logs.

## License
Refer to root project license.
