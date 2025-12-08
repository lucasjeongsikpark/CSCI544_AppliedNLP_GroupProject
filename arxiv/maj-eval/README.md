# MAJ-EVAL Ready (End-to-End)

This repo is a runnable **end-to-end scaffold** that satisfies the project checklist:
- **Datasets**: DiQAD, BioASQ, MMATH (uses the samples provided in `data/`).
- **Evaluatees**: 2 sub-10B placeholders (configurable).
- **Evaluators**: 2–3 sub-10B placeholders (configurable).
- **Metrics**: Accuracy / Macro-F1 for DiQAD; Exact Match for BioASQ & MMATH.
- **Variation**: Persona-based evaluators (StrictReferee, SafetyFocused, PedanticScholar).
- **Discussion**: Logs with per-sample rationale and turns are produced.
- **Outputs**: Predictions, judgments, metrics, and a markdown report.

> By default, the pipeline runs **offline** using deterministic heuristics for predictions and judgments so it works without GPUs or internet. If you enable `runtime.use_hf: true` and provide Hugging Face models, it will use real LLMs.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run everything (offline deterministic mode)
python -m src.cli all
```

Artifacts:
- `results/predictions/*.jsonl`
- `results/evaluations/*.jsonl`
- `results/logs/*.jsonl`
- `results/metrics/aggregate_results.json`
- `results/reports/report.md`

## Config

See `config/run_config.yaml` to:
- Change evaluatee/evaluator model names
- Toggle personas / debate aggregation


