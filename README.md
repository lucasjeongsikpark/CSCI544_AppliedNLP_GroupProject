# Multi-Agent Debate for LLM Evaluation: A Survey

> **CSCI 544 – Applied Natural Language Processing | University of Southern California**
> A systematic benchmark of four state-of-the-art **Multi-Agent Debate Evaluation Frameworks (MADEFs)** for LLM-as-a-Judge across mathematical reasoning, open-domain dialogue, and medical QA.

📄 **[Read the paper »](reports/CS544_Final.pdf)** · 📊 **[Results »](results/)** · 🧩 **[Framework Runner »](framework_runner/README.md)**

---

## TL;DR

We design, implement, and evaluate a unified pipeline that pits **four MADEFs** — _ChatEval_, _Debatable-Intelligence_, _MORE_, and _DEBATE_ — plus a _single-agent_ baseline, against each other on **three domains** (GSM8K, OpenOrca, MedRedQA) using **fully local open-weight models** (Llama-3.1-8B, DeepSeek-R1-Distill-Llama-8B, Gemma-2-2B) served via **Ollama**. Across **583 paired responses × 5 Likert rubrics × 5 evaluators**, we quantify accuracy, scoring variance, inference cost, and human alignment — and show that **lightweight evaluators often match or beat heavyweight debate stacks** when judge capacity is constrained.

## Key Results

| Dataset                       | Best MADEF (Human Alignment)         | Single-Agent Baseline          | Cost Range (sec / sample) |
| ----------------------------- | ------------------------------------ | ------------------------------ | ------------------------- |
| **GSM8K** (math reasoning)    | Single-Agent ≈ MORE                  | competitive                    | **8.6** → 188.6           |
| **OpenOrca** (open-domain QA) | MORE / ChatEval                      | weaker on multi-part reasoning | 17.1 → 125.6              |
| **MedRedQA** (medical QA)     | MORE / ChatEval (evidence-sensitive) | over-confident                 | 14.5 → **358.4**          |

**Takeaways**

- 🏆 **No single MADEF dominates all domains** — framework choice should follow task structure.
- ⚡ **Debate depth ≠ better evaluation.** DEBATE was up to **40× slower** than single-agent yet most unstable under a small judge model.
- 🧪 **Lightweight wins under constraint.** With a 2B judge, simpler frameworks (single-agent, MORE) frequently outperformed heavier multi-round debates.

Full quantitative tables, qualitative rankings, and ablations are in [the final report](reports/CS544_Final.pdf).

---

## What's in this Repo

```
CSCI544_AppliedNLP_GroupProject/
├── framework_runner/           # ⭐ Unified evaluation harness (ABC-based plugin system)
│   ├── base.py                 #    Framework ABC — drop-in extension point
│   ├── single_agent_impl.py    #    Baseline LLM-as-a-Judge
│   ├── debate_impl.py          #    DEBATE (Devil's Advocate)
│   ├── debint_impl.py          #    Debatable-Intelligence (debate-speech rubric)
│   ├── ollama_eval.py          #    Local Ollama inference adapter
│   └── main.py                 #    CLI entry-point
├── ChatEval/                   # ChatEval reproduction & adaptation (Chan et al., 2023)
├── Debatable-Intelligence/     # Debatable-Intelligence + custom debate runtime
│   └── debate_runtime/         #    Affirmative / Negative / Judge agents
├── Adversarial Multi Agent/    # MORE (Multi-Advocate One-Round Eval)
├── DEBATE/                     # Devil's-Advocate evaluator
├── datasets/                   # GSM8K · OpenOrca · MedRedQA (cleaned, paired)
├── prompts/                    # Domain-specific Likert rubrics
├── results/                    # JSON / JSONL outputs + analysis notebooks
├── CARC/                       # USC HPC client/server for batched inference
└── reports/
    ├── CS544_Final.pdf         # 📄 Final paper
    ├── project_status_report.pdf
    └── proposal.pdf
```

---

## Technical Highlights

**🧠 Models & Serving.** Fully local stack — no closed-API dependencies. Response generation with **Meta-Llama-3.1-8B-Instruct** and **DeepSeek-R1-Distill-Llama-8B**; judging with **Gemma-2-2B**. All inference orchestrated through **Ollama**, with a thin client for the `framework_runner` harness and a separate **CARC HPC client/server** layer for batched runs on USC's compute cluster.

**🧩 Pluggable Framework Harness.** A single `Framework` abstract base class (`framework_runner/base.py`) exposes a uniform `evaluate(sample) → score` contract. Each MADEF is a self-contained subclass, so new evaluators (e.g., G-Eval, GPTScore, custom) drop in with one file. Identical CLI, identical I/O schema, identical metrics — eliminating methodological confounds across frameworks.

**📐 Standardized Likert Rubrics.** Domain-tailored 1–5 scales — GSM8K (correctness · reasoning · completeness · accuracy), OpenOrca (relevance · completeness · accuracy · clarity · helpfulness), MedRedQA (medical accuracy · appropriateness · safety · clarity · professionalism).

**📈 Multi-axis Evaluation.**

- Mean & std deviation per (framework × dataset × rubric × generator)
- Per-sample inference latency
- **Human-alignment counts** vs. expert annotations
- Qualitative behavioral analysis on representative samples per domain

**🔁 Reproducibility.** Deterministic configs, fixed prompt templates, paired generation, and JSONL outputs. Every plot in the paper traces back to a script in `results/result_analysis/`.

---

## Contributors & Modules

| Member            | Module                                                                                                                               | Paper                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| **Jeongsik Park** | [`ChatEval/`](ChatEval/README.md) — multi-agent debate w/ role calibration                                                           | [Chan et al., 2023](https://arxiv.org/abs/2308.07201)         |
| **Ketan Joshi**   | [`Debatable-Intelligence/`](Debatable-Intelligence/debate_runtime/README_DEBATE_RUNTIME.md) — Affirmative / Negative / Judge runtime | [Sternlicht et al., 2025](https://arxiv.org/abs/2506.05062)   |
| **Dengheng Shi**  | [`Adversarial Multi Agent/`](Adversarial%20Multi%20Agent/) — MORE (from-scratch impl.)                                               | [Bandi & Harrasse, 2024](https://arxiv.org/html/2410.04663v2) |
| **Brendan Hy**    | [`DEBATE/`](DEBATE/README.md) — Devil's-Advocate evaluator (from-scratch impl.)                                                      | [Kim et al., 2024](https://arxiv.org/abs/2405.09935)          |
| **All**           | [`framework_runner/`](framework_runner/README.md) · [`prompts/`](prompts/) · [`reports/CS544_Final.pdf`](reports/CS544_Final.pdf)    | —                                                             |

---

## Quick Start

```bash
git clone https://github.com/lucasjeongsikpark/CSCI544_AppliedNLP_GroupProject.git
cd CSCI544_AppliedNLP_GroupProject
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r Debatable-Intelligence/requirements.txt     # macOS arm64: requirements_arm64.txt
```

Install [**Ollama**](https://ollama.com) and pull the judge model:

```bash
ollama pull gemma2:2b
```

Run an evaluation across any of the four MADEFs or the single-agent baseline:

```bash
python3 -m framework_runner.main \
  --framework {debint|debate|more|chateval|single} \
  --dataset_path datasets/data/math_cleaned_250.json \
  --dataset_type {math|medical|openqa} \
  --output_file results/my_run.jsonl
```

📚 **Detailed guides**

- Framework Runner architecture & flags → [framework_runner/README.md](framework_runner/README.md)
- Single-agent evaluator → [framework_runner/SINGLE_AGENT_README.md](framework_runner/SINGLE_AGENT_README.md)
- Debate runtime usage & adding a new domain → [Debatable-Intelligence/debate_runtime/README_DEBATE_RUNTIME.md](Debatable-Intelligence/debate_runtime/README_DEBATE_RUNTIME.md)
- HPC batching → [CARC/ReadMe.md](CARC/ReadMe.md)

---

## Extending the Harness

Add a new evaluator in three steps:

```python
# framework_runner/my_eval_impl.py
from .base import Framework

class MyEvaluator(Framework):
    def evaluate(self, sample: dict) -> dict:
        # ... your debate / judge logic ...
        return {"scores": {...}, "rationale": "..."}
```

Register it in `runner.py`, and it inherits the full CLI, dataset loaders, Ollama client, and JSONL output schema — no glue code required.

---

## Citation

```bibtex
@misc{hy2025madef,
  title  = {Multi-Agent Debate for LLM Evaluation: A Survey},
  author = {Hy, Brendan and Joshi, Ketan and Patel, Pranshu Prakash and
            Park, Jeongsik and Sanghvi, Manav and Shi, Dengheng},
  year   = {2025},
  note   = {CSCI 544, University of Southern California}
}
```

## License

Released for academic and research purposes. Underlying frameworks retain their original licenses (see each subdirectory).
