# CSCI544 – Applied NLP (Group Project)

This repository contains all group deliverables, experiment results, and references for **CSCI 544: Applied Natural Language Processing** at USC.

---

### **Jeongsik Park**

#### Project: ChatEval

- **Paper:** [ChatEval: Towards Better Evaluation for Multi-turn Dialogue Models](https://arxiv.org/abs/2308.07201)
- **Codebase:** [thunlp/ChatEval (GitHub)](https://github.com/thunlp/ChatEval)

### **Ketan Joshi**

#### Project: Debateable-Intelligence

- **Paper:** [Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation
](https://arxiv.org/abs/2506.05062)
- **Codebase:** [Debatable-Intelligence](https://github.com/noy-sternlicht/Debatable-Intelligence)

### **Dengheng Shi**

#### Project: Adversarial Multi Agent: MORE
- **Paper:**[Adversarial Multi-Agent Evaluation of Large Language Models through Iterative Debates](https://arxiv.org/html/2410.04663v2)
- **Codebase:** None.Implemented through the paper's algorithm.


### Single Agent Evaluator Framework

The single agent evaluator framework allows you to evaluate responses from a single LLM (e.g., any model available in your local Ollama instance) on a dataset, using a configurable evaluation prompt and outputting results in a standardized JSONL format.

Details on the the single agent evaluator framework can be found [here](framework_runner/SINGLE_AGENT_README.md).


## Framework Runner for Multi-Domain LLM Evaluation

This is packaged module for evaluating and running a framework like the ones listed above, but can be scaled to include other frameworks as well, by creating a new implementation class that adheres to the `Framework` ABC defined in `base.py`. This allowed us to run all the frameworks in a consistent manner, and produce comparable results.

Additional details can be found [here](framework_runner/README.md).

## Repository Structure

```
CSCI544_AppliedNLP_Group/
│
├── datasets
    ├── data
        ├── math_cleaned_250.csv
        ├── math_cleaned_250.json
        ├── med_cleaned.csv
        ├── med_cleaned.json
        ├── openQA_cleaned_250.csv
        └── openQA_cleaned_250.json
├── ChatEval/
├── Debatable-Intelligence/
   ├── debate_runtime/
   ├── src/
   ├── scripts/
├── Adversarial Multi Agent/
├── DEBATE/
├── framework_runner/
    ├── base.py
    ├── debate_impl.py
    ├── debint_impl.py
    ├── huggingface_eval.py
    ├── main.py
    ├── ollama_eval.py
    ├── README.md
    ├── runner.py
    ├── single_agent_impl.py
├── {add yours4}/
├── {add yours5}/
├── reports/
│   ├── proposal.pdf              # Group project proposal
│   ├── project_status_report.pdf # Mid-term progress report
│   └── final_report.pdf          # Final submission
├── prompts/
    ├── eval_math_response.txt
    ├── eval_medical_response.txt
    └── eval_openqa_response.txt
├── results/
└── README.md
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/lucasjeongsikpark/CSCI544_AppliedNLP_GroupProject.git
cd CSCI544_AppliedNLP_GroupProject
```

---
