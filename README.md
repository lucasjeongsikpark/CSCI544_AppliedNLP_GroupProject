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

This framework has been extended to create debates in the style of traditional competitive debates, with two LLM agents, an Affirmative arguing for and a Negative side against the response quality, and a judge LLM making the final decision. The implementation can be found in the `Debatable-Intelligence/debate_runtime/` folder.

Additional details on how this works and is implemented can be found in the [here](Debatable-Intelligence/debate_runtime/README_DEBATE_RUNTIME.md).

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

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### Ollama Setup
Follow the instructions at [Ollama](https://ollama.com) to install Ollama on your local machine.

### Debatable-Intelligence Setup
Set up a virtual environment:
```bash
python -m venv {NAME_OF_YOUR_VENV}
source {NAME_OF_YOUR_VENV}/bin/activate 
```
For windows, use:
```bash
{NAME_OF_YOUR_VENV}\Scripts\activate
```

Install dependencies:
```bash
pip install -r Debatable-Intelligence/requirements.txt
```
On macOS, use:
```bash
pip install -r Debatable-Intelligence/requirements_arm64.txt
```

Now, you will use the `framework_runner` module to run this framework. For this one in particular, you would make sure to be in the root of the repository (`CSCI544_AppliedNLP_GroupProject/`). For example:

```bash
python3 -m framework_runner.main \
  --framework debint \
  --dataset_path path/to/dataset.(json|csv) \
  --output_file path/to/results.jsonl \
  --dataset_type (math|medical|openqa|{CUSTOM})
```

Detailed and elaborated instructions can be found and are referenced in [Framework Runner - Debate Runtime](Debatable-Intelligence/debate_runtime/README_DEBATE_RUNTIME.md#running-debates-via-framework_runner).

If you want to test and add on a new domain/dataset, refer to the instructions [here](Debatable-Intelligence/debate_runtime/README_DEBATE_RUNTIME.md#adding-a-new-domaindataset-configuration).

---
