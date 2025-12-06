## ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate

<p align="center">
  <a href="https://arxiv.org/abs/2308.07201">Paper</a> •
  <a href="#-simple-video-demo">Video Demo</a> •
  <a href="#-getting-started">Getting Started</a> •
  ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
  ---

  <p align="center">
    <a href="https://arxiv.org/abs/2308.07201">Paper</a> •
    <a href="#-simple-video-demo">Video Demo</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#entry-point">Entry Point</a> •
    <a href="#citation">Citation</a>
  </p>

**ChatEval** provides a framework to evaluate and compare generated text using multiple LLM-based agents that debate and score candidate responses. Roles (agents) with different personas discuss, provide evidentiary explanations, and then output final judgments. ChatEval supports both scripted evaluation runs (batch experiments) and an arena-style demo built on top of FastChat.

## Entry Point

- **Script:** `run.sh`
- **Purpose:** Convenience entrypoint to run a default evaluation configuration. By default `run.sh` runs a math evaluation config (`agentverse/tasks/llm_eval/gemma/gemma_math_config.yaml`).

Usage:

```bash
# make executable (optional)
chmod +x run.sh

# run the default evaluation
./run.sh

# or run with bash explicitly
bash run.sh
```

To run a different configuration, either edit `run.sh` (uncomment/change an alternative config line) or call `llm_eval.py` directly:

```bash
python llm_eval.py --config agentverse/tasks/llm_eval/gemma/gemma_openQA_config.yaml
python llm_eval.py --config agentverse/tasks/llm_eval/gemma/gemma_med_config.yaml
```

## 🎥 Simple Video Demo

Our video demo shows how ChatEval can compare two generated texts. While FastChat's arena allows manual voting, our demo uses multiple LLM referees to autonomously determine which response is better and to provide transparent evidence for their decisions.

Steps to run the arena demo (optional):

1. Navigate to the FastChat folder:

```bash
cd ChatEval/FastChat
```

2. Launch the controller:

```bash
python3 -m fastchat.serve.controller
```

3. Register model workers (example):

```bash
# worker 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
```

4. Run the gradio web server:

```bash
python3 -m fastchat.serve.gradio_web_server_multi
```

Open the web UI and observe how LLM referees issue judgments.

## 🚀 Getting Started

### Installation

Clone this repository and install Python dependencies:

```bash
git clone https://github.com/chanchimin/ChatEval.git
cd ChatEval
pip install -r requirements.txt
# or create conda env and run pip install -r requirements_conda.txt
```

You will typically need access to LLM APIs (OpenAI or others). Export your API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
# not needed in this case (when you run the model locally)
```

### Prepare Dataset

An example dataset is provided at `agentverse/tasks/llm_eval/data/faireval/preprocessed_data/test.json`.
Custom data should follow the same JSON schema: a list of items with `question`, `question_id`, and `response` entries mapping model names to response text.

### Configure Custom Debater Agent

Customize agents in `agentverse/tasks` (see `agentverse/tasks/llm_eval/config.yaml` for an example). Each agent entry controls persona, prompts, memory, and LLM settings.

### Run the evaluation script

Run a specific config (example uses the default `config.yaml`):

```bash
python llm_eval.py --config agentverse/tasks/llm_eval/config.yaml
```

Or use the provided `run.sh` convenience script to run the default gemma math configuration:

```bash
./run.sh
```

### Check the evaluation results

Results are saved to the output directory defined by your config (see `config.output_dir`). Each output entry contains the question, candidate responses, and agent evaluations with evidence and scores.

## Citation

If you find this repo helpful, please cite:

```bibtex
@misc{chan2023chateval,
      title={ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate},
      author={Chi-Min Chan and Weize Chen and Yusheng Su and Jianxuan Yu and Wei Xue and Shanghang Zhang and Jie Fu and Zhiyuan Liu},
      year={2023},
      eprint={2308.07201},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

      llm_type: gpt-3.5-turbo-0301
