# Dataset Evaluation Experiments

This directory contains adapted experiments for evaluating LLM responses on three different datasets: Math, OpenQA, and Medical.

## Overview

The framework has been tailored to evaluate LLM-generated responses across three distinct domains:

1. **Math Dataset** (`datasets/data/math.csv`): Mathematical word problems with step-by-step solutions
2. **OpenQA Dataset** (`datasets/data/openQA.csv`): Open-domain question-answering tasks
3. **Medical Dataset** (`datasets/data/med.csv`): Medical advice and patient-doctor interactions

## Dataset Formats

### Math Dataset
- **Columns**: `input`, `output`
- **input**: Math problem/question
- **output**: Step-by-step solution with final answer

### OpenQA Dataset
- **Columns**: `system_prompt`, `input`, `output`
- **system_prompt**: Instructions for the AI assistant
- **input**: Question or task
- **output**: Expected detailed answer

### Medical Dataset
- **Columns**: `document`, `input`, `output`
- **document**: Medical context/research document
- **input**: Patient question
- **output**: Medical professional's response

## Configuration Files

Each dataset has its own configuration file in `src/`:

- `config_math.json` - Math dataset configuration
- `config_openqa.json` - OpenQA dataset configuration
- `config_med.json` - Medical dataset configuration

### Configuration Structure

```json
{
  "data_path": "../datasets/data/[dataset].csv",
  "output_path": "output/[dataset]_results/",
  "huggingface_key_path": "secret_keys/huggingface_key",
  "max_tokens": 2048,
  "temperature": 0.01,
  "dataset_type": "[math|openqa|medical]",
  "experiments": [...],
  "models": {
    "huggingface": [
      "meta-llama/Llama-2-7b-chat-hf",
      "mistralai/Mistral-7B-Instruct-v0.2"
    ]
  },
  "participating_models": ["huggingface"]
}
```

## Evaluation Prompts

Custom evaluation prompts for each dataset type:

- `prompts/eval_math_response.txt` - Evaluates mathematical correctness and reasoning
- `prompts/eval_openqa_response.txt` - Evaluates relevance, completeness, and accuracy
- `prompts/eval_medical_response.txt` - Evaluates medical accuracy, safety, and professionalism

Each prompt uses a 1-5 scoring scale:
- 1 = Very Poor
- 2 = Poor  
- 3 = Fair
- 4 = Good
- 5 = Excellent

## Running Experiments

### Prerequisites

1. **Virtual Environment**: Activate the virtual environment
```bash
source di-venv/bin/activate
```

2. **HuggingFace API Key**: Ensure you have your HuggingFace API key in `secret_keys/huggingface_key`

3. **GPU Access**: For running quantized models, ensure you have GPU access on your HPC machine

### Run Individual Experiments

```bash
# Math dataset evaluation
./scripts/run_math_eval.sh

# OpenQA dataset evaluation
./scripts/run_openqa_eval.sh

# Medical dataset evaluation
./scripts/run_medical_eval.sh
```

### Or run directly with Python

```bash
cd Debatable-Intelligence
python src/dataset_experiment.py --config src/config_math.json
python src/dataset_experiment.py --config src/config_openqa.json
python src/dataset_experiment.py --config src/config_med.json
```

## Recommended HuggingFace Models for HPC

### For Single GPU (24-48GB):
- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2.5-7B-Instruct`
- `microsoft/phi-2`

### For Larger GPUs (40-80GB):
- `meta-llama/Llama-2-13b-chat-hf`
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (with quantization)
- `Qwen/Qwen2.5-14B-Instruct`

### With Quantization (4-bit):
The framework automatically applies 4-bit quantization for models >10B parameters using BitsAndBytes:
- Reduces memory footprint by ~75%
- Enables running 70B models on 40-80GB GPUs
- Minimal accuracy loss with NF4 quantization

## Modifying Configurations

### To add more models:

Edit the config file (e.g., `src/config_math.json`):

```json
{
  "models": {
    "huggingface": [
      "meta-llama/Llama-2-7b-chat-hf",
      "mistralai/Mistral-7B-Instruct-v0.2",
      "YOUR_NEW_MODEL_NAME"
    ]
  }
}
```

### To change evaluation criteria:

Edit the corresponding prompt file in `prompts/eval_[dataset]_response.txt`

### To use different API providers:

Add API keys and update `participating_models`:

```json
{
  "openai_key_path": "secret_keys/openai_key",
  "anthropic_key_path": "secret_keys/anthropic_key",
  "models": {
    "openai": ["gpt-4o"],
    "anthropic": ["claude-3-5-sonnet-20241022"]
  },
  "participating_models": ["openai", "anthropic", "huggingface"]
}
```

## Output

Results are saved to:
- `output/math_results/` - Math evaluation results
- `output/openqa_results/` - OpenQA evaluation results  
- `output/medical_results/` - Medical evaluation results

Each output contains:
- CSV file with original data + model responses + scores
- JSON files with batch completion data
- Log files in `logs/` subdirectory

## Key Differences from Original Framework

1. **Dataset Structure**: Adapted to handle different column formats (input/output vs topic/speech)
2. **Prompt Generation**: Dataset-specific prompt templates with appropriate placeholders
3. **Evaluation Criteria**: Domain-specific evaluation rubrics (math correctness vs. medical safety)
4. **ID Management**: Automatic ID generation for datasets without ID columns
5. **Quantization**: Automatic 4-bit quantization for larger HuggingFace models

## Troubleshooting

### Out of Memory Errors
- Use smaller models (7B instead of 13B+)
- Ensure quantization is enabled for large models
- Reduce batch size in model.py if needed

### API Rate Limits
- Add delays between requests in model.py
- Use batch processing features when available
- Consider using local inference instead of API calls

### CUDA Errors
- Check CUDA compatibility: `nvidia-smi`
- Ensure bitsandbytes is installed: `pip install bitsandbytes`
- Verify PyTorch CUDA version matches system CUDA

## Citation

Original framework adapted from Debatable-Intelligence project for multi-domain LLM evaluation.
