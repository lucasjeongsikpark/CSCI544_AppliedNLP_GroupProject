# Single Agent Evaluator Framework

This framework allows you to evaluate responses from a single LLM (e.g., any model available in your local Ollama instance) on a dataset, using a configurable evaluation prompt and outputting results in a standardized JSONL format.

## Features
- **Model Agnostic**: Use any model available in your Ollama instance (e.g., llama2, mistral, phi, etc.).
- **Prompt Configurable**: Specify any evaluation prompt template (e.g., for medical, math, or openQA tasks).
- **Dual Evaluation**: Automatically evaluates both `llama_output` and `distill_llama_output` fields in each dataset row.
- **Standard Output**: Results are saved in a JSONL format compatible with other frameworks in this repo.

## Usage

### Example Command
```bash
python3 -m framework_runner.main --framework single_agent \
  --dataset_path datasets/data/med_cleaned.json \
  --output_file results/single_agent_medical.jsonl \
  --aspects_file prompts/eval_medical_response.txt \
  --dataset_type medical \
  --model_name mistral \
  --ollama_base_url http://localhost:11434 \
  --prompt_template_path prompts/eval_medical_response.txt \
  --response_field llama_output \
  --max_tokens 512 \
  --temperature 0.2
```

### Key Arguments
- `--framework single_agent` : Use the single agent evaluator.
- `--model_name` : Name of the Ollama model to use (must be pulled and available in Ollama).
- `--ollama_base_url` : URL for your Ollama server (default: `http://localhost:11434`).
- `--prompt_template_path` : Path to the evaluation prompt template.
- `--response_field` : Which field in your dataset to evaluate (default: `llama_output`).
- `--max_tokens` : Max tokens for LLM output (default: 512).
- `--temperature` : Sampling temperature for LLM (default: 0.2).

### Output
- Results are saved as JSONL, one record per input, with fields:
  - `chat_log`: Contains results for both `llama_output` and `distill_llama_output` evaluations, including LLM response, metrics, overall score, scratchpad, and elapsed time for each.
  - `score1`: Dict of aspect scores for `llama_output`.
  - `score2`: Dict of aspect scores for `distill_llama_output`.
  - `attempts`: Number of evaluation attempts (should be 2: one for each field).

## Requirements
- Python 3.8+
- Ollama running locally with your desired model pulled (see [Ollama documentation](https://ollama.com)).
- All dependencies in `requirements.txt` installed.

## Example: Listing Available Models in Ollama
```bash
ollama list
```

## Example: Pulling a Model in Ollama
```bash
ollama pull mistral
```

## Implementation Details

### Architecture
The `SingleAgentFramework` class (`framework_runner/single_agent_impl.py`) is built on top of the `Framework` base class and leverages the `OllamaEvaluator` to interact with local Ollama instances.

### Key Components

#### 1. Prompt Template Handling
- Loads a prompt template from a specified file path
- Uses dynamic placeholder substitution with fields like `{DOCUMENT}`, `{INPUT}`, `{OUTPUT}`, `{RESPONSE}`, and `{SYSTEM_PROMPT}`
- Implements a `_SafeDict` class that returns empty strings for missing placeholders, preventing KeyError exceptions
- Only computes placeholders that are actually present in the template for efficiency

#### 2. Dual Evaluation Strategy
- **Automatic dual evaluation**: For each dataset row, the framework automatically evaluates both `llama_output` and `distill_llama_output` fields
- This allows for side-by-side comparison of different model outputs or variants
- Results are returned as `score1` (for llama_output) and `score2` (for distill_llama_output)

#### 3. Response Parsing
The `_parse_metrics()` method extracts structured data from LLM responses using regex patterns:
- **Metrics extraction**: Looks for `<metrics>...</metrics>` blocks and extracts numeric scores (0-9) for each aspect
- **Overall score**: Extracts a single digit from `<overall_score>...</overall_score>` tags
- **Scratchpad**: Captures reasoning/explanation from `<scratchpad>...</scratchpad>` sections

#### 4. Retry Logic with Quality Control
- Implements up to 3 retries per field when parsing fails
- Validates that responses contain:
  - All expected metrics (one per aspect)
  - An overall score value
  - A non-empty scratchpad/reasoning section
- Only breaks the retry loop when all three conditions are met or max retries exceeded
- Tracks the number of attempts needed for each evaluation

#### 5. Timing and Performance Metrics
- Records elapsed time for each LLM call using Python's `time` module
- Useful for understanding model performance and latency

### Data Flow
1. **Input**: Dataset row (dict) containing `llama_output` and `distill_llama_output` fields
2. **Processing**:
   - Fill prompt template with actual data
   - Call Ollama model for evaluation
   - Parse structured output (metrics, scores, reasoning)
   - Retry if necessary to ensure valid output
3. **Output**: `DataOutput` object containing:
   - `chat_logs`: Full LLM responses and parsed results for each field
   - `score1` & `score2`: Aspect-level scores for each output variant
   - `attempts`: Maximum retry attempts needed across both evaluations

### Integration with Framework
- Extends the base `Framework` interface defined in `framework_runner/base.py`
- Integrates with `framework_runner/main.py` for command-line execution
- Compatible with the broader evaluation framework architecture

## Notes
- Both `llama_output` and `distill_llama_output` are always evaluated for each row; you do not need to specify the field.
- You can use any prompt template, making this framework adaptable to many evaluation tasks.
- For best results, ensure your Ollama server is running and the model is available before running the script.
- The framework expects structured output from the LLM (with `<metrics>`, `<overall_score>`, and `<scratchpad>` tags); design your prompt template accordingly.

## Troubleshooting
- If you get a 404 error, check that the model name matches one available in Ollama (`ollama list`).
- If you want to use a remote Ollama server, set `--ollama_base_url` accordingly.
- If metrics are not being parsed correctly, verify that your LLM output includes the expected XML-style tags.
- If retries are consistently exhausted, review your prompt template to ensure it guides the model toward the expected output format.

---
For more details, see the code in `framework_runner/single_agent_impl.py` and the example commands in your project root or this README.
