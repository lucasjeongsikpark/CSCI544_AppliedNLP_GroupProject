# Debate Runtime Extension for Debatable-Intelligence

This module transforms the original judge-centric Debatable-Intelligence framework into a lightweight MADE (Multi-Agent Debate) system while **retaining its strengths in fine-grained evaluation**.

## Key Concepts
- **Generation Agents**: Roles (AFFIRMATIVE, NEGATIVE, etc.) produce debate speeches each turn.
- **Judge Agent**: After every speech, the judge model evaluates quality and assigns a `<score>1-5</score>` with `<scratchpad>` reasoning.
- **State Tracking**: All speeches + per-turn judgment stored in a `DebateState` object serialized to JSON.

## Files
| File | Purpose |
|------|---------|
| `roles.py` | Provides role templates and role prompt rendering. |
| `state.py` | Data classes for `Speech` and `DebateState`. |
| `judge_bridge.py` | Parses judge model outputs and attaches scores. |
| `orchestrator.py` | Core loop managing rounds, role turns, and judging. |
| `runner_debate.py` | CLI entry point to run a debate from a config file. |
| `debate_config.example.json` | Example configuration to start quickly. |
| `runner_debate_dataset.py` | Batch debate runner over a JSON dataset. |
| `debate_dataset_config.example.json` | Example dataset batch config. |

## Configuration (`debate_config.example.json`)
```json
{
  "topic": "Should universal basic income be implemented?",
  "max_rounds": 2,
  "roles_order": ["AFFIRMATIVE", "NEGATIVE"],
  "judge_role": "JUDGE",
  "temperature": 0.7,
  "max_tokens_generation": 400,
  "max_tokens_judge": 256,
  "generation_models": {
    "AFFIRMATIVE": {"provider": "ollama", "engine": "llama3.1"},
    "NEGATIVE": {"provider": "ollama", "engine": "llama3.1"}
  },
  "judge_model": {"provider": "ollama", "engine": "llama3.1"},
  "output_dir": "debate_outputs"
}
```

### Supported Providers
Currently wired to existing framework wrappers: `openai`, `openai_reasoning`, `huggingface`, `anthropic`, `ollama`.

Add new providers by extending `PROVIDER_CLASS` in `runner_debate.py` or reusing existing `model.py` classes.

## Running a Debate
1. Create a config JSON (copy and adapt the example).
2. Ensure keys or local models are available (e.g., Ollama running, OpenAI API key file paths etc.).
3. Run:
```bash
python -m debate_runtime.runner_debate --config path/to/debate_config.json
```
4. Outputs:
   - JSON state file at `output_dir/debate_state.json` containing all speeches + judgments.
   - Logs under `output_dir/logs/` with per-turn scores.

## Output Structure (`debate_state.json` excerpt)

The output format matches the original Debatable-Intelligence schema with additional debate information:

```json
{
  "chat_log": {
    "llama_output": {
      "response": "<ANSWER_B content>",
      "reasoning": "Judge reasoning for llama_output evaluation"
    },
    "distill_llama_output": {
      "response": "<ANSWER_A content>",
      "reasoning": "Judge reasoning for distill_llama_output evaluation"
    },
    "debate_speeches": [
      {
        "turn": 1,
        "role": "AFFIRMATIVE",
        "content": "...",
        "scores": {
          "overall": 4,
          "reasoning": "Clear defense of social safety nets",
          "metrics": {
            "correctness": 4,
            "reasoning": 5,
            "completeness": 4,
            "accuracy": 3
          }
        }
      },
      {
        "turn": 2,
        "role": "NEGATIVE",
        "content": "...",
        "scores": {
          "overall": 3,
          "reasoning": "Challenges funding but lacks data"
        }
      }
    ]
  },
  "score1": {
    "Correctness": 5,
    "Reasoning": 4,
    "Completeness": 5,
    "Accuracy": 5,
    "reasoning": "Judge reasoning text for llama_output"
  },
  "score2": {
    "Correctness": 5,
    "Reasoning": 4,
    "Completeness": 5,
    "Accuracy": 5,
    "reasoning": "Judge reasoning text for distill_llama_output"
  },
  "attempts": 1,
  "elapsed_time": 12.345
}
```

**Key Fields**:
- `chat_log`: Contains both candidate answers and all debate speeches
  - `llama_output`: ANSWER_B with evaluation reasoning
  - `distill_llama_output`: ANSWER_A with evaluation reasoning
  - `debate_speeches`: Turn-by-turn debate with per-speech scores
- `score1`: Individual metric scores for `llama_output` (ANSWER_B)
- `score2`: Individual metric scores for `distill_llama_output` (ANSWER_A)
- `attempts`: Number of generation attempts (default: 1)
- `elapsed_time`: Total debate runtime in seconds

## Extending Metrics

### Domain-Specific Evaluation
The framework now supports **structured, domain-specific evaluation** using the existing prompt templates from `prompts/`:
- **Math** (`eval_math_response.txt`): Evaluates correctness, reasoning, completeness, accuracy
- **Medical** (`eval_medical_response.txt`): Evaluates medical_accuracy, appropriateness, safety, clarity, professionalism
- **OpenQA** (`eval_openqa_response.txt`): Evaluates relevance, completeness, accuracy, clarity, helpfulness

Enable domain-specific evaluation by adding `domain` to your config:
```json
{
  "dataset_path": "datasets/data/math_cleaned_250.json",
  "domain": "math",
  "ground_truth_field": "output",
  ...
}
```

**Output Format**: Judge responses now include structured metrics:
```json
{
  "turn": 1,
  "role": "AFFIRMATIVE",
  "content": "...",
  "scores": {
    "overall": 4,
    "reasoning": "Strong reasoning but minor calculation error",
    "metrics": {
      "correctness": 4,
      "reasoning": 5,
      "completeness": 4,
      "accuracy": 3
    }
  }
}
```

**How It Works**:
1. If `domain` is specified (`math`, `medical`, or `openqa`), the orchestrator loads the corresponding prompt template
2. Template placeholders are filled with dataset fields (`{INPUT}`, `{OUTPUT}`, `{RESPONSE}`, etc.)
3. Judge output is parsed for both `<overall_score>` and individual `<metrics>` tags
4. Results stored in `scores.metrics` dict for each speech

### Custom Multi-Dimensional Evaluation
To introduce additional evaluation dimensions (e.g. keypoint coverage, rebuttal precision):
1. Expand `judge_bridge.attach_judgment` to include additional fields.
2. Implement analysis modules that post-process `state.speeches` for trends.
3. Optionally create `metrics.py` to compute:
   - Keypoint coverage using existing keypoint datasets.
   - Rebuttal precision via claim extraction & matching.
   - Calibration curves correlating judge score vs agent self-estimated confidence.

## Running Debates Over a Dataset
Use `runner_debate_dataset.py` to iterate over entries where each item supplies a topic and optionally TWO candidate answers (e.g., `llama_output` and `distill_llama_output`) for comparative debate.

Example config (`debate_dataset_config.example.json`):
```json
{
  "dataset_path": "datasets/data/math_cleaned_250.json",
  "topic_field": "input",
  "context_field": "distill_llama_output",
  "secondary_context_field": "llama_output",
  "ground_truth_field": "output",
  "domain": "math",
  "max_items": 3,
  "debate": {"max_rounds": 2, "roles_order": ["AFFIRMATIVE", "NEGATIVE"], "judge_role": "JUDGE"},
  "generation_models": {"AFFIRMATIVE": {"provider": "ollama", "engine": "llama3.1"}, "NEGATIVE": {"provider": "ollama", "engine": "llama3.1"}},
  "judge_model": {"provider": "ollama", "engine": "llama3.1"},
  "output_dir": "debate_outputs_dataset"
}
```

**Comparative Debate Mode**: When both `context_field` and `secondary_context_field` are provided, agents see:
- `<ANSWER_A>` (from `context_field` = `distill_llama_output`)
- `<ANSWER_B>` (from `secondary_context_field` = `llama_output`)

AFFIRMATIVE can defend one answer or synthesize best reasoning; NEGATIVE can critique both or highlight contradictions.

Run:
```bash
python -m debate_runtime.runner_debate_dataset --config debate_runtime/debate_dataset_config.example.json
```

Each entry produces `debate_{index}.json` with both contexts included, enabling evaluation of how agents evaluate competing solutions.

### Mapping Other Datasets
| Dataset | topic_field | context_field | secondary_context_field | domain |
|---------|-------------|---------------|-------------------------|--------|
| `math_cleaned_250.json` | `input` | `distill_llama_output` | `llama_output` | `math` |
| `openQA_cleaned_250.json` | `distill_llama_output` (narrative) | `llama_output` | (omit for single-answer mode) | `openqa` |
| `med_cleaned.json` | `distill_llama_output` (clinical scenario) | `llama_output` | (omit for single-answer mode) | `medical` |

**Single-answer mode**: Omit `secondary_context_field` to debate one answer.  
**Dual-answer mode**: Provide both fields for comparative evaluation of two LLM responses.
**Domain-specific evaluation**: Set `domain` field to enable structured metric evaluation (requires `ground_truth_field`).

To introduce multi-dimensional evaluation (e.g. keypoint coverage, rebuttal precision):
1. Expand `judge_bridge.attach_judgment` to include additional fields.
2. Implement analysis modules that post-process `state.speeches` for trends.
3. Optionally create `metrics.py` to compute:
   - Keypoint coverage using existing keypoint datasets.
   - Rebuttal precision via claim extraction & matching.
   - Calibration curves correlating judge score vs agent self-estimated confidence.

## Integration with `framework_runner`

The debate runtime can be executed through the unified `framework_runner` system, which provides a standardized CLI interface for running debates alongside other evaluation frameworks (single_agent, debint, etc.). This allows you to evaluate multiple datasets and domains using the same command structure.

### How It Works with framework_runner

When you run debates via `framework_runner`, the system:
1. Routes to the `debate` framework implementation (`framework_runner/debate_impl.py`)
2. Automatically selects the appropriate debate config based on `--dataset_type`
3. Processes the dataset and outputs results in standardized JSONL format
4. Integrates with the framework's aspect definition and result aggregation system

### Running Debates via framework_runner

**Basic syntax:**
```bash
python3 -m framework_runner.main \
  --framework debate \
  --dataset_path <path_to_dataset> \
  --output_file <output_path> \
  --dataset_type <domain> \
  [--config_file <config_path>]
```

**Example: Running Math Dataset Debate**
```bash
python3 -m framework_runner.main \
  --framework debate \
  --dataset_path datasets/data/math_cleaned_250.json \
  --output_file results/debate_math_250.jsonl \
  --dataset_type math
```

This command automatically:
- Searches for `debate_dataset_math_config.json` (or `config_math.json`) in the current/parent directory
- Loads the math evaluation prompt from the framework's prompts directory
- Runs debates for each entry in the dataset
- Outputs results in JSONL format compatible with result analysis tools

**Example: Running Medical Dataset with Explicit Config**
```bash
python3 -m framework_runner.main \
  --framework debate \
  --dataset_path datasets/data/med_cleaned.json \
  --output_file results/debate_medical.jsonl \
  --dataset_type medical \
  --config_file debate_runtime/debate_dataset_config_medical.json
```

**Example: Running OpenQA Dataset**
```bash
python3 -m framework_runner.main \
  --framework debate \
  --dataset_path datasets/data/openQA_cleaned_250.json \
  --output_file results/debate_openqa.jsonl \
  --dataset_type openqa
```

### Config File Auto-Discovery

The framework uses a priority-based search to locate debate configs:

1. **Explicit `--config_file`** (highest priority)
   - Uses the file path you specify directly
   - Override everything else

2. **Auto-discovery via `--dataset_type`**
   - Searches for `debate_dataset_<domain>_config.json` or `config_<domain>.json`
   - Looks in current working directory, then parent directory
   - Example: `--dataset_type math` searches for `debate_dataset_math_config.json` or `config_math.json`

3. **Fallback to default**
   - If no domain-specific config found, uses `config.json` or `debate_config.json`

### Adding a New Domain/Dataset Configuration

Let's walk through adding support for a new domain, e.g., **Code Generation**.

#### Step 1: Create Your Domain Config File

Create `debate_runtime/debate_dataset_code_generation_config.json`. Ensure the file name follows the pattern `debate_dataset_<domain>_config.json` for auto-discovery:
```json
{
  "dataset_path": "datasets/data/code_generation_250.json",
  "topic_field": "problem_statement",
  "context_field": "distill_llama_output",
  "secondary_context_field": "llama_output",
  "ground_truth_field": "expected_output",
  "domain": "code_generation",
  "max_items": 250,
  "debate": {
    "max_rounds": 2,
    "roles_order": ["AFFIRMATIVE", "NEGATIVE"],
    "judge_role": "JUDGE",
    "temperature": 0.7,
    "max_tokens_generation": 512,
    "max_tokens_judge": 512
  },
  "generation_models": {
    "AFFIRMATIVE": {"provider": "ollama", "engine": "gemma2:2b"},
    "NEGATIVE": {"provider": "ollama", "engine": "gemma2:2b"}
  },
  "judge_model": {"provider": "ollama", "engine": "gemma2:2b"},
  "output_dir": "code_generation_dataset_outputs"
}
```

**Key fields for your dataset:**
- `dataset_path`: Path to your JSON dataset file
- `topic_field`: Field containing the question/problem (e.g., `problem_statement`)
- `context_field`: First response to compare (usually `distill_llama_output`)
- `secondary_context_field`: Second response to compare (usually `llama_output`)
- `ground_truth_field`: Reference/expected output for evaluation
- `domain`: Name of your domain (used for routing and prompt selection)
- `max_items`: Number of items to process (or omit to process all)

#### Step 2: Create Domain-Specific Evaluation Prompt

Create `Debatable-Intelligence/prompts/eval_code_generation_response.txt`:
```
You are an expert code reviewer evaluating two code solutions.

TASK: Compare the following code solutions for correctness, efficiency, and code quality.

INPUT:
{INPUT}

EXPECTED OUTPUT:
{OUTPUT}

SOLUTION A:
{RESPONSE}

Please provide a detailed evaluation with the following structure:

<scratchpad>
- Analyze correctness: Does the code produce the expected output?
- Evaluate efficiency: Time and space complexity
- Review code quality: Readability, maintainability, proper error handling
- Compare both solutions' strengths and weaknesses
</scratchpad>

<metrics>
- correctness: 1-5 (does it solve the problem correctly?)
- efficiency: 1-5 (is the algorithm optimized?)
- code_quality: 1-5 (is it clean and maintainable?)
- completeness: 1-5 (does it handle edge cases?)
</metrics>

<overall_score>4</overall_score>
```

**Metrics should match your domain requirements.** The framework will parse these metric names from the prompt and use them for scoring.

#### Step 3: Register Your Domain (Optional Auto-Discovery)

For automatic aspect discovery, update `framework_runner/main.py` to include your domain in `DEFAULT_ASPECTS`:

```python
DEFAULT_ASPECTS = {
    'math': ['correctness', 'reasoning', 'completeness', 'accuracy'],
    'openqa': ['relevance', 'completeness', 'accuracy', 'clarity', 'helpfulness'],
    'medical': ['medical_accuracy', 'appropriateness', 'safety', 'clarity', 'professionalism'],
    'code_generation': ['correctness', 'efficiency', 'code_quality', 'completeness'],  # Add this
}
```

#### Step 4: Run Your Domain Debate

```bash
# Option A: Using explicit config file
python3 -m framework_runner.main \
  --framework debate \
  --dataset_path datasets/data/code_generation_250.json \
  --output_file results/debate_code_generation.jsonl \
  --dataset_type code_generation \
  --config_file debate_runtime/debate_dataset_code_generation_config.json

# Option B: Using auto-discovery (if config file in repo root or parent dir)
python3 -m framework_runner.main \
  --framework debate \
  --dataset_path datasets/data/code_generation_250.json \
  --output_file results/debate_code_generation.jsonl \
  --dataset_type code_generation
```

#### Step 5: Verify Output Format

Results will be saved to JSONL, with each line containing:
```json
{
  "chat_log": {
    "llama_output": {
      "response": "<solution text>",
      "reasoning": "<judge reasoning>"
    },
    "distill_llama_output": {
      "response": "<solution text>",
      "reasoning": "<judge reasoning>"
    },
    "debate_speeches": [
      {
        "turn": 1,
        "role": "AFFIRMATIVE",
        "content": "...",
        "scores": {
          "overall": 4,
          "metrics": {
            "correctness": 4,
            "efficiency": 5,
            "code_quality": 4,
            "completeness": 3
          }
        }
      },
      ...
    ]
  },
  "score1": {"correctness": 4, "efficiency": 5, "code_quality": 4, "completeness": 3},
  "score2": {"correctness": 4, "efficiency": 4, "code_quality": 5, "completeness": 4},
  "attempts": 1,
  "elapsed_time": 45.2
}
```

### Comparing Results Across Frameworks

The framework_runner outputs standardized JSONL that allows you to compare debate, debint, and single_agent evaluations:

```bash
# Run all three frameworks on the same dataset
python3 -m framework_runner.main --framework debate --dataset_path datasets/data/code_generation_250.json --output_file results/debate_code.jsonl --dataset_type code_generation

python3 -m framework_runner.main --framework debint --dataset_path datasets/data/code_generation_250.json --output_file results/debint_code.jsonl --dataset_type code_generation

python3 -m framework_runner.main --framework single_agent --dataset_path datasets/data/code_generation_250.json --output_file results/single_agent_code.jsonl --dataset_type code_generation --model_name gemma2:2b --prompt_template_path prompts/eval_code_generation_response.txt
```

Then analyze correlations between evaluation methods using standard analysis tools.

## Design Choices
- **Immediate Judging**: Judge called after each speech (supports real-time feedback loops later).
- **Context Windowing**: `DebateState.history_text()` returns last N speeches to reduce prompt length.
- **Provider Reuse**: Leverages existing `model.py` wrappers; no duplication.

## Next Steps (Suggested Enhancements)
- Add feedback injection: Provide judge reasoning back to next speaking agent.
- Implement advanced metrics (novelty, persuasion trajectory).
- Add early stopping if scores stagnate or max persuasive threshold reached.
- Visualization notebook for per-turn score evolution.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Empty scores (`-1`) | Judge failed to parse `<score>` | Ensure judge prompt includes instructions, increase max_tokens_judge. |
| Repetitive speeches | Temperature too low / narrow context window | Raise temperature or increase history length. |
| Slow generation | Large remote model or batch | Lower max_tokens_generation or switch provider. |

## License & Attribution
Retains original project licensing and extends functionality; attribute original Debatable-Intelligence authors when publishing results.

---
This runtime bridges evaluation and interactive debate—positioning the framework uniquely among MADE systems.
