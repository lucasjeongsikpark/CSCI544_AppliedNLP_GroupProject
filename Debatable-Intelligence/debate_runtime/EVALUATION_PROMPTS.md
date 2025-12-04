# Domain-Specific Evaluation Prompts Integration

## Overview
The debate runtime now supports **structured, metric-based evaluation** using the existing prompt templates from `prompts/`. This enables fine-grained assessment beyond generic debate quality scoring.

## Supported Domains

### 1. Math (`domain: "math"`)
**Prompt File**: `prompts/eval_math_response.txt`

**Evaluated Metrics**:
- `correctness`: Correct final answer (1-5)
- `reasoning`: Logical, coherent step-by-step derivation (1-5)
- `completeness`: All necessary steps shown, no gaps (1-5)
- `accuracy`: Calculations/arithmetic are correct (1-5)

**Required Config Fields**:
- `ground_truth_field`: "output" (expected answer)

**Template Placeholders**:
- `{INPUT}`: Math problem (from topic)
- `{OUTPUT}`: Expected correct answer (from ground_truth_field)
- `{RESPONSE}`: Agent's speech content

---

### 2. Medical (`domain: "medical"`)
**Prompt File**: `prompts/eval_medical_response.txt`

**Evaluated Metrics**:
- `medical_accuracy`: Medical soundness of facts (1-5)
- `appropriateness`: Suitability for patient context (1-5)
- `safety`: Emphasis on safe, non-harmful guidance (1-5)
- `clarity`: Understandability and organization (1-5)
- `professionalism`: Tone and ethical/empathetic stance (1-5)

**Required Config Fields**:
- `ground_truth_field`: "output" (reference response)
- Dataset should include `document` field (medical context)

**Template Placeholders**:
- `{DOCUMENT}`: Medical document/context (from item.document)
- `{PATIENT_QUESTION}`: Patient's question (from topic)
- `{REFERENCE_RESPONSE}`: Expected response (from ground_truth_field)
- `{RESPONSE}`: Agent's speech content

---

### 3. OpenQA (`domain: "openqa"`)
**Prompt File**: `prompts/eval_openqa_response.txt`

**Evaluated Metrics**:
- `relevance`: Addresses the question/task directly (1-5)
- `completeness`: Covers necessary points/details (1-5)
- `accuracy`: Information correctness vs expected answer (1-5)
- `clarity`: Organization and readability (1-5)
- `helpfulness`: Utility for a typical user (1-5)

**Required Config Fields**:
- `ground_truth_field`: "output" (expected answer)
- Dataset should include `system_prompt` field (optional)

**Template Placeholders**:
- `{SYSTEM_PROMPT}`: System instructions (from item.system_prompt)
- `{QUESTION}`: Question/task (from topic)
- `{EXPECTED_ANSWER}`: Expected answer (from ground_truth_field)
- `{RESPONSE}`: Agent's speech content

---

## Example Configuration

```json
{
  "dataset_path": "datasets/data/math_cleaned_250.json",
  "topic_field": "input",
  "context_field": "distill_llama_output",
  "ground_truth_field": "output",
  "domain": "math",
  "max_items": 3,
  "debate": {
    "max_rounds": 2,
    "roles_order": ["AFFIRMATIVE", "NEGATIVE"]
  },
  "generation_models": {
    "AFFIRMATIVE": {"provider": "ollama", "engine": "llama3.1"},
    "NEGATIVE": {"provider": "ollama", "engine": "llama3.1"}
  },
  "judge_model": {"provider": "ollama", "engine": "llama3.1"},
  "output_dir": "debate_outputs_math"
}
```

## Output Format

When domain-specific evaluation is enabled, each speech includes structured metrics:

```json
{
  "turn": 1,
  "role": "AFFIRMATIVE",
  "content": "The solution requires applying the Pythagorean theorem...",
  "scores": {
    "overall": 4,
    "reasoning": "Strong logical flow but minor arithmetic error in final step",
    "metrics": {
      "correctness": 3,
      "reasoning": 5,
      "completeness": 4,
      "accuracy": 3
    }
  }
}
```

## Implementation Details

### How It Works
1. **Loading**: When `domain` is specified in config, `orchestrator.py` calls `load_domain_eval_prompt()` to read the corresponding prompt file
2. **Templating**: `_build_judge_prompt()` replaces placeholders with dataset fields and debate context
3. **Parsing**: `judge_bridge.py` extracts both `<overall_score>` and individual `<metrics>` tags using regex
4. **Storage**: Metrics stored in `speech.scores['metrics']` dict for downstream analysis

### Key Files Modified
- `roles.py`: Added `load_domain_eval_prompt()` and `DOMAIN_EVAL_PROMPTS` mapping
- `judge_bridge.py`: Added `parse_metrics()`, `parse_overall_score()`, domain_eval parameter
- `orchestrator.py`: Added domain fields to `DebateConfig`, `_build_judge_prompt()` method
- `runner_debate_dataset.py`: Extract ground_truth, system_prompt, document fields from dataset items

## Benefits

1. **Consistency**: Uses same prompts as original Debatable-Intelligence judging
2. **Granularity**: Per-speech metric scores enable fine-grained analysis
3. **Comparability**: Results align with existing evaluation baselines
4. **Flexibility**: Easy to add new domains by creating prompt templates

## Future Enhancements

- **Aggregate Metrics**: Compute average metrics across all speeches in a debate
- **Metric Trends**: Track how metrics evolve over debate rounds
- **Ground Truth Validation**: Compare final debate consensus to expected answer
- **Metric Weighting**: Configure importance weights per metric for overall scoring
