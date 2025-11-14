# Evaluation Prompts Integration - Quick Reference

## Are the prompts being used? ✅ YES!

The three evaluation prompts are now **fully integrated** into the debate framework:

```
prompts/eval_math_response.txt     → domain: "math"
prompts/eval_medical_response.txt  → domain: "medical"  
prompts/eval_openqa_response.txt   → domain: "openqa"
```

## How to Enable

### Option 1: Generic Debate Judging (Original Behavior)
```json
{
  "topic": "Should universal basic income be implemented?",
  // NO domain field = generic debate evaluation
}
```
**Judge evaluates**: clarity, coherence, persuasion  
**Output**: Single overall score + reasoning

---

### Option 2: Domain-Specific Structured Evaluation (NEW)
```json
{
  "dataset_path": "datasets/data/math_cleaned_250.json",
  "domain": "math",              // ← Enables structured evaluation
  "ground_truth_field": "output"  // ← Required for comparison
}
```
**Judge evaluates**: correctness, reasoning, completeness, accuracy  
**Output**: Overall score + 4 individual metric scores + reasoning

---

## Example Outputs

### Math Domain
```json
{
  "turn": 1,
  "role": "AFFIRMATIVE",
  "content": "Using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a...",
  "scores": {
    "overall": 4,
    "reasoning": "Correct approach with minor arithmetic error",
    "metrics": {
      "correctness": 4,    // ← From eval_math_response.txt
      "reasoning": 5,
      "completeness": 4,
      "accuracy": 3
    }
  }
}
```

### Medical Domain
```json
{
  "scores": {
    "overall": 4,
    "reasoning": "Medically sound advice with appropriate safety warnings",
    "metrics": {
      "medical_accuracy": 5,      // ← From eval_medical_response.txt
      "appropriateness": 4,
      "safety": 5,
      "clarity": 4,
      "professionalism": 5
    }
  }
}
```

### OpenQA Domain
```json
{
  "scores": {
    "overall": 3,
    "reasoning": "Relevant answer but lacks detail",
    "metrics": {
      "relevance": 4,       // ← From eval_openqa_response.txt
      "completeness": 3,
      "accuracy": 4,
      "clarity": 3,
      "helpfulness": 3
    }
  }
}
```

---

## File Flow

```
Config: domain="math"
    ↓
roles.py: load_domain_eval_prompt("math")
    ↓
Loads: prompts/eval_math_response.txt
    ↓
orchestrator.py: _build_judge_prompt()
    ↓
Replaces: {INPUT}, {OUTPUT}, {RESPONSE}
    ↓
judge_bridge.py: judge_speech(domain_eval=True)
    ↓
Parses: <overall_score>, <metrics>
    ↓
Output: speech.scores with metrics dict
```

---

## Configuration Examples

### Math Dataset
```bash
python -m debate_runtime.runner_debate_dataset \
  --config debate_runtime/debate_dataset_config.example.json
```
**Uses**: `eval_math_response.txt` with correctness/reasoning/completeness/accuracy

### Medical Dataset
```bash
python -m debate_runtime.runner_debate_dataset \
  --config debate_runtime/debate_dataset_config_medical.example.json
```
**Uses**: `eval_medical_response.txt` with medical_accuracy/safety/professionalism

### OpenQA Dataset
```bash
python -m debate_runtime.runner_debate_dataset \
  --config debate_runtime/debate_dataset_config_openqa.example.json
```
**Uses**: `eval_openqa_response.txt` with relevance/completeness/helpfulness

---

## Benefits

✅ **Reuses existing prompts** - No duplication, maintains consistency  
✅ **Structured metrics** - Per-speech granular scoring  
✅ **Backward compatible** - Generic judging still works without domain field  
✅ **Easy to extend** - Add new domains by creating prompt files  
✅ **Ground truth aware** - Compares agent responses to expected answers  

---

## Next Steps

1. **Test with actual runs**: Execute configs to verify prompt integration
2. **Analyze metrics**: Build aggregation scripts to compute average scores
3. **Visualize trends**: Plot metric evolution across debate rounds
4. **Compare modes**: Run same dataset with/without domain evaluation
