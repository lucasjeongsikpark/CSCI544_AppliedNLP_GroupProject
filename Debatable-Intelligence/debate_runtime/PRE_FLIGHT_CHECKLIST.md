# Pre-Flight Checklist - Ready to Run

## ✅ Output Schema Validation

### Required Fields (All Present)
1. ✅ **chat_log**: Contains all conversation data
   - ✅ `llama_output`: ANSWER_B with response + reasoning
   - ✅ `distill_llama_output`: ANSWER_A with response + reasoning
   - ✅ `debate_speeches`: Turn-by-turn debate speeches with scores

2. ✅ **score1**: Individual metrics for `llama_output` (ANSWER_B)
   - Metrics: Correctness, Reasoning, Completeness, Accuracy (domain-specific)
   - Includes reasoning text

3. ✅ **score2**: Individual metrics for `distill_llama_output` (ANSWER_A)
   - Metrics: Correctness, Reasoning, Completeness, Accuracy (domain-specific)
   - Includes reasoning text

4. ✅ **attempts**: Number of generation attempts (default: 1)

5. ✅ **elapsed_time**: Total runtime in seconds

### Sample Output Verified
```json
{
  "chat_log": {
    "llama_output": {"response": "...", "reasoning": "..."},
    "distill_llama_output": {"response": "...", "reasoning": "..."},
    "debate_speeches": [{"turn": 1, "role": "AFFIRMATIVE", ...}]
  },
  "score1": {"Correctness": 5, "Reasoning": 4, ...},
  "score2": {"Correctness": 5, "Reasoning": 4, ...},
  "attempts": 1,
  "elapsed_time": 10.5
}
```

---

## ✅ Domain-Specific Evaluation Integration

### Math Domain
- ✅ Prompt: `prompts/eval_math_response.txt`
- ✅ Metrics: correctness, reasoning, completeness, accuracy
- ✅ Config: `debate_dataset_config.example.json`
- ✅ Dataset: `datasets/data/math_cleaned_250.json`

### Medical Domain
- ✅ Prompt: `prompts/eval_medical_response.txt`
- ✅ Metrics: medical_accuracy, appropriateness, safety, clarity, professionalism
- ✅ Config: `debate_dataset_config_medical.example.json`
- ✅ Dataset: `datasets/data/med_cleaned.json`

### OpenQA Domain
- ✅ Prompt: `prompts/eval_openqa_response.txt`
- ✅ Metrics: relevance, completeness, accuracy, clarity, helpfulness
- ✅ Config: `debate_dataset_config_openqa.example.json`
- ✅ Dataset: `datasets/data/openQA_cleaned_250.json`

---

## ✅ Dual-Answer Comparative Debate

### Configuration
- ✅ `context_field`: "distill_llama_output" → ANSWER_A
- ✅ `secondary_context_field`: "llama_output" → ANSWER_B
- ✅ Agents see both answers and can:
  - Defend one
  - Critique both
  - Synthesize best reasoning

### Evaluation Flow
1. ✅ **Pre-Debate**: Judge evaluates ANSWER_A and ANSWER_B separately
   - Stores metrics in `score2` (distill_llama_output)
   - Stores metrics in `score1` (llama_output)

2. ✅ **During Debate**: Agents debate with both answers as context
   - Each speech evaluated by judge
   - Scores attached to debate_speeches

3. ✅ **Post-Debate**: Complete output with all metrics saved

---

## ✅ Code Quality

### Error Checking
- ✅ `state.py`: No errors
- ✅ `orchestrator.py`: No errors
- ✅ `judge_bridge.py`: No errors
- ✅ `roles.py`: No errors
- ✅ `runner_debate_dataset.py`: No errors

### JSON Validation
- ✅ `debate_config.example.json`: Valid
- ✅ `debate_dataset_config.example.json`: Valid
- ✅ `debate_dataset_config_medical.example.json`: Valid
- ✅ `debate_dataset_config_openqa.example.json`: Valid

### Schema Test
- ✅ `test_output_schema.py`: PASSED
- ✅ All required fields present
- ✅ Structure matches sample data

---

## 🚀 Ready to Run Commands

### Math Dataset (3 items)
```bash
cd /Users/ketan.joshi/USC/CSCI544/CSCI544_AppliedNLP_GroupProject/Debatable-Intelligence

python -m debate_runtime.runner_debate_dataset \
  --config debate_runtime/debate_dataset_config.example.json
```

**Output**: `debate_outputs_dataset/debate_0.json`, `debate_1.json`, `debate_2.json`

### Medical Dataset (3 items)
```bash
python -m debate_runtime.runner_debate_dataset \
  --config debate_runtime/debate_dataset_config_medical.example.json
```

**Output**: `debate_outputs_medical/debate_0.json`, `debate_1.json`, `debate_2.json`

### OpenQA Dataset (3 items)
```bash
python -m debate_runtime.runner_debate_dataset \
  --config debate_runtime/debate_dataset_config_openqa.example.json
```

**Output**: `debate_outputs_openqa/debate_0.json`, `debate_1.json`, `debate_2.json`

---

## 📊 What Gets Evaluated

### For Each Dataset Item:

1. **ANSWER_A** (distill_llama_output)
   - Evaluated against ground truth
   - Metrics stored in `score2`

2. **ANSWER_B** (llama_output)
   - Evaluated against ground truth
   - Metrics stored in `score1`

3. **Debate Speeches**
   - AFFIRMATIVE and NEGATIVE debate both answers
   - Each speech evaluated by judge
   - Scores attached to each turn

### Output Contains:
- Original answers (llama_output + distill_llama_output)
- Individual metric scores for both answers (score1 + score2)
- Full debate transcript with per-speech evaluation
- Reasoning for all evaluations
- Elapsed time and attempts

---

## ⚠️ Prerequisites

### Ollama Required
Ensure Ollama is running with `llama3.1` model:
```bash
ollama list | grep llama3.1
# Should show: llama3.1:latest
```

### Dataset Paths
Verify datasets exist:
```bash
ls -lh datasets/data/math_cleaned_250.json
ls -lh datasets/data/med_cleaned.json
ls -lh datasets/data/openQA_cleaned_250.json
```

### Prompt Files
Verify evaluation prompts exist:
```bash
ls -lh prompts/eval_*.txt
# Should show 3 files: eval_math_response.txt, eval_medical_response.txt, eval_openqa_response.txt
```

---

## 🎯 Expected Results

### Per-Item Output Structure
Each `debate_X.json` will contain:
- **chat_log**: Both candidate answers + debate transcript
- **score1**: Metrics for llama_output (Correctness, Reasoning, etc.)
- **score2**: Metrics for distill_llama_output (Correctness, Reasoning, etc.)
- **attempts**: 1
- **elapsed_time**: ~10-30 seconds per item

### Logs
Check `debate_outputs_*/logs/` for:
- Initial answer evaluations
- Turn-by-turn debate scores
- Runtime information

---

## ✅ All Systems Go!

**Ready to execute debate framework with:**
- ✅ Correct output schema matching sample data
- ✅ Domain-specific evaluation (math/medical/openqa)
- ✅ Dual-answer comparative debate
- ✅ Pre-debate answer evaluation (score1 + score2)
- ✅ Full chat log with reasoning
- ✅ Elapsed time tracking
- ✅ All configs validated

**No blockers detected. Framework ready for execution! 🚀**
