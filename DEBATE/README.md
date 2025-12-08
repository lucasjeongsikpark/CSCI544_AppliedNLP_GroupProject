# DEBATE Framework

A multi-agent debate-based evaluation framework for assessing the quality of LLM-generated responses using iterative critic-scorer interactions.

## Overview

The DEBATE (Debate-based Evaluation) framework uses a dual-agent approach to evaluate text generation outputs. A **Scorer** agent provides initial assessments, while a **Critic** agent plays devil's advocate to challenge and refine those evaluations through multiple rounds of debate. This iterative process helps produce more reliable and robust evaluation scores.

## Key Features

- **Multi-Agent Evaluation**: Leverages two specialized LLM agents (Scorer and Critic) for comprehensive assessment
- **Iterative Refinement**: Multiple debate rounds allow scores to be challenged and refined
- **Multi-Aspect Analysis**: Evaluate responses across multiple dimensions simultaneously
- **Flexible Configuration**: Customizable iteration limits and evaluation aspects
- **Domain Agnostic**: Applicable to medical, mathematical, open QA, and other domains

## Architecture

The framework consists of three main components:

1. **LLMCall**: Handles communication with the Ollama-hosted LLM (gemma2:2b by default)
2. **DEBATEFramework**: Orchestrates the debate between Scorer and Critic agents
3. **MultiAspectDEBATE**: Wrapper for evaluating multiple aspects in parallel

### Evaluation Process

```
1. Commander formulates evaluation prompt
2. Scorer provides initial assessment (Score 1-5 + explanation)
3. For each iteration (up to max_iterations):
   a. Critic reviews and challenges the score
   b. If Critic says "NO ISSUE", debate ends
   c. Otherwise, Scorer reconsiders based on feedback
4. Extract final numerical score (1-5 scale)
```

## Installation (Locally)

### Prerequisites

- Python 3.8+
- Ollama running locally on port 11434

### Setup

```bash
pip install ollama
```

Ensure Ollama is running with the gemma2:2b model:

```bash
ollama pull gemma2:2b
ollama serve
```

## Usage

### Basic Usage

```python
from debate import create_debate_evaluator

evaluator = create_debate_evaluator(max_iterations=3)

result = evaluator.evaluate_dialogue(
    context="Your source text or prompt",
    response="The generated response to evaluate",
    aspects=["naturalness", "coherence", "engagingness", "groundedness"]
)
```

### Medical Response Evaluation

```python
from debate import DEBATEFramework

debate = DEBATEFramework(max_iterations=3)

result = debate.evaluate(
    task="medical question answering",
    aspect="Medical Accuracy",
    source_text="Patient's question or context",
    generated_text="Model's medical response"
)

print(f"Final Score: {result['final_score']}")
print(f"Iterations: {result['iterations']}")
```

### Evaluation Aspects

The framework includes pre-defined aspect sets for different domains:

**Medical Evaluation (MED_ASPECTS)**:
- Medical Accuracy
- Appropriateness
- Safety
- Clarity
- Professionalism

**Math Problem Evaluation (MATH_ASPECTS)**:
- Correctness
- Reasoning
- Completeness
- Accuracy

**Open QA Evaluation (OPEN_QA_ASPECTS)**:
- Relevance
- Completeness
- Accuracy
- Clarity
- Helpfulness

### Example with Custom Aspects

```python
custom_aspects = [
    "Technical Accuracy",
    "Code Quality",
    "Documentation Completeness"
]

evaluator = create_debate_evaluator(max_iterations=3)
result = evaluator.evaluate_dialogue(
    context=code_problem,
    response=generated_solution,
    aspects=custom_aspects
)
```

## Configuration

### Model Settings

Edit the global variables in `debate.py`:

```python
_MODEL = "gemma2:2b"
_OLLAMA_HOST = "http://localhost:11434"
```

### Generation Parameters

Modify in the `LLMCall.generate()` method:

```python
options = {
    "temperature": 0.2,  # Lower = more deterministic
    "top_k": 3,          # Consider top 3 tokens
    "top_p": 0.5,        # Nucleus sampling threshold
}
```

## Output Format

Each evaluation returns a dictionary with:

```python
{
    "final_score": float,           # 1-5 scale
    "final_response": str,          # Scorer's final explanation
    "debate_history": [             # Record of all iterations
        {
            "iteration": int,
            "score_response": str,
            "feedback": str
        }
    ],
    "iterations": int               # Number of debate rounds
}
```

### Saving Results

```python
results = []

for item in dataset:
    result = evaluator.evaluate_dialogue(
        context=item['context'],
        response=item['response'],
        aspects=OPEN_QA_ASPECTS
    )
    results.append(result)

debate.save_results(results, "outputs.json")
```

## Files

- `debate.py`: Core framework implementation
- `usage.py`: Example usage with medical, math, and open QA datasets
- `open_qa_outputs.json`: Example outputs for open QA evaluation
- `math_outputs.json`: Example outputs for math problem evaluation
- `med_outputs.json`: Example outputs for medical question evaluation
- `status-report-deepseek.txt`: Status report for DeepSeek model experiments
- `status-report-gemma2.txt`: Status report for Gemma2 model experiments

## Score Scale

All evaluations use a consistent 1-5 scale:

- **1**: Very Poor
- **2**: Poor
- **3**: Fair
- **4**: Good
- **5**: Excellent

## Advanced Features

### Multi-Aspect Aggregation

```python
results = evaluator.evaluate_dialogue(context, response, aspects)

scores = {aspect: res['final_score'] for aspect, res in results.items()}
avg_score = sum(s for s in scores.values() if s is not None) / len([s for s in scores.values() if s is not None])
```

### Debate History Analysis

```python
for iteration in result['debate_history']:
    print(f"Iteration {iteration['iteration']}:")
    print(f"  Scorer: {iteration['score_response']}")
    print(f"  Critic: {iteration['feedback']}")
```
