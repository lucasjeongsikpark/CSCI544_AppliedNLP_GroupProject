"""Bridge utilities to reuse zero-shot judging prompts for interactive debate.

The judge model outputs a <score>1-5</score> and optional <scratchpad>reasoning</scratchpad>.
We parse these using same logic as in zero_shot_experiment.parse_response / get_reasoning.

Supports both generic debate judging and domain-specific evaluation (math/medical/openqa).
"""
from typing import Dict, Any, Optional
from src.model import LanguageModel
from debate_runtime.state import Speech

import re

SCORE_RE = re.compile(r"<score>\s*([1-5])\s*</score>")
SCRATCHPAD_RE = re.compile(r"<scratchpad>(.*?)</scratchpad>", re.DOTALL)
OVERALL_SCORE_RE = re.compile(r"<overall_score>\s*([1-5])\s*</overall_score>")
METRICS_RE = re.compile(r"<metrics>(.*?)</metrics>", re.DOTALL)


def parse_score(raw: str) -> int:
    """Parse score from judge output (generic <score> tag)."""
    m = SCORE_RE.search(raw)
    if m:
        return int(m.group(1))
    # fallback: search for standalone number 1-5
    for c in ['1','2','3','4','5']:
        if f"score{c}" in raw.lower() or raw.strip()==c:
            return int(c)
    return -1


def parse_overall_score(raw: str) -> int:
    """Parse overall_score from domain-specific evaluation prompts."""
    m = OVERALL_SCORE_RE.search(raw)
    if m:
        return int(m.group(1))
    return parse_score(raw)  # fallback to generic score parsing


def parse_metrics(raw: str) -> Dict[str, int]:
    """Parse individual metric scores from domain-specific evaluation (e.g., correctness: 4, reasoning: 5)."""
    metrics = {}
    m = METRICS_RE.search(raw)
    if not m:
        return metrics
    
    content = m.group(1)
    # Parse lines like "correctness: 4" or "medical_accuracy: 3"
    for line in content.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            metric_name = parts[0].strip()
            try:
                score_val = int(parts[1].strip())
                if 1 <= score_val <= 5:
                    metrics[metric_name] = score_val
            except ValueError:
                continue
    
    return metrics


def parse_reasoning(raw: str) -> str:
    m = SCRATCHPAD_RE.search(raw)
    return m.group(1).strip() if m else ''


def judge_speech(model: LanguageModel, judge_prompt: str, max_tokens: int = 512, temperature: float = 0.0, 
                 domain_eval: bool = False) -> Dict[str, Any]:
    """
    Generate judge evaluation for a speech.
    
    Args:
        model: Judge language model
        judge_prompt: Formatted prompt for evaluation
        max_tokens: Max generation length
        temperature: Sampling temperature
        domain_eval: If True, parse domain-specific metrics (math/med/openqa format)
    
    Returns:
        Dict with 'raw', 'score', 'reasoning', and optionally 'metrics' (for domain eval)
    """
    completion = model.generate(judge_prompt, max_tokens=max_tokens, temperature=temperature)['completion']
    
    if domain_eval:
        score = parse_overall_score(completion)
        metrics = parse_metrics(completion)
        reasoning = parse_reasoning(completion)
        return {
            'raw': completion,
            'score': score,
            'reasoning': reasoning,
            'metrics': metrics  # e.g., {'correctness': 4, 'reasoning': 5, ...}
        }
    else:
        score = parse_score(completion)
        reasoning = parse_reasoning(completion)
        return {'raw': completion, 'score': score, 'reasoning': reasoning}


def attach_judgment(speech: Speech, judgment: Dict[str, Any]) -> None:
    speech.scores = {
        'overall': judgment['score'],
        'reasoning': judgment['reasoning']
    }
    if 'metrics' in judgment:
        speech.scores['metrics'] = judgment['metrics']
