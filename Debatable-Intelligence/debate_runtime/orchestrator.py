from __future__ import annotations
from typing import Dict, List, Optional, Any
import json
import re
from dataclasses import dataclass
"""Debate orchestration loop.

Relative imports expect this package to be executed from the Debatable-Intelligence root
using `python -m debate_runtime.runner_debate`. We therefore use absolute package
paths (`src.model`) referencing existing modules.
"""
from src.model import LanguageModel
from debate_runtime.roles import Role, load_domain_eval_prompt
from debate_runtime.state import DebateState, Speech
from debate_runtime.judge_bridge import judge_speech, attach_judgment

@dataclass
class DebateConfig:
    topic: str
    max_rounds: int = 4
    roles_order: List[str] = None  # sequence per round (e.g., AFFIRMATIVE, NEGATIVE)
    judge_role: str = 'JUDGE'
    temperature: float = 0.7
    max_tokens_generation: int = 512
    max_tokens_judge: int = 256
    initial_context: str = ''  # optional seed context (e.g. dataset distilled answer)
    secondary_context: str = ''  # optional second answer for comparative debate
    domain: Optional[str] = None  # domain for evaluation: 'math', 'medical', 'openqa'
    ground_truth: str = ''  # expected answer for domain-specific evaluation
    system_prompt: str = ''  # for openqa domain evaluation
    document: str = ''  # for medical domain evaluation

    def __post_init__(self):
        if self.roles_order is None:
            self.roles_order = ['AFFIRMATIVE', 'NEGATIVE']

class DebateOrchestrator:
    def __init__(self, config: DebateConfig, generation_models: Dict[str, LanguageModel], judge_model: LanguageModel, logger=None):
        self.config = config
        self.generation_models = generation_models  # map role -> model
        self.judge_model = judge_model
        self.logger = logger
        self.state = DebateState(topic=config.topic, max_rounds=config.max_rounds, 
                                  initial_context=config.initial_context,
                                  secondary_context=config.secondary_context)
        self.roles = {r: Role(r) for r in set(config.roles_order + [config.judge_role])}
        
        # Load domain-specific evaluation prompt if specified
        self.domain_eval_prompt = None
        if config.domain:
            self.domain_eval_prompt = load_domain_eval_prompt(config.domain)
            if self.domain_eval_prompt and self.logger:
                self.logger.info(f"Loaded domain-specific evaluation prompt for: {config.domain}")

    def _build_judge_prompt(self, speech_content: str) -> str:
        """Build judge evaluation prompt (generic or domain-specific)."""
        if self.domain_eval_prompt:
            # Use domain-specific template with placeholders
            prompt = self.domain_eval_prompt
            prompt = prompt.replace('{INPUT}', self.config.topic)
            prompt = prompt.replace('{OUTPUT}', self.config.ground_truth)
            prompt = prompt.replace('{RESPONSE}', speech_content)
            
            # Domain-specific fields
            if self.config.domain == 'medical':
                prompt = prompt.replace('{DOCUMENT}', self.config.document)
                prompt = prompt.replace('{PATIENT_QUESTION}', self.config.topic)
                prompt = prompt.replace('{REFERENCE_RESPONSE}', self.config.ground_truth)
            elif self.config.domain == 'openqa':
                prompt = prompt.replace('{SYSTEM_PROMPT}', self.config.system_prompt)
                prompt = prompt.replace('{QUESTION}', self.config.topic)
                prompt = prompt.replace('{EXPECTED_ANSWER}', self.config.ground_truth)
            elif self.config.domain == 'math':
                prompt = prompt.replace('{PROBLEM}', self.config.topic)
                prompt = prompt.replace('{EXPECTED_ANSWER}', self.config.ground_truth)
            
            return prompt
        else:
            # Generic debate judge prompt
            context = self.state.history_text()
            judge_prompt = self.roles[self.config.judge_role].render_prompt(self.config.topic, context)
            return judge_prompt + "\nLast speech to evaluate:\n" + speech_content

    def _determine_debate_winner(self) -> Dict[str, Any]:
        def _determine_debate_winner(self) -> Dict[str, Any] | None:
                """Use judge model to produce a JSON verdict selecting debate winner and scoring both answers.

                We bypass the generic judge_speech parser because we now expect structured JSON instead of
                <score>/<metrics> tags. Robustly extract the first JSON object from the model output.
                """
                if not self.config.initial_context or not self.config.secondary_context:
                        return None

                debate_summary = self.state.history_text()
                winner_prompt = f"""You are the impartial debate judge. Two debaters (AFFIRMATIVE and NEGATIVE) argued
about two candidate answers to the problem. Determine which DEBATER won the debate (AFFIRMATIVE or NEGATIVE)
based on argumentative quality in the debate history, NOT which underlying answer you prefer.

Then separately EVALUATE EACH ANSWER against the ground truth.

Provide a single JSON object ONLY (no prose before or after) with this schema:
{{
    "winner": "AFFIRMATIVE" | "NEGATIVE",
    "winner_reasoning": "Why this debater's argumentative performance was stronger.",
    "llama_output_scores": {{
        "correctness": 1-5,
        "reasoning": 1-5,
        "completeness": 1-5,
        "accuracy": 1-5
    }},
    "distill_llama_output_scores": {{
        "correctness": 1-5,
        "reasoning": 1-5,
        "completeness": 1-5,
        "accuracy": 1-5
    }}
}}

Problem Statement / Ground Truth:
{self.config.ground_truth}

ANSWER_A (distill_llama_output):
{self.config.initial_context}

ANSWER_B (llama_output):
{self.config.secondary_context}

Debate History:
{debate_summary}
"""

                raw = self.judge_model.generate(
                        winner_prompt,
                        max_tokens=self.config.max_tokens_judge * 3,
                        temperature=0.0
                )["completion"]

                # Attempt to extract JSON object
                json_candidate = None
                try:
                        # Direct attempt
                        json_candidate = raw.strip()
                        # If model added text before/after, use regex to pull first {...}
                        if not json_candidate.startswith('{'):
                                m = re.search(r"\{.*\}", raw, re.DOTALL)
                                if m:
                                        json_candidate = m.group(0)
                        verdict = json.loads(json_candidate)
                except Exception:
                        if self.logger:
                                self.logger.warning("Failed to parse winner JSON from judge output; raw output logged.")
                                self.logger.debug(f"Judge raw winner output: {raw}")
                        return None

                # Basic validation & normalization
                required_top = {"winner", "llama_output_scores", "distill_llama_output_scores"}
                if not required_top.issubset(verdict.keys()):
                        if self.logger:
                                self.logger.warning(f"Winner JSON missing required keys: {required_top - set(verdict.keys())}")
                        return None

                if self.logger:
                        self.logger.info(f"Debate Winner: {verdict.get('winner', 'N/A')}")
                return verdict

        # ---------------- Fallback utilities -----------------
    def _fallback_winner_from_speeches(self) -> Dict[str, Any]:
        """Last-resort winner: uses cumulative scores but crafts role-focused reasoning without answer labels."""
        scores = {role: 0 for role in ['AFFIRMATIVE', 'NEGATIVE']}
        counts = {role: 0 for role in ['AFFIRMATIVE', 'NEGATIVE']}
        for sp in self.state.speeches:
            if sp.role in scores and sp.scores:
                scores[sp.role] += sp.scores.get('overall', 0)
                counts[sp.role] += 1
        winner = 'AFFIRMATIVE' if scores['AFFIRMATIVE'] >= scores['NEGATIVE'] else 'NEGATIVE'
        if scores['AFFIRMATIVE'] == scores['NEGATIVE']:
            reasoning = "Tie on numeric scores; deterministic selection favors AFFIRMATIVE. Improve differentiation via stronger rebuttals and clearer logical scaffolding."
        else:
            if winner == 'AFFIRMATIVE':
                reasoning = (f"AFFIRMATIVE demonstrated clearer structure and more targeted critiques (scores AFFIRMATIVE={scores['AFFIRMATIVE']}, NEGATIVE={scores['NEGATIVE']}).")
            else:
                reasoning = (f"NEGATIVE provided more incisive counter-arguments and logical challenges (scores AFFIRMATIVE={scores['AFFIRMATIVE']}, NEGATIVE={scores['NEGATIVE']}).")
        return {'winner': winner, 'winner_reasoning': reasoning}

    def _evaluate_answer(self, answer_text: str) -> Dict[str, Any]:
        """Domain evaluation of an answer using existing domain_eval_prompt pattern.
        Returns metrics dict WITHOUT explanation text (only numeric scores).
        If unavailable, returns empty dict.
        """
        if not answer_text:
            return {}
        if not self.domain_eval_prompt:
            return {}
        prompt = self.domain_eval_prompt.replace('{INPUT}', self.config.topic)
        prompt = prompt.replace('{OUTPUT}', self.config.ground_truth)
        prompt = prompt.replace('{RESPONSE}', answer_text)
        judgment = judge_speech(self.judge_model, prompt, max_tokens=self.config.max_tokens_judge, temperature=0.0, domain_eval=True)
        metrics = judgment.get('metrics', {})
        # Return only numeric metrics, no explanation
        return {k: v for k, v in metrics.items()}

    def _judge_winner_from_debate(self) -> Dict[str, Any] | None:
        """Judge selects winner focusing on role argumentative quality; avoids referencing raw answers directly."""
        debate_summary = self.state.history_text()
        prompt = (
            "You are the debate judge. Choose which DEBATER (AFFIRMATIVE or NEGATIVE) had superior argumentative performance.\n"
            "Assess: clarity, logical structure, rigor of critique, rebuttal effectiveness, evidence use, and coherence.\n"
            "Do NOT mention literal strings 'Answer A', 'Answer B', 'ANSWER_A', 'ANSWER_B'. Refer only to roles and their critiques.\n"
            "Return ONLY these XML tags (no extra text):\n"
            "<winner>AFFIRMATIVE or NEGATIVE</winner>\n"
            "<scratchpad>Concise justification referencing AFFIRMATIVE/NEGATIVE speeches and their argumentative strengths; no raw answer labels.</scratchpad>\n\n"
            f"Debate History (truncated window):\n{debate_summary}\n"
        )
        judgment = judge_speech(self.judge_model, prompt, max_tokens=self.config.max_tokens_judge, temperature=0.0, domain_eval=False)
        raw = judgment.get('raw', '')
        m = re.search(r"<winner>\s*(AFFIRMATIVE|NEGATIVE)\s*</winner>", raw, re.IGNORECASE)
        if not m:
            return None
        winner = m.group(1).upper()
        winner_reason = judgment.get('reasoning', '')
        # Sanitize any accidental references
        winner_reason = re.sub(r"ANSWER[_ ]?A|ANSWER[_ ]?B|Answer A|Answer B", "", winner_reason)
        winner_reason = winner_reason.strip()
        return {'winner': winner, 'winner_reasoning': winner_reason or f"{winner} provided more coherent, well-structured critiques."}


    def run(self):
        import time
        start_time = time.time()
        
        # Run debate rounds (no pre-debate evaluation)
        turn = 0
        for round_idx in range(self.config.max_rounds):
            for role_name in self.config.roles_order:
                turn += 1
                model = self.generation_models[role_name]
                prompt = self.roles[role_name].render_prompt(self.config.topic, self.state.history_text())
                completion = model.generate(prompt, max_tokens=self.config.max_tokens_generation, temperature=self.config.temperature)['completion']
                speech = Speech(turn=turn, role=role_name, content=completion)
                self.state.add_speech(speech)

                # judge after each speech (domain-specific if configured)
                judge_prompt = self._build_judge_prompt(completion)
                judgment = judge_speech(self.judge_model, judge_prompt, 
                                       max_tokens=self.config.max_tokens_judge, 
                                       temperature=0.0,
                                       domain_eval=(self.domain_eval_prompt is not None))
                attach_judgment(speech, judgment)

                if self.logger:
                    score_info = f"Score {speech.scores['overall']}"
                    if 'metrics' in speech.scores:
                        metrics_str = ', '.join(f"{k}:{v}" for k, v in speech.scores['metrics'].items())
                        score_info += f" [{metrics_str}]"
                    self.logger.info(f"Round {round_idx+1} Turn {turn} Role {role_name} {score_info}")

        # After debate completes, determine winner
        verdict = self._determine_debate_winner()
        if verdict:
            # Structured JSON path succeeded
            self.state.debate_winner = verdict.get('winner', '')
            if verdict.get('winner_reasoning'):
                self.state.meta['winner_reasoning'] = verdict.get('winner_reasoning')
            self.state.llama_output_scores = verdict.get('llama_output_scores', {})
            self.state.distill_llama_output_scores = verdict.get('distill_llama_output_scores', {})
        else:
            # Try judge-based winner selection using tags
            judge_verdict = self._judge_winner_from_debate()
            if judge_verdict:
                self.state.debate_winner = judge_verdict.get('winner', '')
                self.state.meta['winner_reasoning'] = judge_verdict.get('winner_reasoning', '')
            else:
                # Fallback to cumulative scores
                fallback = self._fallback_winner_from_speeches()
                self.state.debate_winner = fallback.get('winner', '')
                self.state.meta['winner_reasoning'] = fallback.get('winner_reasoning', '')
            # Evaluate both answers regardless of winner path
            self.state.llama_output_scores = self._evaluate_answer(self.config.secondary_context)
            self.state.distill_llama_output_scores = self._evaluate_answer(self.config.initial_context)

        # Track elapsed time
        self.state.elapsed_time = time.time() - start_time
        
        return self.state
