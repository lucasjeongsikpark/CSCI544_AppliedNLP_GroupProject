import enum
import json
from typing import Dict, List, Optional
import re
import ollama


class Role(enum.Enum):
    SCORER = 0
    CRITIC = 1

_MODEL = "gemma2:2b"


class LLMCall:
    def __init__(self, role: Role):
        self._model = _MODEL
        self.role = role
        
    def generate(self, prompt: str):
        # TODO(brendan): make this write to txt file or something
        response_generate = ollama.generate(model=self._model, prompt=prompt)
        result = response_generate['response']
        return result


class DEBATEFramework:
    def __init__(
        self,
        max_iterations: int = 3
    ):
        self.scorer = LLMCall(role=Role.SCORER)
        self.critic = LLMCall(role=Role.CRITIC)
        self.max_iterations = max_iterations
        
    def _get_devil_advocate_prompt(self) -> str:
        return (
            "Your role is to play a Devil's Advocate. Your logic has to be step-by-step. "
            "Critically review the score provided and assess whether the score is accurate. "
            "If you don't think that the score is accurate, criticize the score. "
            "Try to criticize the score as much as possible."
        )
    
    def _commander_formulate_prompt(
        self,
        task: str,
        aspect: str,
        source_text: str,
        generated_text: str
    ) -> str:
        prompt = f"""Task: {task}
Evaluation Aspect: {aspect}

Source Text:
{source_text}

Generated Text:
{generated_text}

Please evaluate the generated text for {aspect}. Let's think step by step."""
        return prompt
    
    def _scorer_initial_prompt(self, base_prompt: str) -> str:
        return f"""{base_prompt}

Provide a score from 1 to 5 where:
1 = Very Poor
2 = Poor
3 = Fair
4 = Good
5 = Excellent

Format your response as: Score: [number]
Explanation: [your reasoning]"""
    
    def _commander_send_to_critic(
        self,
        base_prompt: str,
        score_response: str
    ) -> str:
        devil_advocate = self._get_devil_advocate_prompt()
        return f"""{base_prompt}

The Scorer provided the following assessment:
{score_response}

{devil_advocate}

If you find no issues, respond with "NO ISSUE". Otherwise, provide your criticism."""
    
    def _commander_send_to_scorer(
        self,
        base_prompt: str,
        feedback: str,
        previous_score: str
    ) -> str:
        return f"""{base_prompt}

Your previous assessment:
{previous_score}

The Critic provided the following feedback:
{feedback}

Please reconsider your score based on this feedback. Let's think step by step.

Provide a score from 1 to 5 where:
1 = Very Poor
2 = Poor
3 = Fair
4 = Good
5 = Excellent

Format your response as: Score: [number]
Explanation: [your reasoning]"""
    
    def _extract_score(self, response: str) -> Optional[float]:
        patterns = [
            r'Score:\s*(\d+\.?\d*)',
            r'score:\s*(\d+\.?\d*)',
            r'Score\s*=\s*(\d+\.?\d*)',
            r'score\s*=\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*out of 5',
            r'(\d+\.?\d*)/5'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    if 1 <= score <= 5:
                        return score
                except ValueError:
                    continue
        
        return None
    
    def save_results(self, results: List[Dict], output_file: str):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def evaluate(
        self,
        task: str,
        aspect: str,
        source_text: str,
        generated_text: str
    ) -> Dict:
        base_prompt = self._commander_formulate_prompt(
            task, aspect, source_text, generated_text
        )
        
        scorer_prompt = self._scorer_initial_prompt(base_prompt)
        score_response = self.scorer.generate(scorer_prompt)
        
        debate_history = []
        
        for iteration in range(self.max_iterations):
            critic_prompt = self._commander_send_to_critic(
                base_prompt, score_response
            )
            feedback = self.critic.generate(critic_prompt)
            
            debate_history.append({
                'iteration': iteration + 1,
                'score_response': score_response,
                'feedback': feedback
            })
            
            if 'NO ISSUE' in feedback.upper():
                break
            
            revised_scorer_prompt = self._commander_send_to_scorer(
                base_prompt, feedback, score_response
            )
            score_response = self.scorer.generate(revised_scorer_prompt)
        
        final_score = self._extract_score(score_response)
        
        return {
            'final_score': final_score,
            'final_response': score_response,
            'debate_history': debate_history,
            'iterations': len(debate_history)
        }


class MultiAspectDEBATE:
    def __init__(
        self,
        max_iterations: int = 3
    ):
        self.debate = DEBATEFramework(max_iterations)
    
    def evaluate_dialogue(
        self,
        context: str,
        response: str,
        aspects: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        if aspects is None:
            aspects = ['naturalness', 'coherence', 'engagingness', 'groundedness']
        
        results = {}
        for aspect in aspects:
            result = self.debate.evaluate(
                task='dialogue generation',
                aspect=aspect,
                source_text=context,
                generated_text=response
            )
            results[aspect] = result
        
        return results


def create_debate_evaluator(
    max_iterations: int = 3
) -> MultiAspectDEBATE:
    return MultiAspectDEBATE(max_iterations)

