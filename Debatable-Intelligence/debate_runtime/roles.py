from dataclasses import dataclass
from typing import Dict, Optional
import os

# Simple role prompt templates; can be expanded or replaced by external prompt files.
ROLE_TEMPLATES: Dict[str, str] = {
    "AFFIRMATIVE": "You are the AFFIRMATIVE debater. Advocate for the proposition: {TOPIC}. When multiple candidate answers are provided (ANSWER_A, ANSWER_B), you may defend one or synthesize the best reasoning. Respond to prior arguments while strengthening your case.",
    "NEGATIVE": "You are the NEGATIVE debater. Challenge the proposition: {TOPIC}. When multiple candidate answers are provided (ANSWER_A, ANSWER_B), you may critique both or identify flaws in reasoning. Rebut prior affirmative points and introduce counter-arguments.",
    "MODERATOR": "You are the MODERATOR. Keep debate focused on {TOPIC}. Summarize key points and request clarification when arguments are vague.",
    "JUDGE": "You are the JUDGE. Evaluate clarity, keypoint coverage, coherence, factuality, style, and persuasion for the last submitted speech on {TOPIC}. Provide <scratchpad>reasoning</scratchpad> and a <score>1-5</score>.",
}

# Domain-specific evaluation prompt file paths (relative to Debatable-Intelligence/)
DOMAIN_EVAL_PROMPTS: Dict[str, str] = {
    "math": "prompts/eval_math_response.txt",
    "medical": "prompts/eval_medical_response.txt",
    "openqa": "prompts/eval_openqa_response.txt",
}


def load_domain_eval_prompt(domain: str, base_path: str = ".") -> Optional[str]:
    """Load domain-specific evaluation prompt template if available."""
    if domain.lower() not in DOMAIN_EVAL_PROMPTS:
        return None
    
    prompt_path = os.path.join(base_path, DOMAIN_EVAL_PROMPTS[domain.lower()])
    if not os.path.exists(prompt_path):
        return None
    
    with open(prompt_path, 'r') as f:
        return f.read().strip()

@dataclass
class Role:
    name: str

    def render_prompt(self, topic: str, context: str) -> str:
        base = ROLE_TEMPLATES.get(self.name.upper(), "Respond appropriately.")
        return base.replace('{TOPIC}', topic) + "\nContext so far:\n" + context.strip()
