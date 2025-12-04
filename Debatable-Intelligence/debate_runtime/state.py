from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class Speech:
    turn: int
    role: str
    content: str
    scores: Dict[str, Any] | None = None  # per-dimension scores

@dataclass
class DebateState:
    topic: str
    max_rounds: int
    initial_context: str = ''  # optional seed context (e.g., distilled answer or reference solution)
    secondary_context: str = ''  # optional second answer for comparative debate
    speeches: List[Speech] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation tracking fields
    # Winner's evaluation of each answer separately (score1 & score2 in output JSON)
    llama_output_scores: Optional[Dict[str, Any]] = None  # winner's evaluation of llama_output (ANSWER_B)
    distill_llama_output_scores: Optional[Dict[str, Any]] = None  # winner's evaluation of distill_llama_output (ANSWER_A)
    attempts: int = 1
    elapsed_time: float = 0.0
    
    # Winner tracking
    debate_winner: str = ''  # AFFIRMATIVE or NEGATIVE

    def add_speech(self, speech: Speech) -> None:
        self.speeches.append(speech)

    def history_text(self) -> str:
        lines = []
        if self.initial_context and self.secondary_context:
            lines.append(f"<ANSWER_A>\n{self.initial_context}\n</ANSWER_A>")
            lines.append(f"<ANSWER_B>\n{self.secondary_context}\n</ANSWER_B>")
        elif self.initial_context:
            lines.append(f"<INITIAL_CONTEXT>\n{self.initial_context}\n</INITIAL_CONTEXT>")
        for s in self.speeches:
            lines.append(f"Turn {s.turn} [{s.role}]:\n{s.content}\n")
        return "\n".join(lines[-12:])  # window to control context length (after initial context)

    def to_json(self) -> str:
        """Output format with debate winner and winning answer's scores."""
        output = {
            'chat_log': {}
        }
        
        # Build chat_log with llama_output and distill_llama_output sub-objects
        if self.secondary_context:  # llama_output (ANSWER_B)
            output['chat_log']['llama_output'] = {
                'response': self.secondary_context,
                'reasoning': ''  # No pre-debate reasoning
            }
        
        if self.initial_context:  # distill_llama_output (ANSWER_A)
            output['chat_log']['distill_llama_output'] = {
                'response': self.initial_context,
                'reasoning': ''  # No pre-debate reasoning
            }
        
        # Add debate speeches to chat_log
        output['chat_log']['debate_speeches'] = [
            {
                'turn': s.turn,
                'role': s.role,
                'content': s.content
            }
            for s in self.speeches
        ]
        
        # Add winner information
        if self.debate_winner:
            output['winner'] = self.debate_winner
        
        # Add score1 and score2 (winner's evaluation of both answers)
        if self.llama_output_scores:
            output['score1'] = self.llama_output_scores
        
        if self.distill_llama_output_scores:
            output['score2'] = self.distill_llama_output_scores
        
        # Add attempts and elapsed_time
        output['attempts'] = self.attempts
        output['elapsed_time'] = self.elapsed_time
        
        # Add metadata
        if self.meta:
            output['meta'] = self.meta
        
        return json.dumps(output, indent=2)
