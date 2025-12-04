"""Test script to verify output schema matches expected format."""
import json
from debate_runtime.state import DebateState, Speech

def test_output_schema():
    """Verify output JSON contains all required fields."""
    
    # Create a sample debate state
    state = DebateState(
        topic="Test problem",
        max_rounds=2,
        initial_context="Answer A: The solution is 42",
        secondary_context="Answer B: The solution involves calculating 6*7"
    )
    
    # Add sample speech
    speech = Speech(turn=1, role="AFFIRMATIVE", content="I believe answer A is correct...")
    speech.scores = {
        'overall': 4,
        'reasoning': 'Strong argument',
        'metrics': {'correctness': 4, 'reasoning': 5, 'completeness': 4}
    }
    state.add_speech(speech)
    
    # Set evaluation scores
    state.llama_output_scores = {
        'Correctness': 5,
        'Reasoning': 4,
        'Completeness': 5,
        'Accuracy': 5,
        'reasoning': 'Judge reasoning for llama_output'
    }
    
    state.distill_llama_output_scores = {
        'Correctness': 5,
        'Reasoning': 4,
        'Completeness': 5,
        'Accuracy': 5,
        'reasoning': 'Judge reasoning for distill_llama_output'
    }
    
    state.attempts = 1
    state.elapsed_time = 10.5
    
    # Generate JSON output
    output_json = state.to_json()
    output = json.loads(output_json)
    
    # Verify required fields
    required_fields = ['chat_log', 'score1', 'score2', 'attempts', 'elapsed_time']
    for field in required_fields:
        assert field in output, f"Missing required field: {field}"
    
    # Verify chat_log structure
    assert 'llama_output' in output['chat_log'], "Missing chat_log.llama_output"
    assert 'distill_llama_output' in output['chat_log'], "Missing chat_log.distill_llama_output"
    assert 'debate_speeches' in output['chat_log'], "Missing chat_log.debate_speeches"
    
    # Verify llama_output structure
    assert 'response' in output['chat_log']['llama_output'], "Missing llama_output.response"
    assert 'reasoning' in output['chat_log']['llama_output'], "Missing llama_output.reasoning"
    
    # Verify distill_llama_output structure
    assert 'response' in output['chat_log']['distill_llama_output'], "Missing distill_llama_output.response"
    assert 'reasoning' in output['chat_log']['distill_llama_output'], "Missing distill_llama_output.reasoning"
    
    # Verify score1 structure (llama_output metrics)
    assert 'Correctness' in output['score1'], "Missing score1.Correctness"
    assert 'Reasoning' in output['score1'], "Missing score1.Reasoning"
    assert 'reasoning' in output['score1'], "Missing score1.reasoning"
    
    # Verify score2 structure (distill_llama_output metrics)
    assert 'Correctness' in output['score2'], "Missing score2.Correctness"
    assert 'Reasoning' in output['score2'], "Missing score2.Reasoning"
    assert 'reasoning' in output['score2'], "Missing score2.reasoning"
    
    # Verify attempts and elapsed_time
    assert output['attempts'] == 1, "Incorrect attempts value"
    assert output['elapsed_time'] == 10.5, "Incorrect elapsed_time value"
    
    print("✓ All required fields present")
    print("\nSample output structure:")
    print(json.dumps(output, indent=2))
    
    return True

if __name__ == '__main__':
    test_output_schema()
    print("\n✓ Output schema validation PASSED")
