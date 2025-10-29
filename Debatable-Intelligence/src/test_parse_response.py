from dataset_experiment import parse_response

def test_multi_metric():
    resp = """
<metrics>
correctness: 5
reasoning: 4
completeness: 4
accuracy: 5
</metrics>
<overall_score>
5
</overall_score>
<scratchpad>
All steps shown; minor brevity in reasoning.
</scratchpad>
""".strip()
    overall, metrics = parse_response(resp)
    assert overall == 5
    assert metrics['correctness'] == 5
    assert metrics['reasoning'] == 4
    assert metrics['accuracy'] == 5

def test_legacy():
    resp = "<score>4</score>"
    overall, metrics = parse_response(resp)
    assert overall == 4
    assert metrics == {}

if __name__ == '__main__':
    test_multi_metric(); test_legacy(); print('parse_response smoke tests passed')