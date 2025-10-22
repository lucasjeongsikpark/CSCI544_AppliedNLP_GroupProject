import re
import time
import ollama
from datasets import load_dataset
import json


def llm_call(prompt: str, role: str) -> str:
    print(f"\n----- LLM CALL ({role.upper()}) to llama3:8b -----")
    print(f"prompt sent:\n---\n{prompt[:300]}...\n---\n")
    
    try:
        response = ollama.chat(
            model='llama3:8b', 
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content']
        print("LLM Response Received.")
        return content
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return "Error: LLM call failed."
    
ADVOCATE_INITIAL_PROMPT = """
**Your Role:** You are an expert advocate. Your team's goal is to build the strongest possible argument to defend a given answer.

**The Question:**
"{question}"

**The Answer Your Team MUST Defend:**
"{answer_to_defend}"

**Your Task:**
Generate a strong, initial argument. Focus on one or two key points to start the debate.

**Begin. Your Initial Argument:**
"""

ADVOCATE_FINAL_PROMPT = """
**Your Role:** You are part of a team of expert advocates. Your team's goal is to collaboratively build the strongest possible argument to defend a given answer.

**The Question:**
"{question}"

**The Answer Your Team MUST Defend:**
"{answer_to_defend}"

**Your Teammate's Initial Argument:**
"{teammate_argument}"

**Instructions for You:**
1. Read your teammate's argument carefully.
2. **Do not simply repeat their points.**
3. **Strengthen the case by:** adding new evidence, refining the logic, and addressing potential weaknesses.
4. Integrate your ideas with your teammate's to produce a single, cohesive, and more powerful **Final Argument**.

**Begin. Your Final Argument:**
"""


JUDGE_PROMPT = """
**Your Role:** You are a fair and impartial judge. Your task is to evaluate the arguments presented by two advocate teams.

**The Question:**
"{question}"

---
**Argument for Answer 1 ("{answer1}"):**
{final_argument_A}
---
**Argument for Answer 2 ("{answer2}"):**
{final_argument_B}
---

**Your Evaluation Criteria:**
Score each argument on a scale of 1-20 for each of the six criteria:
1. Relevance to the question
2. Accuracy of information
3. Depth of analysis
4. Clarity of expression
5. Strength of reasoning
6. Effectiveness in addressing opponent’s points

**Your Output Format:**
Provide your response in the exact structured format below.

**Detailed Scores:**
- Relevance: [score_A, score_B]
- Accuracy: [score_A, score_B]
- Depth: [score_A, score_B]
- Clarity: [score_A, score_B]
- Reasoning: [score_A, score_B]
- Addressing Counterarguments: [score_A, score_B]

**Comprehensive Feedback (50 words each):**
- Feedback for Advocate Team 1:
- Feedback for Advocate Team 2:

**Final Score Tally (sum of the above):**
(total_score_A, total_score_B)
"""


def run_more_debate(question, answer1, answer2):
    
    print("="*50)
    print(" memulai DEBATE")
    print("="*50)
    print(f"Question: {question}\n")

    print("\n--- Team A is building its argument... ---")
    # A1
    prompt_A1 = ADVOCATE_INITIAL_PROMPT.format(question=question, answer_to_defend=answer1)
    argument_A1 = llm_call(prompt_A1, role="advocate_initial")
    # A2
    prompt_A2 = ADVOCATE_FINAL_PROMPT.format(question=question, answer_to_defend=answer1, teammate_argument=argument_A1)
    final_argument_A = llm_call(prompt_A2, role="advocate_final")
    print(f"\nTeam A's Final Argument:\n{final_argument_A}")

    print("\n--- Team B is building its argument... ---")
    # B1
    prompt_B1 = ADVOCATE_INITIAL_PROMPT.format(question=question, answer_to_defend=answer2)
    argument_B1 = llm_call(prompt_B1, role="advocate_initial")
    # B2
    prompt_B2 = ADVOCATE_FINAL_PROMPT.format(question=question, answer_to_defend=answer2, teammate_argument=argument_B1)
    final_argument_B = llm_call(prompt_B2, role="advocate_final")
    print(f"\nTeam B's Final Argument:\n{final_argument_B}")

    print("\n--- The Judge is evaluating the arguments... ---")
    judge_prompt = JUDGE_PROMPT.format(
        question=question,
        answer1=answer1,
        final_argument_A=final_argument_A,
        answer2=answer2,
        final_argument_B=final_argument_B
    )
    judge_response = llm_call(judge_prompt, role="judge")
    print("\nJudge's Full Response:")
    print(judge_response)

    print("\n--- Parsing the final results... ---")
    try:
        last_paren_match = re.search(r".*\((.*)\)", judge_response, re.DOTALL)
        if not last_paren_match:
            return "parse_error_no_paren"

        # Inside that last parenthesis, find the first two numbers
        content_in_paren = last_paren_match.group(1)
        numbers = re.findall(r"\d+", content_in_paren)
            
        if len(numbers) < 2:
            return "parse_error_not_enough_numbers"

        score_A = int(numbers[0])
        score_B = int(numbers[1])

        if score_A > score_B: return 'model_a'
        elif score_B > score_A: return 'model_b'
        else: return 'tie'    
    except Exception as e:
        print(f"An error occurred while parsing the results: {e}")

def get_assistant_answer(conversation):
    if len(conversation) > 1 and conversation[1]['role'] == 'assistant':
        return conversation[1]['content']
    for message in conversation:
        if message['role'] == 'assistant':
            return message['content']
    return None

if __name__ == "__main__":

    # debate_question = "Is vegetarianism a more sustainable and ethical choice for the modern world?"
    # ans1 = "Yes, vegetarianism is a more sustainable and ethical choice."
    # ans2 = "No, a balanced diet that includes meat can also be sustainable and ethical."
    
    # run_more_debate(debate_question, ans1, ans2)
    print("Loading MT-Bench human judgments dataset...")
    judgments_dataset = load_dataset("lmsys/mt_bench_human_judgments")
    print("Dataset loaded. Using the first entry as an example.")
    
    print("Loading questions from local file 'question.jsonl'...")
    questions_data = []
    with open('question.jsonl', 'r') as f:
        for line in f:
            questions_data.append(json.loads(line))
    questions_dict = {item['question_id']: item['turns'][0] for item in questions_data}
    sample_judgment = judgments_dataset['human'][1]
    
    q_id = sample_judgment['question_id']
    debate_question = questions_dict.get(q_id, "Question not found.")
    
    ans1 = get_assistant_answer(sample_judgment['conversation_a'])
    ans2 = get_assistant_answer(sample_judgment['conversation_b'])
    human_winner = sample_judgment['winner']
    
    system_winner = run_more_debate(debate_question, ans1, ans2)
    print("\n" + "="*50)
    print("  EVALUATION AGAINST HUMAN JUDGMENT")
    print("="*50)
    print(f"Human Judgment (Ground Truth): '{human_winner}'")
    print(f"Our System's Judgment:         '{system_winner}'")
    
    is_correct = (system_winner == human_winner) or \
                 (system_winner == 'tie' and 'tie' in human_winner)
                 
    if is_correct:
        print("\n Correct! Our system's judgment matches the human preference.")
    else:
        print("\n Incorrect. Our system's judgment differs from the human preference.")

