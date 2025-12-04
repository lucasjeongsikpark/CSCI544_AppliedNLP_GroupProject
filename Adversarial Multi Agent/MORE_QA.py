import re
import ollama
import json
import time

NUM_SAMPLES = None
MODEL = 'gemma2:2b'

def llm_call(prompt: str, role: str) -> str:
    print(f"\n----- LLM CALL ({role.upper()}) to {MODEL} -----")
    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt
        )
        content = response['response']
        return content
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return "Error: LLM call failed."

ADVOCATE_INITIAL_PROMPT = """
**Your Role:** You are an expert advocate. Your goal is to build the strongest possible argument to defend a given AI's answer.

**The Original AI Task Was:**
- System Prompt: "{system_prompt}"
- Input Question: "{input_question}"

**The Specific Answer Your Team MUST Defend:**
"{answer_to_defend}"

**Your Task:**
Generate a strong, initial argument defending this answer's quality (relevance, accuracy, clarity). Focus on one or two key strengths to start the debate.

**Begin. Your Initial Argument:**
"""

ADVOCATE_FINAL_PROMPT = """
**Your Role:** You are part of a team of expert advocates. Your goal is to collaboratively build the strongest possible argument to defend a given AI's answer.

**The Original AI Task Was:**
- System Prompt: "{system_prompt}"
- Input Question: "{input_question}"

**The Specific Answer Your Team MUST Defend:**
"{answer_to_defend}"

**Your Teammate's Initial Argument:**
"{teammate_argument}"

**Instructions for You:**
1. Read your teammate's argument carefully.
2. **Do not simply repeat their points.**
3. **Strengthen the case by:** adding new evidence, refining the logic, and addressing potential weaknesses in the answer you are defending.
4. Integrate your ideas with your teammate's to produce a single, cohesive, and more powerful **Final Argument**.

**Begin. Your Final Argument:**
"""

JUDGE_PROMPT = """
**Your Role:** You are a fair and impartial judge. Your task is to evaluate the arguments presented for two different AI-generated answers to the same question.

**The Original AI Task Was:**
- System Prompt: "{system_prompt}"
- Input Question: "{input_question}"

---
**Argument for Answer A ("{answer_A_short}"):**
{final_argument_A}
---
**Argument for Answer B ("{answer_B_short}"):**
{final_argument_B}
---

**Your Evaluation Criteria:**
For EACH answer, evaluate its quality based on the arguments provided. Rate EACH of the following criteria on a scale of 1-5, where 1 is Very Poor and 5 is Excellent.

1.  **Relevance**: Does it address the question appropriately?
2.  **Completeness**: Is the response comprehensive and detailed?
3.  **Accuracy**: Is the information correct?
4.  **Clarity**: Is the response well-structured and clear?
5.  **Helpfulness**: Would this response be useful to the user?

**Your Output Format:**
Provide your response in the exact XML format below. Calculate the `total_score` by summing the five criteria scores. Do not add any other text or explanation.

<evaluation>
  <answer_A>
    <scores>
      <relevance>[1-5]</relevance>
      <completeness>[1-5]</completeness>
      <accuracy>[1-5]</accuracy>
      <clarity>[1-5]</clarity>
      <helpfulness>[1-5]</helpfulness>
    </scores>
    <feedback>[Provide 20-30 words of concise feedback for Team A's argument]</feedback>
  </answer_A>
  <answer_B>
    <scores>
      <relevance>[1-5]</relevance>
      <completeness>[1-5]</completeness>
      <accuracy>[1-5]</accuracy>
      <clarity>[1-5]</clarity>
      <helpfulness>[1-5]</helpfulness>
    </scores>
    <feedback>[Provide 20-30 words of concise feedback for Team B's argument]</feedback>
  </answer_B>
</evaluation>
"""


def parse_judge_scores(judge_response: str):
    scores = {}
    try:
        def extract_value(pattern, text, is_numeric=False):
            match = re.search(pattern, text, re.DOTALL)
            if not match: return None
            
            val_str = match.group(1).strip()
            
            if is_numeric:
                try:
                    return int(val_str)
                except ValueError:
                    return None
            else:
                return val_str if val_str else None

        scores['A_relevance'] = extract_value(r"<answer_A>.*?<relevance>\[?(\d+)\]?</relevance>", judge_response, is_numeric=True)
        scores['A_completeness'] = extract_value(r"<answer_A>.*?<completeness>\[?(\d+)\]?</completeness>", judge_response, is_numeric=True)
        scores['A_accuracy'] = extract_value(r"<answer_A>.*?<accuracy>\[?(\d+)\]?</accuracy>", judge_response, is_numeric=True)
        scores['A_clarity'] = extract_value(r"<answer_A>.*?<clarity>\[?(\d+)\]?</clarity>", judge_response, is_numeric=True)
        scores['A_helpfulness'] = extract_value(r"<answer_A>.*?<helpfulness>\[?(\d+)\]?</helpfulness>", judge_response, is_numeric=True)
        scores['A_feedback'] = extract_value(r"<answer_A>.*?<feedback>(.*?)</feedback>", judge_response, is_numeric=False)

        scores['B_relevance'] = extract_value(r"<answer_B>.*?<relevance>\[?(\d+)\]?</relevance>", judge_response, is_numeric=True)
        scores['B_completeness'] = extract_value(r"<answer_B>.*?<completeness>\[?(\d+)\]?</completeness>", judge_response, is_numeric=True)
        scores['B_accuracy'] = extract_value(r"<answer_B>.*?<accuracy>\[?(\d+)\]?</accuracy>", judge_response, is_numeric=True)
        scores['B_clarity'] = extract_value(r"<answer_B>.*?<clarity>\[?(\d+)\]?</clarity>", judge_response, is_numeric=True)
        scores['B_helpfulness'] = extract_value(r"<answer_B>.*?<helpfulness>\[?(\d+)\]?</helpfulness>", judge_response, is_numeric=True)
        scores['B_feedback'] = extract_value(r"<answer_B>.*?<feedback>(.*?)</feedback>", judge_response, is_numeric=False)

        return scores
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return {}
    
OPENQA_CRITERIA_KEYS = [
    'A_relevance', 'A_completeness', 'A_accuracy', 'A_clarity', 'A_helpfulness', 'A_feedback',
    'B_relevance', 'B_completeness', 'B_accuracy', 'B_clarity', 'B_helpfulness', 'B_feedback'
]

def is_parsing_successful(parsed_scores):
    if not parsed_scores: 
        return False
    for key in OPENQA_CRITERIA_KEYS:
        if parsed_scores.get(key) is None:
            print(f"Parsing check failed: Missing or invalid key '{key}'")
            return False
    return True

def run_more_debate(data_point: dict):    
    system_prompt = data_point.get('system_prompt', 'N/A')
    input_question = data_point.get('input', 'N/A')
    reference_output = data_point.get('output', 'N/A')
    answer_A = data_point.get('llama_output', '')
    answer_B = data_point.get('distill_llama_output', '')
    
    print("="*50 + f"\nDebating OpenQA question: {input_question[:80]}...\n" + "="*50)

    prompt_A1 = ADVOCATE_INITIAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_A)
    argument_A1 = llm_call(prompt_A1, "advocate_initial_A")
    prompt_A2 = ADVOCATE_FINAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_A, teammate_argument=argument_A1)
    final_argument_A = llm_call(prompt_A2, "advocate_final_A")
    
    prompt_B1 = ADVOCATE_INITIAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_B)
    argument_B1 = llm_call(prompt_B1, "advocate_initial_B")
    prompt_B2 = ADVOCATE_FINAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_B, teammate_argument=argument_B1)
    final_argument_B = llm_call(prompt_B2, "advocate_final_B")

    print("--- Running Judge & Parsing Loop ---")
    MAX_ATTEMPTS = 3
    attempts = 0
    parsed_scores = {}
    judge_response = ""
    parsing_success = False

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Judge attempt {attempts}/{MAX_ATTEMPTS}...")
        
        judge_prompt = JUDGE_PROMPT.format(
            system_prompt=system_prompt, input_question=input_question,
            answer_A_short=answer_A[:80]+"...", final_argument_A=final_argument_A,
            answer_B_short=answer_B[:80]+"...", final_argument_B=final_argument_B
        )
        judge_response = llm_call(judge_prompt, "judge")
        print(f"\nJudge's Raw Response (Attempt {attempts}):\n{judge_response}")

        parsed_scores = parse_judge_scores(judge_response)
        parsing_success = is_parsing_successful(parsed_scores)

        if parsing_success:
            print(f"Parsing successful on attempt {attempts}.")
            break
        else:
            print(f"Parsing failed on attempt {attempts}. Retrying...")
    
    if not parsing_success:
        print("Failed to parse judge output perfectly after max attempts.")
    
    score_A_total = None
    score_B_total = None

    a_scores = [
        parsed_scores.get('A_relevance'),
        parsed_scores.get('A_completeness'),
        parsed_scores.get('A_accuracy'),
        parsed_scores.get('A_clarity'),
        parsed_scores.get('A_helpfulness')
    ]
    b_scores = [
        parsed_scores.get('B_relevance'),
        parsed_scores.get('B_completeness'),
        parsed_scores.get('B_accuracy'),
        parsed_scores.get('B_clarity'),
        parsed_scores.get('B_helpfulness')
    ]

    if all(isinstance(s, int) for s in a_scores):
        score_A_total = sum(a_scores)
    
    if all(isinstance(s, int) for s in b_scores):
        score_B_total = sum(b_scores)
        
    print(f"Calculated Score A: {score_A_total}, Calculated Score B: {score_B_total}")
        
    winner = 'tie'
    if isinstance(score_A_total, int) and isinstance(score_B_total, int):
        if score_A_total > score_B_total: 
            winner = 'llama_output'
        elif score_B_total > score_A_total: 
            winner = 'distill_llama_output'
    else:
        winner = 'parse_error'
        
    score1 = {
        "Relevance": parsed_scores.get('A_relevance'),
        "Completeness": parsed_scores.get('A_completeness'),
        "Accuracy": parsed_scores.get('A_accuracy'),
        "Clarity": parsed_scores.get('A_clarity'),
        "Helpfulness": parsed_scores.get('A_helpfulness'),
        "total_score": score_A_total,
        "feedback": parsed_scores.get('A_feedback')
    }
    score2 = {
        "Relevance": parsed_scores.get('B_relevance'),
        "Completeness": parsed_scores.get('B_completeness'),
        "Accuracy": parsed_scores.get('B_accuracy'),
        "Clarity": parsed_scores.get('B_clarity'),
        "Helpfulness": parsed_scores.get('B_helpfulness'),
        "total_score": score_B_total,
        "feedback": parsed_scores.get('B_feedback')
    }

    result = {
        "input": input_question,
        "output": reference_output,
        "system_prompt": system_prompt,
        "response": {
            "llama": answer_A,
            "distill_llama": answer_B
        },
        "evaluation": [
            {
                "role": "Debate Judge",
                "evaluation_raw": judge_response, 
                "final_argument_A": final_argument_A,
                "final_argument_B": final_argument_B
            }
        ],
        "score1": score1,
        "score2": score2,
        "attempt": attempts,
        "winner": winner
    }
    
    return result

def load_data_from_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    try:
        print("Loading openQA dataset from 'openQA_cleaned_250.json'...")
        all_data = load_data_from_json('openQA_cleaned_250.json')
        print(f"Dataset loaded. Found {len(all_data)} entries.")
    except FileNotFoundError:
        print("Error: 'openQA_cleaned_250.json' not found. Please create this file and add your data.")
        exit()
        
    all_results = []
    
    data_to_process = all_data[:NUM_SAMPLES]
    for i, data_point in enumerate(data_to_process):
        print(f"\n\n===== PROCESSING OpenQA ENTRY  {i+1}/{len(data_to_process)} =====")
        
        start_time = time.time()
        debate_result = run_more_debate(data_point)
        end_time = time.time()
        debate_result["elapsed_time"] = end_time - start_time
        
        all_results.append(debate_result)
        
        print("\n" + "="*50 + "\n  DEBATE SUMMARY\n" + "="*50)
        print(f"Question: {debate_result['input'][:100]}...")

        print(f"Parsing Attempts: {debate_result['attempt']}")
        print(f"Elapsed Time: {debate_result['elapsed_time']:.2f}s") 

    json_output_path = f'openqa_debate_results_{NUM_SAMPLES}_samples.json'
    jsonl_output_path = f'openqa_debate_results_{NUM_SAMPLES}_samples.jsonl'
    
    print(f"\n--- Saving {len(all_results)} results ---")

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"✅ All detailed OpenQA results saved to {json_output_path}")

    try:
        with open(jsonl_output_path, 'w', encoding='utf-8') as f:
            for entry in all_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ All detailed OpenQA results also saved to {jsonl_output_path} (JSONL format)")
    except Exception as e:

        print(f"Could not save to JSONL: {e}")
