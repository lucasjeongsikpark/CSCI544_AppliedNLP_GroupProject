import re
import ollama
import json
import time

MODEL_NAME = 'gemma2:2b'
INPUT_FILENAME = 'math_cleaned_250.json' 
NUM_SAMPLES_TO_RUN = None 

# --- LLM and Prompts Definition ---

def llm_call(prompt: str, role: str) -> str:
    """Sends a prompt to the specified LLM and returns the response."""
    print(f"\n----- LLM CALL ({role.upper()}) to {MODEL_NAME} -----")
    # print(f"prompt sent:\n---\n{prompt[:400]}...\n---\n")
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt
        )
        content = response['response']
        # print("LLM Response Received.")
        return content
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return "Error: LLM call failed."

# --- MATH-SPECIFIC PROMPTS ---

ADVOCATE_INITIAL_PROMPT_MATH = """
**Your Role:** You are an expert mathematical advocate. Your goal is to build the strongest possible argument to defend a given mathematical solution.
**The Math Problem:**
"{input_question}"
**The Solution Your Team MUST Defend:**
"{answer_to_defend}"
**Your Task:**
Generate a strong, initial argument defending this solution. Focus on key strengths, such as the correctness of the final answer, the logical clarity of the reasoning steps, or the accuracy of the calculations.
**Begin. Your Initial Argument:**
"""

ADVOCATE_FINAL_PROMPT_MATH = """
**Your Role:** You are part of a team of expert mathematical advocates. Your goal is to collaboratively build the strongest argument to defend a given solution.
**The Math Problem:**
"{input_question}"
**The Solution Your Team MUST Defend:**
"{answer_to_defend}"
**Your Teammate's Initial Argument:**
"{teammate_argument}"
**Instructions for You:**
1.  Read your teammate's argument carefully.
2.  **Do not simply repeat their points.**
3.  **Strengthen the case by:** adding new evidence from the solution, highlighting the logical flow of the reasoning, or verifying the accuracy of the calculations.
4.  Integrate your ideas to produce a single, cohesive, and more powerful **Final Argument**.
**Begin. Your Final Argument:**
"""

JUDGE_PROMPT_MATH = """
**Your Role:** You are a fair and impartial judge, specializing in evaluating mathematical reasoning. Your task is to evaluate the arguments for two different solutions to the same math problem.
**The Math Problem:**
"{input_question}"
---
**Argument for Solution A ("{answer_A_short}"):**
{final_argument_A}
---
**Argument for Solution B ("{answer_B_short}"):**
{final_argument_B}
---
**Your Task:**
For EACH solution, evaluate its quality based on the arguments provided. Rate EACH of the following criteria on a scale of 1-5, where 1 is Very Poor and 5 is Excellent.
1.  **Correctness**: Does it arrive at the correct final answer?
2.  **Reasoning**: Is the step-by-step reasoning clear and logical?
3.  **Completeness**: Are all necessary steps shown?
4.  **Accuracy**: Are the intermediate calculations correct?
**Your Output Format:**
Provide your response in the exact XML format below. The `total_score` should be the sum of the four criteria scores. Do not add any other text or explanation.
<evaluation>
  <solution_A>
    <scores>
      <correctness>[1-5]</correctness>
      <reasoning>[1-5]</reasoning>
      <completeness>[1-5]</completeness>
      <accuracy>[1-5]</accuracy>
    </scores>
    <feedback>[Provide 20-30 words of concise feedback for Team A's argument]</feedback>
  </solution_A>
  <solution_B>
    <scores>
      <correctness>[1-5]</correctness>
      <reasoning>[1-5]</reasoning>
      <completeness>[1-5]</completeness>
      <accuracy>[1-5]</accuracy>
    </scores>
    <feedback>[Provide 20-30 words of concise feedback for Team B's argument]</feedback>
  </solution_B>
</evaluation>
"""

# --- Core Logic ---

def parse_math_judge_scores(judge_response: str):
    """Parses the judge's XML response for the math evaluation."""
    scores = {}
    try:
        def extract_value(pattern, text, is_numeric=False):
            """
            Helper function to extract a value.
            [NEW] Regex now optionally matches brackets around the digit.
            """
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

        scores['A_correctness'] = extract_value(r"<solution_A>.*?<correctness>\[?(\d+)\]?</correctness>", judge_response, is_numeric=True)
        scores['A_reasoning'] = extract_value(r"<solution_A>.*?<reasoning>\[?(\d+)\]?</reasoning>", judge_response, is_numeric=True)
        scores['A_completeness'] = extract_value(r"<solution_A>.*?<completeness>\[?(\d+)\]?</completeness>", judge_response, is_numeric=True)
        scores['A_accuracy'] = extract_value(r"<solution_A>.*?<accuracy>\[?(\d+)\]?</accuracy>", judge_response, is_numeric=True)
        scores['A_feedback'] = extract_value(r"<solution_A>.*?<feedback>(.*?)</feedback>", judge_response, is_numeric=False)

        scores['B_correctness'] = extract_value(r"<solution_B>.*?<correctness>\[?(\d+)\]?</correctness>", judge_response, is_numeric=True)
        scores['B_reasoning'] = extract_value(r"<solution_B>.*?<reasoning>\[?(\d+)\]?</reasoning>", judge_response, is_numeric=True)
        scores['B_completeness'] = extract_value(r"<solution_B>.*?<completeness>\[?(\d+)\]?</completeness>", judge_response, is_numeric=True)
        scores['B_accuracy'] = extract_value(r"<solution_B>.*?<accuracy>\[?(\d+)\]?</accuracy>", judge_response, is_numeric=True)
        scores['B_feedback'] = extract_value(r"<solution_B>.*?<feedback>(.*?)</feedback>", judge_response, is_numeric=False)

        return scores
    except Exception as e:
        print(f"An error during parsing: {e}")
        return {}

MATH_CRITERIA_KEYS = [
    'A_correctness', 'A_reasoning', 'A_completeness', 'A_accuracy', 'A_feedback',
    'B_correctness', 'B_reasoning', 'B_completeness', 'B_accuracy', 'B_feedback'
]

def is_parsing_successful(parsed_scores):
    """Checks if all expected math keys were parsed correctly."""
    if not parsed_scores: 
        return False
    for key in MATH_CRITERIA_KEYS:
        if parsed_scores.get(key) is None:
            print(f"Parsing check failed: Missing or invalid key '{key}'")
            return False
    return True

def run_more_debate(data_point: dict):
    """Runs the full debate and judging process for a single math data point."""
    
    input_question = data_point.get('input', 'N/A')
    reference_output = data_point.get('output', 'N/A') 
    # document = data_point.get('document', 'N/A')
    answer_A = data_point.get('llama_output', '')
    answer_B = data_point.get('distill_llama_output', '')
    
    print("="*50 + f"\nDebating math problem: {input_question[:80]}...\n" + "="*50)

    # --- 2. 运行辩论 (Advocates) ---
    prompt_A1 = ADVOCATE_INITIAL_PROMPT_MATH.format(input_question=input_question, answer_to_defend=answer_A)
    argument_A1 = llm_call(prompt_A1, "advocate_initial_A")
    prompt_A2 = ADVOCATE_FINAL_PROMPT_MATH.format(input_question=input_question, answer_to_defend=answer_A, teammate_argument=argument_A1)
    final_argument_A = llm_call(prompt_A2, "advocate_final_A")
    
    prompt_B1 = ADVOCATE_INITIAL_PROMPT_MATH.format(input_question=input_question, answer_to_defend=answer_B)
    argument_B1 = llm_call(prompt_B1, "advocate_initial_B")
    prompt_B2 = ADVOCATE_FINAL_PROMPT_MATH.format(input_question=input_question, answer_to_defend=answer_B, teammate_argument=argument_B1)
    final_argument_B = llm_call(prompt_B2, "advocate_final_B")

    # --- 3. 运行评判 & 解析循环 ---
    print("--- Running Judge & Parsing Loop ---")
    MAX_ATTEMPTS = 3
    attempts = 0
    parsed_scores = {}
    judge_response = ""
    parsing_success = False

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"Judge attempt {attempts}/{MAX_ATTEMPTS}...")
        
        judge_prompt = JUDGE_PROMPT_MATH.format(
            input_question=input_question,
            answer_A_short=answer_A[:80]+"...", final_argument_A=final_argument_A,
            answer_B_short=answer_B[:80]+"...", final_argument_B=final_argument_B
        )
        judge_response = llm_call(judge_prompt, "judge")
        print(f"\nJudge's Raw Response (Attempt {attempts}):\n{judge_response}")

        parsed_scores = parse_math_judge_scores(judge_response)
        parsing_success = is_parsing_successful(parsed_scores)

        if parsing_success:
            print(f"Parsing successful on attempt {attempts}.")
            break
        else:
            print(f"Parsing failed on attempt {attempts}. Retrying...")
    
    if not parsing_success:
        print("Failed to parse judge output perfectly after max attempts.")
    
    # --- 4. 在Python中计算总分 ---
    score_A_total = None
    score_B_total = None

    a_scores = [
        parsed_scores.get('A_correctness'),
        parsed_scores.get('A_reasoning'),
        parsed_scores.get('A_completeness'),
        parsed_scores.get('A_accuracy')
    ]
    b_scores = [
        parsed_scores.get('B_correctness'),
        parsed_scores.get('B_reasoning'),
        parsed_scores.get('B_completeness'),
        parsed_scores.get('B_accuracy')
    ]

    if all(isinstance(s, int) for s in a_scores):
        score_A_total = sum(a_scores)
    
    if all(isinstance(s, int) for s in b_scores):
        score_B_total = sum(b_scores)
        
    print(f"Calculated Score A: {score_A_total}, Calculated Score B: {score_B_total}")
        
    # --- 5. 判定获胜者 ---
    winner = 'tie'
    if isinstance(score_A_total, int) and isinstance(score_B_total, int):
        if score_A_total > score_B_total: 
            winner = 'llama_output'
        elif score_B_total > score_A_total: 
            winner = 'distill_llama_output'
    else:
        winner = 'parse_error'
        
    # --- 6. 结构化最终输出 ---
    score1 = {
        "Correctness": parsed_scores.get('A_correctness'),
        "Reasoning": parsed_scores.get('A_reasoning'),
        "Completeness": parsed_scores.get('A_completeness'),
        "Accuracy": parsed_scores.get('A_accuracy'),
        "total_score": score_A_total,
        "feedback": parsed_scores.get('A_feedback')
    }
    score2 = {
        "Correctness": parsed_scores.get('B_correctness'),
        "Reasoning": parsed_scores.get('B_reasoning'),
        "Completeness": parsed_scores.get('B_completeness'),
        "Accuracy": parsed_scores.get('B_accuracy'),
        "total_score": score_B_total,
        "feedback": parsed_scores.get('B_feedback')
    }

    result = {
        "input": input_question,
        "output": reference_output,
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
    """Loads data from a standard JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print(f"Loading MATH dataset from '{INPUT_FILENAME}'...")
        all_data = load_data_from_json(INPUT_FILENAME)
        print(f"Dataset loaded. Found {len(all_data)} total entries.")
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILENAME}' not found.")
        exit()
        
    data_to_process = all_data[:NUM_SAMPLES_TO_RUN]
    print(f"Starting debates for the first {len(data_to_process)} entries using '{MODEL_NAME}'...")

    all_results = []
    for i, data_point in enumerate(data_to_process):
        print(f"\n\n===== PROCESSING MATH ENTRY {i+1}/{len(data_to_process)} =====")
        
        start_time = time.time()
        debate_result = run_more_debate(data_point)
        end_time = time.time()
        debate_result["elapsed_time"] = end_time - start_time
        
        all_results.append(debate_result)
        
        print("\n" + "="*50 + "\n  DEBATE SUMMARY\n" + "="*50)
        print(f"Math Problem: {debate_result['input'][:100]}...")
        print(f"Total Score for Solution A: {debate_result['score1'].get('total_score', 'N/A')}")
        print(f"Total Score for Solution B: {debate_result['score2'].get('total_score', 'N/A')}")
        print(f"System's Judgment: '{debate_result['winner']}'")
        print(f"Parsing Attempts: {debate_result['attempt']}")
        print(f"Elapsed Time: {debate_result['elapsed_time']:.2f}s")

    # 保存结果
    json_output_path = f'math_debate_results_{NUM_SAMPLES_TO_RUN}_samples.json'
    jsonl_output_path = f'math_debate_results_{NUM_SAMPLES_TO_RUN}_samples.jsonl'
    
    print(f"\n--- Saving {len(all_results)} results ---")

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"✅ All detailed math results saved to {json_output_path}")

    try:
        with open(jsonl_output_path, 'w', encoding='utf-8') as f:
            for entry in all_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ All detailed math results also saved to {jsonl_output_path} (JSONL format)")
    except Exception as e:
        print(f"Could not save to JSONL: {e}")