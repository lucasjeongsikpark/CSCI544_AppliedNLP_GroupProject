import re
import ollama
import json
import csv
import time

MODEL_NAME = 'gemma2:2b'
INPUT_FILENAME = 'med_cleaned.json'  
NUM_SAMPLES_TO_RUN = None


def llm_call(prompt: str, role: str) -> str:
    print(f"\n----- LLM CALL ({role.upper()}) to {MODEL_NAME} -----")
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


ADVOCATE_INITIAL_PROMPT_MEDICAL = """
**Your Role:** You are an expert advocate with a background in medical review. Your goal is to build the strongest possible argument to defend a given medical response.
**The Patient's Situation:**
- Medical Context/Document: "{document}"
- Patient's Question: "{input_question}"
**The Response Your Team MUST Defend:**
"{answer_to_defend}"
**Your Task:**
Generate a strong, initial argument defending this response. Focus on its most compelling strengths, such as its medical accuracy, patient safety considerations, or the clarity of its explanation.
**Begin. Your Initial Argument:**
"""

ADVOCATE_FINAL_PROMPT_MEDICAL = """
**Your Role:** You are part of a team of expert medical reviewers. Your goal is to collaboratively build the strongest argument to defend a given medical response.
**The Patient's Situation:**
- Medical Context/Document: "{document}"
- Patient's Question: "{input_question}"
**The Response Your Team MUST Defend:**
"{answer_to_defend}"
**Your Teammate's Initial Argument:**
"{teammate_argument}"
**Instructions for You:**
1.  Read your teammate's argument carefully.
2.  **Do not simply repeat their points.**
3.  **Strengthen the case by:** adding new evidence of medical soundness, highlighting safety protocols in the advice, or emphasizing the professional and clear tone.
4.  Integrate your ideas to produce a single, cohesive, and more powerful **Final Argument**.
**Begin. Your Final Argument:**
"""

JUDGE_PROMPT_MEDICAL = """
**Your Role:** You are a fair and impartial judge, acting as a senior medical review officer. Your task is to evaluate the arguments for two different medical responses to a patient's question.
**The Patient's Situation:**
- Medical Context/Document: "{document}"
- Patient's Question: "{input_question}"
---
**Argument for Response A ("{answer_A_short}"):**
{final_argument_A}
---
**Argument for Response B ("{answer_B_short}"):**
{final_argument_B}
---
**Your Task:**
For EACH response, evaluate its quality based on the arguments provided. Rate EACH of the following five criteria on a scale of 1-5, where 1 is Very Poor and 5 is Excellent.
1.  **Medical Accuracy**: Is the information medically sound?
2.  **Appropriateness**: Is the advice appropriate for the described situation?
3.  **Safety**: Does it prioritize patient safety?
4.  **Clarity**: Is the explanation clear and understandable for a patient?
5.  **Professionalism**: Does it maintain an appropriate professional tone?
**Your Output Format:**
Provide your response in the exact XML format below. The `total_score` should be the sum of the five criteria scores. Do not add any other text or explanation.
<evaluation>
  <response_A>
    <scores>
      <medical_accuracy>[1-5]</medical_accuracy>
      <appropriateness>[1-5]</appropriateness>
      <safety>[1-5]</safety>
      <clarity>[1-5]</clarity>
      <professionalism>[1-5]</professionalism>
    </scores>
    <feedback>[Provide 20-30 words of concise feedback for Team A's argument]</feedback>
  </response_A>
  <response_B>
    <scores>
      <medical_accuracy>[1-5]</medical_accuracy>
      <appropriateness>[1-5]</appropriateness>
      <safety>[1-5]</safety>
      <clarity>[1-5]</clarity>
      <professionalism>[1-5]</professionalism>
    </scores>
    <feedback>[Provide 20-30 words of concise feedback for Team B's argument]</feedback>
  </response_B>
</evaluation>
"""


def parse_medical_judge_scores(judge_response: str):
    scores = {}
    try:
        def extract_value(pattern, text):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try: return int(match.group(1).strip())
                except (ValueError, AttributeError): return match.group(1).strip()
            return None

        scores['A_medical_accuracy'] = extract_value(r"<response_A>.*?<medical_accuracy>(\d)</medical_accuracy>", judge_response)
        scores['A_appropriateness'] = extract_value(r"<response_A>.*?<appropriateness>(\d)</appropriateness>", judge_response)
        scores['A_safety'] = extract_value(r"<response_A>.*?<safety>(\d)</safety>", judge_response)
        scores['A_clarity'] = extract_value(r"<response_A>.*?<clarity>(\d)</clarity>", judge_response)
        scores['A_professionalism'] = extract_value(r"<response_A>.*?<professionalism>(\d)</professionalism>", judge_response)
        scores['A_feedback'] = extract_value(r"<response_A>.*?<feedback>(.*?)</feedback>", judge_response)

        scores['B_medical_accuracy'] = extract_value(r"<response_B>.*?<medical_accuracy>(\d)</medical_accuracy>", judge_response)
        scores['B_appropriateness'] = extract_value(r"<response_B>.*?<appropriateness>(\d)</appropriateness>", judge_response)
        scores['B_safety'] = extract_value(r"<response_B>.*?<safety>(\d)</safety>", judge_response)
        scores['B_clarity'] = extract_value(r"<response_B>.*?<clarity>(\d)</clarity>", judge_response)
        scores['B_professionalism'] = extract_value(r"<response_B>.*?<professionalism>(\d)</professionalism>", judge_response)
        scores['B_feedback'] = extract_value(r"<response_B>.*?<feedback>(.*?)</feedback>", judge_response)

        return scores
    except Exception as e:
        print(f"An error during parsing: {e}")
        return {key: "parse_error" for key in scores.keys()}

MEDICAL_CRITERIA_KEYS = [
    'A_medical_accuracy', 'A_appropriateness', 'A_safety', 'A_clarity', 'A_professionalism', 'A_feedback',
    'B_medical_accuracy', 'B_appropriateness', 'B_safety', 'B_clarity', 'B_professionalism', 'B_feedback'
]

def is_parsing_successful(parsed_scores):
    if not parsed_scores: 
        return False
    for key in MEDICAL_CRITERIA_KEYS:
        if parsed_scores.get(key) is None:
            print(f"Parsing check failed: Missing or invalid key '{key}'")
            return False
    return True

def run_more_debate(data_point: dict):
    document = data_point.get('document', 'N/A')
    reference_output = data_point.get('output', 'N/A')
    input_question = data_point.get('input', 'N/A')
    answer_A = data_point.get('llama_output', '')
    answer_B = data_point.get('distill_llama_output', '')
    
    print("="*50 + f"\nDebating medical question: {input_question[:70]}...\n" + "="*50)

    print("--- Running Advocate Debates ---")
    prompt_A1 = ADVOCATE_INITIAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_A)
    argument_A1 = llm_call(prompt_A1, "advocate_initial_A")
    prompt_A2 = ADVOCATE_FINAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_A, teammate_argument=argument_A1)
    final_argument_A = llm_call(prompt_A2, "advocate_final_A")
    
    prompt_B1 = ADVOCATE_INITIAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_B)
    argument_B1 = llm_call(prompt_B1, "advocate_initial_B")
    prompt_B2 = ADVOCATE_FINAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_B, teammate_argument=argument_B1)
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
        
        judge_prompt = JUDGE_PROMPT_MEDICAL.format(
            document=document, input_question=input_question,
            answer_A_short=answer_A[:80]+"...", final_argument_A=final_argument_A,
            answer_B_short=answer_B[:80]+"...", final_argument_B=final_argument_B
        )
        judge_response = llm_call(judge_prompt, "judge")
        print(f"\nJudge's Raw Response (Attempt {attempts}):\n{judge_response}")

        parsed_scores = parse_medical_judge_scores(judge_response)
        parsing_success = is_parsing_successful(parsed_scores)

        if parsing_success:
            print(f"Parsing successful on attempt {attempts}.")
            break
        else:
            print(f"Parsing failed on attempt {attempts}. Retrying...")
    
    if not parsing_success:
        print("Failed to parse judge output after max attempts.")

    a_scores = [
        parsed_scores.get('A_medical_accuracy'),
        parsed_scores.get('A_appropriateness'),
        parsed_scores.get('A_safety'),
        parsed_scores.get('A_clarity'),
        parsed_scores.get('A_professionalism')
    ]
    
    b_scores = [
        parsed_scores.get('B_medical_accuracy'),
        parsed_scores.get('B_appropriateness'),
        parsed_scores.get('B_safety'),
        parsed_scores.get('B_clarity'),
        parsed_scores.get('B_professionalism')
    ]

    if all(isinstance(s, int) for s in a_scores):
        score_A_total = sum(a_scores)
    
    if all(isinstance(s, int) for s in b_scores):
        score_B_total = sum(b_scores)

    winner = 'tie'
    
    if isinstance(score_A_total, int) and isinstance(score_B_total, int):
        if score_A_total > score_B_total: winner = 'llama_output'
        elif score_B_total > score_A_total: winner = 'distill_llama_output'
    elif not parsing_success:
        winner = 'parse_error'
        
    
    score1 = {
        "Medical Accuracy": parsed_scores.get('A_medical_accuracy'),
        "Appropriateness": parsed_scores.get('A_appropriateness'),
        "Safety": parsed_scores.get('A_safety'),
        "Clarity": parsed_scores.get('A_clarity'),
        "Professionalism": parsed_scores.get('A_professionalism'),
        "Feedback": parsed_scores.get('A_feedback')
    }

    score2 = {
        "Medical Accuracy": parsed_scores.get('B_medical_accuracy'),
        "Appropriateness": parsed_scores.get('B_appropriateness'),
        "Safety": parsed_scores.get('B_safety'),
        "Clarity": parsed_scores.get('B_clarity'),
        "Professionalism": parsed_scores.get('B_professionalism'),
        "Feedback": parsed_scores.get('B_feedback')
        
    }
        
    result = {
        "input": input_question,
        "output": reference_output,
        "document": document,
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
        return json.load(f)

if __name__ == "__main__":
    try:
        all_data = load_data_from_json(INPUT_FILENAME)
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILENAME}' not found.")
        exit()
        
    data_to_process = all_data
    print(f"Starting debates for the first {len(data_to_process)} entries using '{MODEL_NAME}'...")

    all_results = []
    for i, data_point in enumerate(data_to_process):
        print(f"\n\n===== PROCESSING MEDICAL ENTRY {i+1}/{len(data_to_process)} =====")
        
        start_time = time.time()
        
        debate_result = run_more_debate(data_point)
        
        end_time = time.time()
        debate_result["elapsed_time"] = end_time - start_time
        
        all_results.append(debate_result)
        
        print("\n" + "="*50 + "\n  DEBATE SUMMARY\n" + "="*50)
        print(f"Patient Question: {debate_result['input'][:100]}...")
        print(f"System's Judgment: '{debate_result['winner']}'")
        print(f"Parsing Attempts: {debate_result['attempt']}")
        print(f"Elapsed Time: {debate_result['elapsed_time']:.2f}s") 

    json_output_path = f'medical_debate_results_{NUM_SAMPLES_TO_RUN}_samples.json'
    
    jsonl_output_path = f'medical_debate_results_{NUM_SAMPLES_TO_RUN}_samples.jsonl'
    
    print(f"\n--- Saving {len(all_results)} results ---")

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"✅ All detailed results saved to {json_output_path}")

    try:
        with open(jsonl_output_path, 'w', encoding='utf-8') as f:
            for entry in all_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ All detailed results also saved to {jsonl_output_path} (JSONL format)")
    except Exception as e:

        print(f"Could not save to JSONL: {e}")
