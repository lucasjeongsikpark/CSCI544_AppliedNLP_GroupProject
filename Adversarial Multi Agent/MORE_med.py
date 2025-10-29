import re
import ollama
import json
import csv

MODEL_NAME = 'deepseek-r1:1.5b'
INPUT_FILENAME = 'med_cleaned.json'  
NUM_SAMPLES_TO_RUN = 2

# --- LLM and Prompts Definition ---

def llm_call(prompt: str, role: str) -> str:
    """Sends a prompt to the specified LLM and returns the response."""
    print(f"\n----- LLM CALL ({role.upper()}) to {MODEL_NAME} -----")
    # print(f"prompt sent:\n---\n{prompt[:400]}...\n---\n")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content']
        # print("LLM Response Received.")
        return content
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return "Error: LLM call failed."

# --- MEDICAL-SPECIFIC PROMPTS ---

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
    <total_score>[Sum of the 5 scores above]</total_score>
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
    <total_score>[Sum of the 5 scores above]</total_score>
    <feedback>[Provide 20-30 words of concise feedback for Team B's argument]</feedback>
  </response_B>
</evaluation>
"""

# --- Core Logic ---

def parse_medical_judge_scores(judge_response: str):
    """Parses the judge's XML response for the medical evaluation."""
    scores = {}
    try:
        def extract_value(pattern, text):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try: return int(match.group(1).strip())
                except (ValueError, AttributeError): return match.group(1).strip()
            return None

        # Extract scores for Response A
        scores['A_medical_accuracy'] = extract_value(r"<response_A>.*?<medical_accuracy>(\d)</medical_accuracy>", judge_response)
        scores['A_appropriateness'] = extract_value(r"<response_A>.*?<appropriateness>(\d)</appropriateness>", judge_response)
        scores['A_safety'] = extract_value(r"<response_A>.*?<safety>(\d)</safety>", judge_response)
        scores['A_clarity'] = extract_value(r"<response_A>.*?<clarity>(\d)</clarity>", judge_response)
        scores['A_professionalism'] = extract_value(r"<response_A>.*?<professionalism>(\d)</professionalism>", judge_response)
        scores['A_total_score'] = extract_value(r"<response_A>.*?<total_score>(\d+)</total_score>", judge_response)
        scores['A_feedback'] = extract_value(r"<response_A>.*?<feedback>(.*?)</feedback>", judge_response)

        # Extract scores for Response B
        scores['B_medical_accuracy'] = extract_value(r"<response_B>.*?<medical_accuracy>(\d)</medical_accuracy>", judge_response)
        scores['B_appropriateness'] = extract_value(r"<response_B>.*?<appropriateness>(\d)</appropriateness>", judge_response)
        scores['B_safety'] = extract_value(r"<response_B>.*?<safety>(\d)</safety>", judge_response)
        scores['B_clarity'] = extract_value(r"<response_B>.*?<clarity>(\d)</clarity>", judge_response)
        scores['B_professionalism'] = extract_value(r"<response_B>.*?<professionalism>(\d)</professionalism>", judge_response)
        scores['B_total_score'] = extract_value(r"<response_B>.*?<total_score>(\d+)</total_score>", judge_response)
        scores['B_feedback'] = extract_value(r"<response_B>.*?<feedback>(.*?)</feedback>", judge_response)

        return scores
    except Exception as e:
        print(f"An error during parsing: {e}")
        return {key: "parse_error" for key in scores.keys()}

def run_more_debate(data_point: dict):
    """Runs the full debate and judging process for a single medical data point."""
    
    # Use .get() to safely access keys that might not exist in all data points
    document = data_point.get('document', 'N/A')
    input_question = data_point.get('input', 'N/A')
    answer_A = data_point.get('llama_output', '')
    answer_B = data_point.get('distill_llama_output', '')
    
    print("="*50 + f"\nDebating medical question: {input_question[:70]}...\n" + "="*50)

    # --- Teams build arguments using MEDICAL prompts ---
    prompt_A1 = ADVOCATE_INITIAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_A)
    argument_A1 = llm_call(prompt_A1, "advocate_initial_A")
    prompt_A2 = ADVOCATE_FINAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_A, teammate_argument=argument_A1)
    final_argument_A = llm_call(prompt_A2, "advocate_final_A")
    
    prompt_B1 = ADVOCATE_INITIAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_B)
    argument_B1 = llm_call(prompt_B1, "advocate_initial_B")
    prompt_B2 = ADVOCATE_FINAL_PROMPT_MEDICAL.format(document=document, input_question=input_question, answer_to_defend=answer_B, teammate_argument=argument_B1)
    final_argument_B = llm_call(prompt_B2, "advocate_final_B")

    # --- Judge evaluates using MEDICAL prompt ---
    judge_prompt = JUDGE_PROMPT_MEDICAL.format(
        document=document, input_question=input_question,
        answer_A_short=answer_A[:80]+"...", final_argument_A=final_argument_A,
        answer_B_short=answer_B[:80]+"...", final_argument_B=final_argument_B
    )
    judge_response = llm_call(judge_prompt, "judge")
    print(f"\nJudge's Full Response:\n{judge_response}")

    # --- Parse detailed medical scores ---
    parsed_scores = parse_medical_judge_scores(judge_response)
    
    winner = 'tie'
    score_A = parsed_scores.get('A_total_score')
    score_B = parsed_scores.get('B_total_score')
    if isinstance(score_A, int) and isinstance(score_B, int):
        if score_A > score_B: winner = 'llama_output'
        elif score_B > score_A: winner = 'distill_llama_output'
    else: winner = 'parse_error'
        
    # --- Collate all detailed results ---
    result = {
        "input_question": input_question,
        "winner": winner,
        "judge_raw_response": judge_response,
        **{f"A_{k.replace('A_','')}": v for k, v in parsed_scores.items() if k.startswith('A_')},
        **{f"B_{k.replace('B_','')}": v for k, v in parsed_scores.items() if k.startswith('B_')},
        "final_argument_A": final_argument_A,
        "final_argument_B": final_argument_B
    }
    return result

def load_data_from_json(filepath: str):
    """Loads data from a standard JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print(f"Loading MEDICAL dataset from '{INPUT_FILENAME}'...")
        all_data = load_data_from_json(INPUT_FILENAME)
        print(f"Dataset loaded. Found {len(all_data)} total entries.")
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILENAME}' not found.")
        exit()
        
    data_to_process = all_data[:NUM_SAMPLES_TO_RUN]
    print(f"Starting debates for the first {len(data_to_process)} entries using '{MODEL_NAME}'...")

    all_results = []
    for i, data_point in enumerate(data_to_process):
        print(f"\n\n===== PROCESSING MEDICAL ENTRY {i+1}/{len(data_to_process)} =====")
        debate_result = run_more_debate(data_point)
        all_results.append(debate_result)
        
        print("\n" + "="*50 + "\n  DEBATE SUMMARY\n" + "="*50)
        print(f"Patient Question: {debate_result['input_question'][:100]}...")
        print(f"Total Score for Response A (llama_output): {debate_result.get('A_total_score', 'N/A')}")
        print(f"Total Score for Response B (distill_llama_output): {debate_result.get('B_total_score', 'N/A')}")
        print(f"System's Judgment: '{debate_result['winner']}'")

    # Save results to uniquely named files
    json_output_path = f'medical_debate_results_{NUM_SAMPLES_TO_RUN}_samples.json'
    csv_output_path = f'medical_debate_results_{NUM_SAMPLES_TO_RUN}_samples.csv'
    
    if all_results:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"\n✅ All detailed medical results saved to {json_output_path}")

        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"✅ All detailed medical results also saved to {csv_output_path}")