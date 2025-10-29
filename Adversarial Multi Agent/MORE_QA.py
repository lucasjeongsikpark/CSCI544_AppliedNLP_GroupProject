import re
import ollama
import json
import csv

# --- LLM and Prompts Definition ---
NUM_SAMPLES = 2
MODEL = 'gemma2:2b'
def llm_call(prompt: str, role: str) -> str:
    """Sends a prompt to the local LLM and returns the response."""
    print(f"\n----- LLM CALL ({role.upper()}) to {MODEL} -----")
    # Print a snippet of the prompt for logging
    # print(f"prompt sent:\n---\n{prompt[:350]}...\n---\n")
    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt
        )
        content = response['response']
        # print("LLM Response Received.")
        return content
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return "Error: LLM call failed."

# Advocate prompts now include the original system prompt and input for context
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

# Judge prompt updated for the new 1-5 scale and criteria
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
    <total_score>[Sum of the 5 scores above]</total_score>
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
    <total_score>[Sum of the 5 scores above]</total_score>
    <feedback>[Provide 20-30 words of concise feedback for Team B's argument]</feedback>
  </answer_B>
</evaluation>
"""

# --- Core Logic ---

def parse_judge_scores(judge_response: str):
    """
    Parses the judge's new XML response to get detailed scores and feedback for both answers.
    """
    scores = {}
    try:
        # Helper function to extract a value
        def extract_value(pattern, text):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Try to convert to int, otherwise return the string
                try:
                    return int(match.group(1).strip())
                except (ValueError, AttributeError):
                    return match.group(1).strip()
            return None # Return None if no match

        # Extract scores for Answer A
        scores['A_relevance'] = extract_value(r"<answer_A>.*?<relevance>(\d)</relevance>", judge_response)
        scores['A_completeness'] = extract_value(r"<answer_A>.*?<completeness>(\d)</completeness>", judge_response)
        scores['A_accuracy'] = extract_value(r"<answer_A>.*?<accuracy>(\d)</accuracy>", judge_response)
        scores['A_clarity'] = extract_value(r"<answer_A>.*?<clarity>(\d)</clarity>", judge_response)
        scores['A_helpfulness'] = extract_value(r"<answer_A>.*?<helpfulness>(\d)</helpfulness>", judge_response)
        scores['A_total_score'] = extract_value(r"<answer_A>.*?<total_score>(\d+)</total_score>", judge_response)
        scores['A_feedback'] = extract_value(r"<answer_A>.*?<feedback>(.*?)</feedback>", judge_response)

        # Extract scores for Answer B
        scores['B_relevance'] = extract_value(r"<answer_B>.*?<relevance>(\d)</relevance>", judge_response)
        scores['B_completeness'] = extract_value(r"<answer_B>.*?<completeness>(\d)</completeness>", judge_response)
        scores['B_accuracy'] = extract_value(r"<answer_B>.*?<accuracy>(\d)</accuracy>", judge_response)
        scores['B_clarity'] = extract_value(r"<answer_B>.*?<clarity>(\d)</clarity>", judge_response)
        scores['B_helpfulness'] = extract_value(r"<answer_B>.*?<helpfulness>(\d)</helpfulness>", judge_response)
        scores['B_total_score'] = extract_value(r"<answer_B>.*?<total_score>(\d+)</total_score>", judge_response)
        scores['B_feedback'] = extract_value(r"<answer_B>.*?<feedback>(.*?)</feedback>", judge_response)

        return scores

    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return {key: "parse_error" for key in scores.keys()}

def run_more_debate(data_point: dict):
    """Runs the full debate and judging process for a single OpenQA data point."""
    
    system_prompt = data_point['system_prompt']
    input_question = data_point['input']
    answer_A = data_point['llama_output'] # Team A defends llama_output
    answer_B = data_point['distill_llama_output'] # Team B defends distill_llama_output
    
    print("="*50 + f"\nDebating question: {input_question[:80]}...\n" + "="*50)
    # --- Team A builds its argument for `llama_output` ---
    # print("\n--- Team A is building its argument... ---")
    prompt_A1 = ADVOCATE_INITIAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_A)
    argument_A1 = llm_call(prompt_A1, role="advocate_initial_A")
    
    prompt_A2 = ADVOCATE_FINAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_A, teammate_argument=argument_A1)
    final_argument_A = llm_call(prompt_A2, role="advocate_final_A")
    print(f"\nTeam A's Final Argument:\n{final_argument_A[:400]}...")

    # --- Team B builds its argument for `distill_llama_output` ---
    # print("\n--- Team B is building its argument... ---")
    prompt_B1 = ADVOCATE_INITIAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_B)
    argument_B1 = llm_call(prompt_B1, role="advocate_initial_B")
    
    prompt_B2 = ADVOCATE_FINAL_PROMPT.format(system_prompt=system_prompt, input_question=input_question, answer_to_defend=answer_B, teammate_argument=argument_B1)
    final_argument_B = llm_call(prompt_B2, role="advocate_final_B")
    print(f"\nTeam B's Final Argument:\n{final_argument_B[:400]}...")

    # --- The Judge evaluates the arguments ---
    # print("\n--- The Judge is evaluating the arguments... ---")
    judge_prompt = JUDGE_PROMPT.format(
        system_prompt=system_prompt,
        input_question=input_question,
        answer_A_short=answer_A[:80] + "...",
        final_argument_A=final_argument_A,
        answer_B_short=answer_B[:80] + "...",
        final_argument_B=final_argument_B
    )
    judge_response = llm_call(judge_prompt, role="judge")
    print("\nJudge's Full Response:")
    print(judge_response)

    # --- Parse results and determine winner ---
    parsed_scores = parse_judge_scores(judge_response)    
    winner = 'tie'
    score_A = parsed_scores.get('A_total_score')
    score_B = parsed_scores.get('B_total_score')
    
    if score_A is not None and score_B is not None:
        if score_A > score_B:
            winner = 'llama_output'
        elif score_B > score_A:
            winner = 'distill_llama_output'
    else:
        winner = 'parse_error'
        
    # --- Collate all results into a dictionary ---
    result = {
        "input_question": input_question,
        "winner": winner,
        "judge_raw_response": judge_response,
        **{f"A_{k}": v for k, v in parsed_scores.items() if k.startswith('A_')}, # Add all A scores
        **{f"B_{k}": v for k, v in parsed_scores.items() if k.startswith('B_')}, # Add all B scores
        "final_argument_A": final_argument_A,
        "final_argument_B": final_argument_B
    }
    result = {k.replace('A_A_', 'A_').replace('B_B_', 'B_'): v for k, v in result.items()}
    
    return result

def load_data_from_json(filepath: str):
    """Loads data from a standard JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# --- Main Execution ---

if __name__ == "__main__":
    
    # Load the OpenQA dataset from a local file
    try:
        print("Loading openQA dataset from 'openQA_cleaned.json'...")
        all_data = load_data_from_json('openQA_cleaned.json')
        print(f"Dataset loaded. Found {len(all_data)} entries.")
    except FileNotFoundError:
        print("Error: 'openQA_cleaned.json' not found. Please create this file and add your data.")
        exit()
        
    all_results = []
    if NUM_SAMPLES > len(all_data):
        print(f"Warning: You asked for {NUM_SAMPLES} samples, but the file only contains {len(all_data)}. Running on all available data.")
        data_to_process = all_data
    else:
        data_to_process = all_data[:NUM_SAMPLES]
    # Process each entry in the dataset
    for i, data_point in enumerate(data_to_process):
        print(f"\n\n===== PROCESSING ENTRY {i+1}/{len(data_to_process)} =====")
        debate_result = run_more_debate(data_point)
        all_results.append(debate_result)
        
        print("\n" + "="*50 + "\n  DEBATE SUMMARY\n" + "="*50)
        print(f"Question: {debate_result['input_question'][:100]}...")
        print(f"Total Score for Team A (llama_output): {debate_result.get('A_total_score', 'N/A')}")
        print(f"Total Score for Team B (distill_llama_output): {debate_result.get('B_total_score', 'N/A')}")
        print(f"System's Judgment: '{debate_result['winner']}'")

    # Save results to JSON file
    json_output_path = 'QA_debate_results1.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ All results saved to {json_output_path}")

    # Save results to CSV file
    csv_output_path = 'QA_debate_results1.csv'
    if all_results:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"✅ All results also saved to {csv_output_path}")