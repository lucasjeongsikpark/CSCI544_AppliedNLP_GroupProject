from openai import OpenAI
import pandas as pd
from tqdm import tqdm

# ==============================
# MODEL CONFIGURATION
# ==============================
MODEL_CONFIGS = {
    "llama": {
        "base_url": "http://localhost:8080/v1",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "distill_llama": {
        "base_url": "http://localhost:8081/v1",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # replace with actual model
    },
}

# ==============================
# GENERAL INFERENCE FUNCTION
# ==============================
def run_inference(df, dataset_type, model_key="llama", system_prompt=None, save_path=None):
    """
    Run inference on given dataset using specified model and configuration.

    Args:
        df (pd.DataFrame): Input dataframe.
        dataset_type (str): One of ["gsm8k", "medredqa", "openorca"].
        model_key (str): Which model to use ("llama" or "huggingface").
        system_prompt (str): Optional system prompt (used for gsm8k, medredqa).
        save_path (str): Optional CSV path to save results.

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    config = MODEL_CONFIGS[model_key]
    llm = OpenAI(base_url=config["base_url"], api_key="EMPTY")
    model_name = config["model_name"]

    preds = []

    # system_prompt_add = " Include the assistant's reasoning and final answer only. Do not include any user messages or system instructions."
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Inferencing {dataset_type}"):
        # --- Construct message list depending on dataset ---
        if dataset_type == "gsm8k":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["input"]}
            ]
        elif dataset_type == "medredqa":
            user_content = f"Context:\n{row['document']}\n\nQuestion:\n{row['input']}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        elif dataset_type == "openorca":
            messages = [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["input"]}
            ]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # --- Query model ---
        try:
            resp = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2048,
            )
            output = resp.choices[0].message.content
        except Exception as e:
            output = f"Error: {str(e)}"

        preds.append(output)

    df[f"{model_key}_output"] = preds

    # --- Save results ---
    if save_path:
        df.to_csv(f"../data/{model_key}_{save_path}", index=False, encoding="utf-8-sig")
        print(f"✅ Saved predictions to {save_path}")

    return df


# ==============================
# EXAMPLE USAGE
# ==============================

# Example system prompts
gsm8k_system_prompt = """You are a math reasoning assistant. 
For each problem, think step by step and show your reasoning. 
At the end, write “#### [final answer]” where [final answer] is the numeric result.
"""

medredqa_system_prompt = """Answer the question based on the following context."""

# Example: Run for GSM8K
df_gsm8k = pd.read_csv("../data/math.csv")
df_gsm8k = df_gsm8k[0:2]
gsm8k_results = run_inference(df_gsm8k, "gsm8k", model_key="llama", system_prompt=gsm8k_system_prompt, save_path="math.csv")

# Example: Run for MedRedQA
df_medredqa = pd.read_csv("../data/med.csv")
df_medredqa = df_medredqa[0:2]
medredqa_results = run_inference(df_medredqa, "medredqa", model_key="llama", system_prompt=medredqa_system_prompt, save_path="med.csv")

# Example: Run for OpenOrca
df_openorca = pd.read_csv("../data/openQA.csv")
df_openorca = df_openorca[0:2]
openorca_results = run_inference(df_openorca, "openorca", model_key="llama", save_path="openQA.csv")
