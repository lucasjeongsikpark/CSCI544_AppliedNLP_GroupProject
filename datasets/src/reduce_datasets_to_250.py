import pandas as pd
import numpy as np
import json

# File paths

math_csv = "datasets/data/math_cleaned.csv"
openqa_csv = "datasets/data/openQA_cleaned.csv"
math_out_csv = "datasets/data/math_cleaned_250.csv"
openqa_out_csv = "datasets/data/openQA_cleaned_250.csv"
math_out_json = "datasets/data/math_cleaned_250.json"
openqa_out_json = "datasets/data/openQA_cleaned_250.json"

# Helper to sample and save
def reduce_and_save(csv_path, out_csv, out_json):
    df = pd.read_csv(csv_path)
    # Keep top 2 rows, sample 248 from the rest
    top2 = df.iloc[:2]
    rest = df.iloc[2:]
    sampled = rest.sample(n=248, random_state=42)
    reduced = pd.concat([top2, sampled], ignore_index=True)
    reduced.to_csv(out_csv, index=False)
    # Save as JSON (list of dicts)
    records = reduced.to_dict(orient="records")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

reduce_and_save(math_csv, math_out_csv, math_out_json)
reduce_and_save(openqa_csv, openqa_out_csv, openqa_out_json)
print("Done: created reduced CSV and JSON files for both datasets.")
