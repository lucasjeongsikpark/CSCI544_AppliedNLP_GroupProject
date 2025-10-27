#!/usr/bin/env python3
"""
View the N rows with the smallest `input` value (by string length),
along with their corresponding `output` values, for either med.csv or openQA.csv.

Usage:
    # Choose dataset by name (auto-picks default path)
    python3 view_smallest_med_inputs.py --dataset med --top 5
    python3 view_smallest_med_inputs.py --dataset openqa --top 5

    # Or provide a custom CSV path (overrides --dataset)
    python3 view_smallest_med_inputs.py --data ../data/med.csv --top 5

Notes:
- Expects columns: input, output (med also has document; openQA also has system_prompt)
- Sorts by len(input) ascending; ties keep original order
- Prints a compact table to stdout
"""
import argparse
import os
import sys
from typing import Optional

import pandas as pd


def load_csv_safe(path: str) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read {path} with common encodings. Last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["med", "openqa"], default="med",
                    help="Which dataset to use if --data is not provided")
    ap.add_argument("--data", default=None,
                    help="Optional explicit CSV path; overrides --dataset")
    ap.add_argument("--top", type=int, default=5, help="Number of rows to display")
    args = ap.parse_args()

    # Resolve data path
    data_path = args.data
    if data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_map = {
            "med": os.path.join(base_dir, "..", "data", "med.csv"),
            "openqa": os.path.join(base_dir, "..", "data", "openQA.csv"),
        }
        data_path = default_map[args.dataset]

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = load_csv_safe(data_path)

    if "input" not in df.columns or "output" not in df.columns:
        print("❌ Expected columns 'input' and 'output' in the CSV.", file=sys.stderr)
        print(f"Columns found: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    # Compute input length (handle NaN safely)
    inputs = df["input"].fillna("")
    df = df.copy()
    df["input_len"] = inputs.astype(str).str.len()

    # Sort by input length ascending and take top N
    smallest = df.sort_values(["input_len"], ascending=[True]).head(args.top)

    # Build a compact view
    view_cols = ["input_len", "input", "output"]
    # Truncate long fields for readability in terminal
    def trunc(s: str, max_len: int = 140) -> str:
        s = str(s)
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    print(f"Showing {len(smallest)} smallest inputs from: {data_path}\n")
    for idx, row in smallest.iterrows():
        print("-" * 80)
        print(f"Row index: {idx} | input_len: {row['input_len']}")
        print("INPUT:")
        print(trunc(row["input"]))
        print()
        print("OUTPUT:")
        print(trunc(row["output"]))
    print("-" * 80)


if __name__ == "__main__":
    main()
