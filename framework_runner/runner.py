import json
import os
from datetime import datetime
from typing import List

import pandas as pd

from .base import Framework, LoggedOutput



class FrameworkRunner:
    """Generic runner for a Framework implementation.
    ...existing docstring...
    """
    def __init__(self,
                 framework: Framework,
                 dataset_path: str,
                 output_file: str,
                 aspects: List[str],
                 start_from: int = 0,
                 auto_resume: bool = True,
                 limit: int = None):
        self.framework = framework
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset(dataset_path)
        self.aspects = aspects
        self.output_file = output_file
        self.progress_file = output_file + ".progress"
        self.start_from = start_from
        self.last_saved_ind = start_from - 1
        self.auto_resume = auto_resume
        self.limit = limit

        assert output_file.endswith(".ndjson") or output_file.endswith(".jsonl"), \
            "output_file must end with .ndjson or .jsonl"

        if auto_resume and start_from == 0:
            detected = self._detect_resume_index()
            if detected > 0:
                print(f"[FrameworkRunner] Auto-resume detected last index {detected}; starting from {detected+1}")
                self.start_from = detected + 1
                self.last_saved_ind = detected

    def _load_dataset(self, path: str) -> pd.DataFrame:
        if path.endswith('.json'):
            return pd.read_json(path)
        if path.endswith('.csv'):
            return pd.read_csv(path)
        raise ValueError(f"Unsupported dataset format: {path}. Use .json or .csv")

    def _detect_resume_index(self) -> int:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as pf:
                    val = pf.read().strip()
                return int(val)
            except Exception:
                pass
        if not os.path.exists(self.output_file):
            return -1
        try:
            with open(self.output_file, 'r') as f:
                lines = sum(1 for _ in f)
            return lines - 1
        except Exception:
            return -1

    def evaluate_dataset(self):
        processed = 0
        for ind, row in self.dataset.iterrows():
            if ind < self.start_from:
                continue
            if self.limit is not None and processed >= self.limit:
                print(f"[FrameworkRunner] Limit {self.limit} reached. Stopping.")
                break
            data = row.to_dict()
            print(f"[FrameworkRunner] Processing index {ind}")
            start = datetime.now()
            try:
                result = self.framework.run(data=data, aspects=self.aspects)
            except Exception as e:
                print(f"[FrameworkRunner] ERROR at index {ind}: {e}. Saving progress and aborting.")
                self._write_progress()
                raise
            elapsed_time = (datetime.now() - start).total_seconds()
            logged = LoggedOutput.new(elapsed_time=elapsed_time, data_output=result)
            # Print all output fields for inspection
            out_dict = logged.to_dict()
            print(f"[FrameworkRunner] Output for index {ind}:")
            for k in ["score1", "score2", "chat_log", "attempts", "elapsed_time"]:
                v = out_dict.get(k)
                print(f"  {k}: {v}")
                if (k in ["score1", "score2"] and (not v or not isinstance(v, dict) or not any(v.values()))) or (k in ["chat_log"] and not v):
                    print(f"  [ALERT] {k} is missing or empty!")
            self.save_results([logged])
            self.last_saved_ind = ind
            self._write_progress()
            processed += 1
        print("[FrameworkRunner] Completed dataset.")

    def save_results(self, results: List[LoggedOutput]):
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')

    def _write_progress(self):
        try:
            with open(self.progress_file, 'w') as pf:
                pf.write(str(self.last_saved_ind))
        except Exception as e:
            print(f"[FrameworkRunner] Failed to write progress file: {e}")
