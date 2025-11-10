import json
from datetime import datetime

import pandas as pd

from CSCI544_AppliedNLP_GroupProject.framework_runner.base import Framework, LoggedOutput


class FrameworkRunner:
    def __init__(
            self,
            framework: Framework,
            dataset_path: str,
            output_file: str,
            aspects: list[str],
            start_from: int = 0
    ):
        self.framework = framework
        self.dataset = pd.read_json(dataset_path)
        self.start_from = start_from
        self.aspects = aspects

        # use ndjson or jsonl so we can append results periodically
        assert output_file.split(".", 1)[1] == "ndjson" or output_file.split(".", 1)[1] == "jsonl"
        self.output_file = output_file

    def evaluate_dataset(self, start_from: int = 0):
        start_from = max(self.start_from, start_from)
        for ind, row in self.dataset.iterrows():
            if ind < start_from:
                continue

            data = row.to_dict()
            print("FrameworkRunner: starting index", ind)
            start = datetime.now()
            result = self.framework.run(data=data, aspects=self.aspects)
            elapsed_time = (datetime.now() - start).total_seconds()

            self.save_results(results=[LoggedOutput.new(elapsed_time=elapsed_time, data_output=result)])
            print("FrameworkRunner: saved results for index", ind)
            self.last_saved_ind = ind

    def save_results(self, results: list[LoggedOutput]):
        with open(self.output_file, "a", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
