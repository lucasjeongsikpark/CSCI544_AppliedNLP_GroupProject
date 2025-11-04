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

        # use ndjson so we can append results periodically
        assert output_file.split(".", 1)[1] == "ndjson"
        self.output_file = self.output_file

    def evaluate_dataset(self):
        results = []
        for ind, row in self.dataset.iterrows():
            if ind < self.start_from:
                continue

            data = row.to_dict()
            start = datetime.now()
            result = self.framework.run(data=data)
            elapsed_time = datetime.now() - start
            results.append(LoggedOutput.new(elapsed_time=elapsed_time, data_output=result).to_dict())

            if ind % 10 == 0:
                self.save_results(results=results)
                results = []
                print("FrameworkRunner: saved results for index %s", ind)

        self.save_results(results=results)
        print("FrameworkRunner: saved results for index %s", ind)

    def save_results(self, results: list[dict]):
        with open(self.output_file, "a", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
