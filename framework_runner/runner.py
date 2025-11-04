import abc
import dataclasses
import json
from datetime import datetime
import pandas as pd


@dataclasses.dataclass
class ChatLog:
    role: str
    evaluation: str

    def to_dict(self):
        return {
            "role": self.role,
            "evaluation": self.evaluation
        }


@dataclasses.dataclass
class DataOutput:
    chat_log: ChatLog
    score1: dict[str, int]
    score2: dict[str, int]
    attempts: int

    def to_dict(self) -> dict:
        return {
            "chat_log": self.chat_log.to_dict(),
            "score1": self.score1,
            "score2": self.score2,
            "attempts": self.attempts,
            "elapsed_time": self.elapsed_time,
        }


@dataclasses.dataclass
class LoggedOutput(DataOutput):
    elapsed_time: float

    @classmethod
    def new(cls, elapsed_time: float, data_output: DataOutput):
        return cls(
            chat_log=data_output.chat_log,
            score1=data_output.score1,
            score2=data_output.score2,
            attempts=data_output.attempts,
            elapsed_time=elapsed_time,
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "elapsed_time": self.elapsed_time
        }


@dataclasses.dataclass
class Framework(abc.ABC):
    name: str

    @abc.abstractmethod
    def run(self, data: dict) -> DataOutput:
        """Takes in single row of data and outputs final scores with number of attempts"""
        raise NotImplemented


class FrameworkRunner(abc.ABC):
    def __init__(self, framework: Framework, dataset_path: str, output_file: str, start_from: int = 0):
        self.framework = framework
        self.dataset_path = dataset_path
        self.start_from = start_from

        # use ndjson so we can append results periodically
        assert output_file.split(".", 1)[1] == "ndjson"
        self.output_file = self.output_file

    def evaluate_dataset(self):
        df = pd.read_json("/Users/brendanhy/Downloads/csci544/CSCI544_AppliedNLP_GroupProject/datasets/data/math_cleaned.json")
        results = []
        for ind, row in df.iterrows():
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
                print("saved results for index %s", ind)

        self.save_results(results=results)
        print("saved results for index %s", ind)

    def save_results(self, results: list[dict]):
        with open(self.output_file, "a", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
