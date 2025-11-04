import abc
import dataclasses
from typing import Any


@dataclasses.dataclass
class DataOutput:
    chat_logs: Any  # up to your discretion based on your framework
    score1: dict[str, int]
    score2: dict[str, int]
    attempts: int

    def to_dict(self) -> dict:
        return {
            "chat_log": self.chat_logs,
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
            chat_logs=data_output.chat_logs,
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
    def run(self, data: dict, aspects: list[str]) -> DataOutput:
        """Takes in single row of data + aspects to evaluate on and outputs final scores with number of attempts"""
        raise NotImplemented