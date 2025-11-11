from DEBATE.debate import create_debate_evaluator
from framework_runner.base import Framework, DataOutput


class DebateFramework(Framework):
    @property
    def evaluator(self):
        return create_debate_evaluator(max_iterations=3)

    def run(self, data: dict, aspects: list[str]) -> DataOutput:
        data_output = {}
        chat_logs = {}
        for key in ["llama_output", "distill_llama_output"]:
            chat_logs[key] = {}

            result = self.evaluator.evaluate_dialogue(
                context=data.get("document", "") + data.get("system_prompt", "") + data["input"],
                response=data.get(key),
                aspects=aspects
            )

            scores = {}
            for aspect, res in result.items():
                scores[aspect] = res["final_score"]
                chat_logs[key][aspect] = result[aspect]["debate_history"]

            if key == "llama_output":
                data_output["score1"] = scores
            else:
                data_output["score2"] = scores

        data_output["chat_logs"] = chat_logs
        data_output["attempts"] = 2 * len(aspects)  # fixed number of runs for my framework
        return DataOutput(**data_output)


debate_framework = DebateFramework(name="DEBATE")
