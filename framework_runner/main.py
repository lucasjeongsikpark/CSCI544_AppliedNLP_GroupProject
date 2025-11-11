import enum

from framework_runner.debate_impl import debate_framework
from framework_runner.runner import FrameworkRunner

MED_ASPECTS = [
    "1. Medical Accuracy: Is the information medically sound?",
    "2. Appropriateness: Is the advice appropriate for the described situation?",
    "3. Safety: Does it prioritize patient safety?",
    "4. Clarity: Is the explanation clear and understandable?",
    "5. Professionalism: Does it maintain appropriate professional tone?",
]

MATH_ASPECTS = [
    "1. Correctness: Does it arrive at the correct answer?",
    "2. Reasoning: Is the step-by-step reasoning clear and logical?",
    "3. Completeness: Are all necessary steps shown?",
    "4. Accuracy: Are calculations correct?"
]


OPEN_QA_ASPECTS = [
    "1. Relevance: Does it address the question appropriately?",
    "2. Completeness: Is the response comprehensive and detailed?",
    "3. Accuracy: Is the information correct compared to the reference?",
    "4. Clarity: Is the response well-structured and clear?",
    "5. Helpfulness: Would this response be useful to the user?"
]


class DatasetTypes(enum.Enum):
    OPEN_QA = "open_qa"
    MED = "med"
    MATH = "math"

    def get_dataset_path(self) -> str | None:
        if self == self.MED:
            return "datasets/data/med_cleaned.json"
        elif self == self.MATH:
            return "datasets/data/math_cleaned_250.json"
        elif self == self.OPEN_QA:
            return "datasets/data/openQA_cleaned_250.json"

    def get_aspects(self) -> list[str] | None:
        if self == self.MED:
            return MED_ASPECTS
        elif self == self.MATH:
            return MATH_ASPECTS
        elif self == self.OPEN_QA:
            return OPEN_QA_ASPECTS


if __name__ == "__main__":
    for dataset_type in DatasetTypes:
        framework_runner = FrameworkRunner(
            framework=debate_framework,  # specify your framework here
            output_file=f"results/{debate_framework.name}_{dataset_type.value}.jsonl",
            dataset_path=dataset_type.get_dataset_path(),
            aspects=dataset_type.get_aspects(),
        )
        start_from = 0
        while True:
            try:
                framework_runner.evaluate_dataset(start_from=start_from)
            except Exception as e:
                if framework_runner.last_saved_ind == start_from:
                    print(f"Error on dataset {dataset_type.name} index {start_from}: {e}")
                    break
                print(f"Retrying on dataset {dataset_type.name} from index {framework_runner.last_saved_ind}")
                start_from = framework_runner.last_saved_ind
            else:
                print(f"Completed {dataset_type.name} dataset")
                break
