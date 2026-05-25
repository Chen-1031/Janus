from datasets import load_dataset

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask


class AIME2024Task(BaseTask):
    name = "AIME2024"
    prompt_key = "aime2024"

    def __init__(
        self,
        prompt_file=None,
        hf_path="Maxwell-Jia/AIME_2024",
        split="train",
    ):
        self.prompt_file = prompt_file
        self.hf_path = hf_path
        self.split = split
        self._dataset = None

    def _load(self):
        if self._dataset is not None:
            return
        self._dataset = load_dataset(self.hf_path, split=self.split)

    def _question(self, item):
        # AIME_2024 uses "Problem" as the question field.
        return (
            item.get("Problem", "")
            or item.get("question", "")
            or item.get("Question", "")
        )

    def _answer(self, item):
        # Primary label is "Answer" (int64 in the dataset card).
        answer = item.get("Answer", "")
        if answer == "":
            answer = item.get("answer", "")
        if answer == "":
            answer = item.get("Solution", "")
        if answer == "":
            answer = item.get("solution", "")
        return str(answer).strip()

    def _solution(self, item):
        return item.get("Solution", "") or item.get("solution", "")

    def _qid(self, item, idx):
        for key in ("ID", "id", "problem_id"):
            if key in item:
                return str(item[key])
        return str(idx)

    def iter_entries(self):
        self._load()
        for idx, item in enumerate(self._dataset):
            item_copy = dict(item)
            item_copy["_qid"] = self._qid(item_copy, idx)
            yield item_copy

    def total_entries(self):
        self._load()
        return len(self._dataset)

    def build_prompt(self, entry):
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
        )

    def build_inputs(self, entry):
        return {
            "question": self._question(entry),
            "answer": self._answer(entry),
            "solution": self._solution(entry),
            "qid": entry.get("_qid", ""),
        }

    def evaluate_entry(self, output, entry):
        answer = self._answer(entry)
        return 1.0 if answer and answer in str(output) else 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "question": self._question(entry),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

    def get_prompt(self):
        raise NotImplementedError("AIME2024 uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("AIME2024 uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("AIME2024 uses evaluate_entry(output, entry).")
