from pathlib import Path

import pandas as pd

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.math_answer_utils import normalized_last_boxed


class MATH500Task(BaseTask):
    name = "MATH500"
    prompt_key = "math"

    def __init__(self, prompt_file=None, data_dir=None, split="test"):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        # Prefer a dedicated folder, but also allow the common nested layout
        # used during quick conversions: data/MATH/math500/test-*.parquet.
        default_primary = root / "data" / "MATH500"
        default_fallback = root / "data" / "MATH" / "math500"
        self.data_dir = Path(data_dir) if data_dir else default_primary
        self.fallback_data_dir = default_fallback
        split = str(split or "test").strip().lower()
        # Convenience alias requested by users: --test-split 500
        self.split = "test" if split in {"500", "math500"} else split
        self._entries = None

    def _candidate_files(self):
        dirs = [self.data_dir]
        if self.fallback_data_dir != self.data_dir:
            dirs.append(self.fallback_data_dir)
        files = []
        for d in dirs:
            if not d.exists():
                continue
            # Accept either nested (<subject>/<split>-*.parquet) or flat
            # (<split>-*.parquet) layouts.
            files.extend(sorted(d.glob(f"*/{self.split}-*.parquet")))
            files.extend(sorted(d.glob(f"{self.split}-*.parquet")))
        return files

    def _load(self):
        if self._entries is not None:
            return
        split_files = self._candidate_files()
        if not split_files:
            raise RuntimeError(
                "No MATH500 parquet files found. Checked: "
                f"{self.data_dir} and {self.fallback_data_dir} "
                f"with split={self.split}"
            )

        entries = []
        for split_file in split_files:
            subject = split_file.parent.name if split_file.parent != self.data_dir else "math500"
            frame = pd.read_parquet(split_file)
            for idx, item in enumerate(frame.to_dict(orient="records")):
                row = dict(item)
                row["_subject"] = subject
                row["_qid"] = f"{subject}-{idx}"
                entries.append(row)
        self._entries = entries

    def _question(self, item):
        return item.get("problem", "") or item.get("question", "")

    def _solution(self, item):
        return item.get("solution", "") or item.get("answer", "") or item.get("Solution", "")

    def _gold_final_answer(self, item):
        return normalized_last_boxed(self._solution(item))

    def iter_entries(self):
        self._load()
        for item in self._entries:
            yield item

    def total_entries(self):
        self._load()
        return len(self._entries)

    def build_prompt(self, entry):
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._question(entry),
        )

    def build_inputs(self, entry):
        return {
            "qid": entry.get("_qid", ""),
            "subject": entry.get("_subject", ""),
            "question": self._question(entry),
            "solution": self._solution(entry),
            "gold_final_answer": self._gold_final_answer(entry),
        }

    def evaluate_entry(self, output, entry):
        pred = normalized_last_boxed(output)
        gold = self._gold_final_answer(entry)
        return 1.0 if pred and gold and pred == gold else 0.0

    def build_memory_record(self, entry, output, feedback, score):
        return {
            "qid": entry.get("_qid", ""),
            "subject": entry.get("_subject", ""),
            "question": self._question(entry),
            "gold_final_answer": self._gold_final_answer(entry),
            "model_output": str(output),
            "feedback": feedback,
            "score": score,
        }

