"""HumanEval (openai/openai_humaneval) task adapter.

Data: `data/HumanEval/test.parquet` (download via `python data/HumanEval/download.py`).
Official source: https://huggingface.co/datasets/openai/openai_humaneval

Evaluation: execution-based pass@1 (default flow: a single inference per
sample, no resampling). The model's extracted Python code is concatenated
with the example's `test` source and `check(entry_point)` is executed in a
subprocess. `score == 1.0` iff the subprocess exits cleanly.

Splitting: HumanEval ships only a single 164-problem split. To support the
two-phase train/test workflow (memory build on train, frozen eval on test),
the loader deterministically shuffles all 164 problems by ``sample_seed``
and returns the first half as ``train`` and the second half as ``test``.
"""

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from prompts.prompt_loader import get_prompt
from tasks.base_task import BaseTask
from tasks.humaneval_code_exec import evaluate as humaneval_evaluate
from tasks.humaneval_code_exec import extract_python_code


class HumanEvalTask(BaseTask):
    """openai/openai_humaneval test split: 164 hand-written Python problems."""

    name = "HumanEval"
    prompt_key = "humaneval"

    # --- knobs main.py can override after construction ---
    humaneval_timeout: int = 10  # seconds per problem

    def __init__(
        self,
        prompt_file: Optional[str] = None,
        data_file: Optional[str] = None,
        split: str = "full",
        sample_seed: int = 0,
    ):
        self.prompt_file = prompt_file
        root = Path(__file__).resolve().parents[1]
        self.data_file = (
            Path(data_file) if data_file else root / "data" / "HumanEval" / "test.parquet"
        )
        self.split = str(split or "full").strip().lower()
        self.sample_seed = int(sample_seed)
        self._entries: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _split_mode(self) -> str:
        if self.split in {"train", "trainset", "memory", "support"}:
            return "train"
        if self.split in {"test", "testset", "eval", "validation", "val"}:
            return "test"
        if self.split in {"full", "all", "raw", ""}:
            return "full"
        raise ValueError(
            "Unsupported HumanEval split. Use train, test, or full. "
            "train/test produce a deterministic 50/50 disjoint split of "
            "all 164 problems, seeded by --sample-seed."
        )

    def _stable_shuffle(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        digest = hashlib.sha256(
            f"HumanEval:{self.sample_seed}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        rows = list(rows)
        rng.shuffle(rows)
        return rows

    def _load(self) -> None:
        if self._entries is not None:
            return
        if not self.data_file.exists():
            raise RuntimeError(
                f"HumanEval data file not found: {self.data_file}\n"
                f"Run: python {self.data_file.parent}/download.py"
            )
        frame = pd.read_parquet(self.data_file)
        entries: List[Dict[str, Any]] = []
        for idx, row in enumerate(frame.to_dict(orient="records")):
            item = dict(row)
            # Use task_id as qid so logs are human-friendly ("HumanEval/23").
            item["_qid"] = str(item.get("task_id") or f"row-{idx}")
            entries.append(item)

        mode = self._split_mode()
        if mode != "full":
            shuffled = self._stable_shuffle(entries)
            train_size = 50
            entries = shuffled[:train_size] if mode == "train" else shuffled[train_size:]
            # half = len(shuffled) // 2
            # entries = shuffled[:half] if mode == "train" else shuffled[half:]
        self._entries = entries

    # ------------------------------------------------------------------ #
    # Field accessors
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prompt_src(item: Dict[str, Any]) -> str:
        return str(item.get("prompt") or "")

    @staticmethod
    def _test_src(item: Dict[str, Any]) -> str:
        return str(item.get("test") or "")

    @staticmethod
    def _entry_point(item: Dict[str, Any]) -> str:
        return str(item.get("entry_point") or "").strip()

    # ------------------------------------------------------------------ #
    # BaseTask API
    # ------------------------------------------------------------------ #

    def iter_entries(self) -> Iterable[Dict[str, Any]]:
        self._load()
        for item in self._entries or []:
            yield item

    def total_entries(self) -> int:
        self._load()
        return len(self._entries or [])

    def build_prompt(self, entry: Dict[str, Any]) -> str:
        return get_prompt(
            self.prompt_key,
            self.prompt_file,
            question=self._prompt_src(entry),
            entry_point=self._entry_point(entry),
        )

    def build_inputs(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "qid": entry.get("_qid", ""),
            "task_id": str(entry.get("task_id") or ""),
            "question": self._prompt_src(entry),
            "entry_point": self._entry_point(entry),
            "test": self._test_src(entry),
        }

    def get_query(self, entry: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        # Retrieval (HistoryRAG etc.) uses the function signature + docstring.
        return str(inputs.get("question") or self._prompt_src(entry))

    def evaluate_entry(self, output: str, entry: Dict[str, Any]) -> float:
        code = extract_python_code(output)
        result = humaneval_evaluate(
            user_code=code,
            test_src=self._test_src(entry),
            entry_point=self._entry_point(entry),
            timeout=int(self.humaneval_timeout),
        )
        self._last_eval_info = {
            "extracted_code": code,
            "passed_cases": result["passed"],
            "total_cases": result["total"],
            "pass_rate": result["pass_rate"],
            "first_error": result["first_error"],
            "stderr_tail": result.get("stderr_tail", ""),
            "mode": result["mode"],
        }
        return 1.0 if result["all_pass"] else 0.0

    def build_memory_record(
        self,
        entry: Dict[str, Any],
        output: str,
        feedback: str,
        score: float,
    ) -> Dict[str, Any]:
        info = getattr(self, "_last_eval_info", None) or {}
        passed = info.get("passed_cases", 0)
        total = info.get("total_cases", 1)
        first_err = info.get("first_error", "")
        human_feedback = (
            "success: check() passed"
            if score >= 1.0
            else "failure: check() raised"
            + (f" | first_error={first_err}" if first_err else "")
        )
        return {
            "qid": entry.get("_qid", ""),
            "task_id": str(entry.get("task_id") or ""),
            "entry_point": self._entry_point(entry),
            "question": self._prompt_src(entry),
            "test": self._test_src(entry),
            "model_output": str(output),
            "extracted_code": info.get("extracted_code", ""),
            "passed_cases": passed,
            "total_cases": total,
            "pass_rate": info.get("pass_rate", 0.0),
            "first_error": first_err,
            "stderr_tail": info.get("stderr_tail", ""),
            "feedback": human_feedback,
            "score": score,
        }

    # ------------------------------------------------------------------ #
    # Abstract stubs (BaseTask requires them; HumanEval uses streaming)
    # ------------------------------------------------------------------ #

    def get_prompt(self):
        raise NotImplementedError("HumanEval uses build_prompt(entry) in stream mode.")

    def get_inputs(self):
        raise NotImplementedError("HumanEval uses build_inputs(entry) in stream mode.")

    def evaluate(self, output):
        raise NotImplementedError("HumanEval uses evaluate_entry(output, entry).")