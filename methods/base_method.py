from abc import ABC, abstractmethod
import copy

from utils.io import write_json, write_jsonl


class BaseMethod(ABC):
    name = "BaseMethod"

    def __init__(self):
        self.memory = []

    def _snapshot_record(self, record):
        # Avoid recursive expansion when exporting per-step memory snapshots.
        return {k: v for k, v in record.items() if k not in {"memory"}}

    def build_task_memory_snapshot(self, task_name):
        rows = [row for row in self.memory if row.get("task") == task_name]
        return [self._snapshot_record(row) for row in rows]

    def get_latest_task_record(self, task_name):
        for row in reversed(self.memory):
            if row.get("task") == task_name:
                return copy.deepcopy(row)
        return None

    def reset_task_state(self, task):
        # Optional hook for methods that keep per-task states.
        return None

    def get_current_memory(self, task):
        # Optional hook returning the memory currently deployed for ``task``.
        # Methods that maintain a single rolling memory artifact (e.g. a
        # cheatsheet string for DC-RS or an insight list for ExpeL) should
        # override this so that callers can snapshot the post-update state.
        # Returning ``None`` opts out of per-step memory logging.
        return None

    # ------------------------------------------------------------------ test
    # Batched test-time interface. Any method that wants the batched
    # ``run_batched_test`` fast path (vLLM continuous batching over the
    # entire test split) must override ``test_build_solver_prompt``. The
    # contract at test time is "freeze memory, but still allow per-question
    # embedding retrieval": implementations MUST NOT mutate their internal
    # state in this method and SHOULD use ``retrieved`` -- the result of
    # :meth:`retrieve_memory` for the current question -- when relevant.
    def test_build_solver_prompt(self, task, entry, retrieved, inputs):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement test_build_solver_prompt. "
            "Either add it or disable the batched test fast path via --test-no-batch."
        )

    def test_generation_model_name(self) -> str:
        return str(getattr(self, "generation_model_name", ""))

    def test_generation_params(self) -> dict:
        return {
            "temperature": float(getattr(self, "temperature", 0.0)),
            "max_new_tokens": int(getattr(self, "max_new_tokens", 512)),
        }

    def supports_batched_test(self) -> bool:
        fn = getattr(type(self), "test_build_solver_prompt", None)
        base_fn = BaseMethod.test_build_solver_prompt
        return callable(fn) and fn is not base_fn

    def get_timing_adjustments(self, task):
        # Optional hook for stage re-attribution.
        return {}

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        # Optional high-fidelity trial hook. Return float score when handled.
        return None

    def finalize_task_state(self, task):
        # Optional hook for end-of-task finalization.
        return None

    def export_task_memory(self, task_name, output_path):
        rows = [row for row in self.memory if row.get("task") == task_name]
        write_jsonl(output_path, rows)
        readable_path = output_path.with_name(f"{output_path.stem}_readable.json")
        write_json(readable_path, rows)

    @abstractmethod
    def retrieve_memory(self, task, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate(self, task, prompt, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_memory(self, task, output, **kwargs):
        raise NotImplementedError
