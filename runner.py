import json
import random
from pathlib import Path

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        return iterable


class Runner:
    def __init__(self, method, tasks, timer, output_dir):
        self.method = method
        self.tasks = tasks
        self.timer = timer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _phase_filename(base_name, phase_label):
        if not phase_label:
            return base_name
        if base_name.endswith("_memory.jsonl"):
            stem = base_name[: -len("_memory.jsonl")]
            return f"{stem}_{phase_label}_memory.jsonl"
        return f"{phase_label}_{base_name}"

    @staticmethod
    def _predictions_filename(memory_filename, phase_label):
        suffix = "_predictions.jsonl"
        if memory_filename.endswith("_memory.jsonl"):
            stem = memory_filename[: -len("_memory.jsonl")]
        else:
            stem = memory_filename
        if phase_label:
            return f"{stem}_{phase_label}{suffix}"
        return f"{stem}{suffix}"

    @staticmethod
    def _step_memory_filename(memory_filename):
        # Always co-locate with the regular memory file, but use a stable
        # suffix that does not include the phase label (the parent directory
        # already disambiguates train/test). This matches the layout the
        # spec asks for: <task>_<method>_step_memory.jsonl
        if memory_filename.endswith("_memory.jsonl"):
            stem = memory_filename[: -len("_memory.jsonl")]
        else:
            stem = memory_filename
        return f"{stem}_step_memory.jsonl"

    def _materialize_entries(self, task, sample_size, sample_seed, shuffle):
        entries = list(task.iter_entries())
        if shuffle:
            rng = random.Random(int(sample_seed))
            rng.shuffle(entries)
        if sample_size is not None and len(entries) > sample_size:
            entries = entries[: int(sample_size)]
        return entries

    def run(
        self,
        update_memory=True,
        run_first_k=None,
        sample_size=None,
        sample_seed=0,
        shuffle_sampling=False,
        reset_task_state=True,
        phase_label="",
        log_predictions=False,
    ):
        results = []
        for task in self.tasks:
            task_name = task.name
            if reset_task_state:
                self.method.reset_task_state(task)
            total = 0
            success = 0
            limit = max(0, int(run_first_k)) if run_first_k is not None else None

            if sample_size is not None or shuffle_sampling:
                entries_iter = self._materialize_entries(
                    task, sample_size, sample_seed, shuffle_sampling
                )
                expected_total = len(entries_iter)
                if limit is not None:
                    expected_total = min(expected_total, limit)
            else:
                entries_iter = task.iter_entries()
                expected_total = task.total_entries()
                if limit is not None and expected_total is not None:
                    expected_total = min(expected_total, limit)
                elif limit is not None and expected_total is None:
                    expected_total = limit

            desc = f"{self.method.name}/{task_name}"
            if phase_label:
                desc = f"{desc}[{phase_label}]"
            progress = tqdm(
                entries_iter,
                desc=desc,
                unit="sample",
                total=expected_total,
                dynamic_ncols=True,
            )
            if not TQDM_AVAILABLE:
                print(
                    f"[Progress disabled] Install tqdm to show progress bar: {desc}"
                )

            base_memory_name = task.memory_filename(self.method.name)
            memory_out_name = self._phase_filename(base_memory_name, phase_label)
            memory_out_path = self.output_dir / memory_out_name
            pred_path = None
            predictions = []
            if log_predictions:
                pred_path = self.output_dir / self._predictions_filename(
                    base_memory_name, phase_label
                )
                pred_path.write_text("", encoding="utf-8")

            step_memory_path = None
            if update_memory:
                step_memory_path = self.output_dir / self._step_memory_filename(
                    base_memory_name
                )
                memory_out_path.write_text("", encoding="utf-8")
                # Start a fresh per-run JSONL log, then append one row per step.
                step_memory_path.write_text("", encoding="utf-8")

            for entry in progress:
                if limit is not None and total >= limit:
                    break
                with self.timer.track(f"{self.method.name}/{task_name}/sample_total"):
                    total += 1
                    inputs = task.build_inputs(entry)

                    score = self.method.run_trial(
                        task=task,
                        entry=entry,
                        inputs=inputs,
                        timer=self.timer,
                        update_memory=update_memory,
                    )

                    output_text = None
                    feedback = None
                    if score is None:
                        prompt = task.build_prompt(entry)
                        query = task.get_query(entry, inputs)

                        with self.timer.track(f"{self.method.name}/{task_name}/retrieve"):
                            memory = self.method.retrieve_memory(task, query=query, entry=entry, **inputs)

                        with self.timer.track(f"{self.method.name}/{task_name}/generate"):
                            output = self.method.generate(task, prompt, memory=memory, entry=entry, **inputs)

                        score = task.evaluate_entry(output, entry)
                        feedback = "success" if score >= 1.0 else "failure"
                        output_text = str(output)

                        if update_memory:
                            record = task.build_memory_record(entry, output, feedback, score)
                            with self.timer.track(f"{self.method.name}/{task_name}/update"):
                                self.method.update_memory(task, output, record=record, entry=entry, **inputs)

                    success += int(score >= 1.0)

                    if log_predictions and pred_path is not None:
                        predictions.append(
                            {
                                "qid": inputs.get("qid", ""),
                                "question": inputs.get("question", ""),
                                "model_output": output_text or "",
                                "score": float(score),
                                "feedback": feedback
                                or ("success" if score >= 1.0 else "failure"),
                            }
                        )
                        with pred_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(predictions[-1], ensure_ascii=False) + "\n")

                    if update_memory:
                        latest_record = self.method.get_latest_task_record(task_name)
                        if latest_record is not None:
                            with memory_out_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(latest_record, ensure_ascii=False) + "\n")

                        if step_memory_path is not None:
                            current_memory = self.method.get_current_memory(task)
                            if current_memory is not None:
                                with step_memory_path.open("a", encoding="utf-8") as f:
                                    f.write(
                                        json.dumps(
                                            {"step": total, "memory": current_memory},
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )

            accuracy = (success / total) if total else 0.0
            self.method.finalize_task_state(task)
            results.append(
                {
                    "task": task_name,
                    "method": self.method.name,
                    "phase": phase_label or None,
                    "total": total,
                    "success": success,
                    "accuracy": accuracy,
                }
            )

            if update_memory:
                # Final export keeps compatibility with the existing
                # *_memory_readable.json artifact.
                self.method.export_task_memory(task_name, memory_out_path)
        return results
