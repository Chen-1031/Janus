"""Batched test-time inference shared by ``test_only.py`` and ``main.py``.

Design
------
The legacy :class:`Runner` invokes the model once per sample. For test
(``update_memory=False``) every sample is independent, so we can stream
the entire split through vLLM in a single batched call per chunk.

``run_batched_test`` is method-agnostic: any base method that overrides
:meth:`BaseMethod.test_build_solver_prompt` automatically benefits. The
contract at test time is

    freeze memory, but still allow per-question embedding retrieval

so we:
  1. call ``method.retrieve_memory`` per entry (per-question, varies)
  2. call ``method.test_build_solver_prompt`` per entry (frozen memory)
  3. send the resulting prompts to :func:`generate_batch` in chunks

DC-RS has an extra optional mode (``run_batched_dc_rs`` with
``use_curator=True``) where we additionally run the curator pass at test
time (original DC-RS behaviour, 2x LLM calls per question).
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.llm_client import generate_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dc_rs_base(method):
    """Return the underlying ``DC_RS`` instance, or ``None`` if the
    method is neither ``DC_RS`` nor a Janus wrapper over ``DC_RS``."""
    base = getattr(method, "base", method)
    if base.__class__.__name__ == "DC_RS":
        return base
    return None


def supports_batched_test(method) -> bool:
    fn = getattr(method, "supports_batched_test", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _predictions_filename(task_name: str, method_name: str, phase_label: str) -> str:
    if phase_label:
        return f"{task_name}_{method_name}_{phase_label}_predictions.jsonl"
    return f"{task_name}_{method_name}_predictions.jsonl"


# ---------------------------------------------------------------------------
# Generic batched test (works for any method implementing
# ``test_build_solver_prompt``): DC-RS / ExpeL / ExpRAG / Janus-* / ...
# ---------------------------------------------------------------------------


def run_batched_test(
    method,
    tasks,
    output_dir: Path,
    batch_size: int = 64,
    run_first_k=None,
    phase_label: str = "test",
    prompt_out_path: Path | None = None,
):
    """Method-agnostic batched test-time inference with frozen memory.

    Requires ``method.test_build_solver_prompt(task, entry, retrieved, inputs)``.
    Retrieval is per-question and per-method; LLM generation is batched
    across a chunk of questions in a single vLLM call.
    """
    if not supports_batched_test(method):
        raise RuntimeError(
            f"{type(method).__name__} does not support batched test. "
            "Override test_build_solver_prompt or pass --test-no-batch."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = method.test_generation_model_name()
    gen_params = method.test_generation_params()

    results = []
    if prompt_out_path is not None:
        prompt_out_path = Path(prompt_out_path)
        prompt_out_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_out_path.write_text("", encoding="utf-8")
    for task in tasks:
        task_name = task.name
        entries = list(task.iter_entries())
        if run_first_k is not None:
            entries = entries[: int(run_first_k)]
        total = len(entries)
        print(
            f"[batched-test] {method.name}/{task_name}: {total} entries, "
            f"batch_size={batch_size}, phase={phase_label}, "
            f"model={model_name}"
        )

        pred_path = output_dir / _predictions_filename(
            task_name, method.name, phase_label
        )
        pred_path.write_text("", encoding="utf-8")

        success = 0
        processed = 0
        for start in range(0, total, batch_size):
            chunk = entries[start : start + batch_size]

            built = []
            for entry in chunk:
                inputs = task.build_inputs(entry)
                try:
                    query = task.get_query(entry, inputs)
                except Exception:
                    query = ""
                try:
                    retrieved = method.retrieve_memory(
                        task, query=query, entry=entry, **inputs
                    )
                except Exception:
                    retrieved = []
                try:
                    solver_prompt = method.test_build_solver_prompt(
                        task, entry, retrieved, inputs
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{type(method).__name__}.test_build_solver_prompt failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                built.append(
                    {
                        "entry": entry,
                        "inputs": inputs,
                        "prompt": solver_prompt,
                    }
                )

            gen_outputs = generate_batch(
                model_name=model_name,
                prompts=[b["prompt"] for b in built],
                temperature=gen_params["temperature"],
                max_new_tokens=gen_params["max_new_tokens"],
            )

            with pred_path.open("a", encoding="utf-8") as f:
                pf = (
                    prompt_out_path.open("a", encoding="utf-8")
                    if prompt_out_path is not None
                    else None
                )
                try:
                    for b, out in zip(built, gen_outputs):
                        if pf is not None:
                            prompt_row = {
                                "task": task_name,
                                "phase": phase_label,
                                "qid": b["inputs"].get("qid", ""),
                                "question": b["inputs"].get("question", ""),
                                "prompt": b["prompt"],
                            }
                            pf.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
                        try:
                            score = task.evaluate_entry(out, b["entry"])
                        except Exception:
                            score = 0.0
                        feedback = "success" if score >= 1.0 else "failure"
                        success += int(score >= 1.0)
                        processed += 1
                        row = {
                            "qid": b["inputs"].get("qid", ""),
                            "question": b["inputs"].get("question", ""),
                            "model_output": str(out),
                            "score": float(score),
                            "feedback": feedback,
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                finally:
                    if pf is not None:
                        pf.close()

            print(
                f"  processed {processed}/{total} "
                f"acc_so_far={success/max(1, processed):.4f}"
            )

        accuracy = success / total if total else 0.0
        results.append(
            {
                "task": task_name,
                "method": method.name,
                "phase": phase_label or None,
                "total": total,
                "success": success,
                "accuracy": accuracy,
            }
        )
    return results


# ---------------------------------------------------------------------------
# DC-RS specific: curator + generator (2x LLM calls per question).
# Kept as a separate function because the curator pass is DC-RS-only and
# the standard frozen-memory path is identical to run_batched_test above.
# ---------------------------------------------------------------------------


def run_batched_dc_rs(
    method,
    tasks,
    output_dir: Path,
    batch_size: int = 64,
    run_first_k=None,
    use_curator: bool = False,
    phase_label: str = "test",
    prompt_out_path: Path | None = None,
):
    """DC-RS / Janus-DC_RS batched test.

    When ``use_curator=False`` this is equivalent to the generic
    :func:`run_batched_test` but kept for backward compatibility.
    When ``use_curator=True`` we additionally run the curator pass per
    question (original DC-RS semantics).
    """
    if not use_curator:
        return run_batched_test(
            method=method,
            tasks=tasks,
            output_dir=output_dir,
            batch_size=batch_size,
            run_first_k=run_first_k,
            phase_label=phase_label,
            prompt_out_path=prompt_out_path,
        )

    dc = dc_rs_base(method)
    assert dc is not None, (
        "Batched DC-RS curator path requires DC-RS or Janus-DC_RS."
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if prompt_out_path is not None:
        prompt_out_path = Path(prompt_out_path)
        prompt_out_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_out_path.write_text("", encoding="utf-8")
    for task in tasks:
        task_name = task.name
        entries = list(task.iter_entries())
        if run_first_k is not None:
            entries = entries[: int(run_first_k)]
        total = len(entries)
        frozen_cs = dc._task_cheatsheets.get(task.name, dc.cheatsheet_default)
        print(
            f"[batched-test/curator] {method.name}/{task_name}: {total} entries, "
            f"batch_size={batch_size}, phase={phase_label}, "
            f"cheatsheet_len={len(str(frozen_cs))}"
        )

        pred_path = output_dir / _predictions_filename(
            task_name, method.name, phase_label
        )
        pred_path.write_text("", encoding="utf-8")

        success = 0
        processed = 0
        for start in range(0, total, batch_size):
            chunk = entries[start : start + batch_size]

            built = []
            for entry in chunk:
                inputs = task.build_inputs(entry)
                prompt = task.build_prompt(entry)
                query = task.get_query(entry, inputs)
                retrieved = dc.retrieve_memory(
                    task, query=query, entry=entry, **inputs
                )
                retrieved_pairs = dc._build_retrieved_pairs_block(retrieved)
                curator_prompt = dc._build_curator_prompt(
                    previous_cheatsheet=frozen_cs,
                    retrieved_pairs=retrieved_pairs,
                    next_input=prompt,
                )
                built.append(
                    {
                        "entry": entry,
                        "inputs": inputs,
                        "prompt": prompt,
                        "retrieved_pairs": retrieved_pairs,
                        "curator_prompt": curator_prompt,
                    }
                )

            curator_outputs = generate_batch(
                model_name=dc.curator_model_name,
                prompts=[b["curator_prompt"] for b in built],
                temperature=dc.temperature,
                max_new_tokens=max(dc.max_new_tokens, 512),
            )

            gen_prompts = []
            for b, cur_out in zip(built, curator_outputs):
                cs = dc._extract_tag_content(str(cur_out or ""), "cheatsheet")
                if not cs:
                    cs = b["retrieved_pairs"]
                gen_prompts.append(dc._build_generator_prompt(b["prompt"], cs))

            gen_outputs = generate_batch(
                model_name=dc.generation_model_name,
                prompts=gen_prompts,
                temperature=dc.temperature,
                max_new_tokens=dc.max_new_tokens,
            )

            with pred_path.open("a", encoding="utf-8") as f:
                pf = (
                    prompt_out_path.open("a", encoding="utf-8")
                    if prompt_out_path is not None
                    else None
                )
                try:
                    for b, out, cur_out, gen_prompt in zip(
                        built, gen_outputs, curator_outputs, gen_prompts
                    ):
                        if pf is not None:
                            prompt_row = {
                                "task": task_name,
                                "phase": phase_label,
                                "qid": b["inputs"].get("qid", ""),
                                "question": b["inputs"].get("question", ""),
                                "curator_prompt": b["curator_prompt"],
                                "curator_output": str(cur_out),
                                "generator_prompt": gen_prompt,
                            }
                            pf.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
                        try:
                            score = task.evaluate_entry(out, b["entry"])
                        except Exception:
                            score = 0.0
                        feedback = "success" if score >= 1.0 else "failure"
                        success += int(score >= 1.0)
                        processed += 1
                        row = {
                            "qid": b["inputs"].get("qid", ""),
                            "question": b["inputs"].get("question", ""),
                            "model_output": str(out),
                            "score": float(score),
                            "feedback": feedback,
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                finally:
                    if pf is not None:
                        pf.close()

            print(
                f"  processed {processed}/{total} "
                f"acc_so_far={success/max(1, processed):.4f}"
            )

        accuracy = success / total if total else 0.0
        results.append(
            {
                "task": task_name,
                "method": method.name,
                "phase": phase_label or None,
                "total": total,
                "success": success,
                "accuracy": accuracy,
            }
        )
    return results
