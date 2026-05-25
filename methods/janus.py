"""Janus: a plugin-like memory-deployment controller.

Janus wraps a base updater ``P`` (e.g., ``DC_RS`` or ``ExpeL``) as composition
(not inheritance) and intercepts only the ``update_memory`` stage:

    retrieve_memory -> delegate to base
    generate        -> delegate to base
    update_memory   -> run base.update_memory, read {old, new} via
                       base.get_step_memories, run MMT trigger, on trigger
                       compare old vs new on a compact support+fresh set,
                       commit chosen memory via the adapter.

Adapters provide a small base-method-specific interface (memory snapshot,
restore, and fixed-memory solve) so Janus stays generic.
"""

from __future__ import annotations

import copy
import random

import numpy as np

from methods.base_method import BaseMethod
from methods.janus_core import (
    JanusState,
    cosine,
    momentum_update,
    run_kmeans,
    should_trigger_mmt,
    to_vec,
)
from utils.llm_client import generate as llm_generate
from utils.llm_client import generate_batch as llm_generate_batch


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class BaseJanusAdapter:
    """Adapter interface for making a base updater Janus-compatible."""

    def memory_text(self, base, task) -> str:
        raise NotImplementedError

    def snapshot(self, base, task) -> dict:
        raise NotImplementedError

    def restore(self, base, task, snap: dict) -> None:
        raise NotImplementedError

    def patch_last_record(self, base, row: dict, chosen: str, deployed_text: str) -> None:
        # Optional: rewrite the deployed memory text inside the exported record
        # so downstream inspection reflects what Janus actually deployed.
        return None

    def solve_with_memory(self, base, task, entry, memory_text: str) -> str:
        """Solve ``entry`` with a fixed ``memory_text`` (no memory update)."""
        raise NotImplementedError

    def build_solver_prompt(self, base, task, entry, memory_text: str) -> str:
        """Build the generator prompt used by :meth:`solve_with_memory`.

        Subclasses should override so Janus can batch multiple prompts into
        a single vLLM call instead of looping ``solve_with_memory`` one
        entry at a time.
        """
        raise NotImplementedError

    def solver_model_name(self, base) -> str:
        """Name of the model used for :meth:`solve_with_memory`."""
        return getattr(base, "generation_model_name", "")


class DCRSJanusAdapter(BaseJanusAdapter):
    def memory_text(self, base, task) -> str:
        return str(
            base._task_cheatsheets.get(task.name, base.cheatsheet_default)
        )

    def snapshot(self, base, task) -> dict:
        return {
            "cheatsheet": base._task_cheatsheets.get(task.name, base.cheatsheet_default),
        }

    def restore(self, base, task, snap: dict) -> None:
        base._task_cheatsheets[task.name] = snap.get("cheatsheet", base.cheatsheet_default)

    def patch_last_record(self, base, row: dict, chosen: str, deployed_text: str) -> None:
        # DC-RS stores the deployed cheatsheet under row["memory"] as a string.
        if isinstance(row.get("memory"), str):
            row["memory"] = deployed_text

    def solve_with_memory(self, base, task, entry, memory_text: str) -> str:
        gen_prompt = self.build_solver_prompt(base, task, entry, memory_text)
        return llm_generate(
            model_name=base.generation_model_name,
            prompt=gen_prompt,
            temperature=base.temperature,
            max_new_tokens=base.max_new_tokens,
        )

    def build_solver_prompt(self, base, task, entry, memory_text: str) -> str:
        prompt = task.build_prompt(entry)
        return base._build_generator_prompt(prompt, memory_text)


class ExpeLJanusAdapter(BaseJanusAdapter):
    def memory_text(self, base, task) -> str:
        return str(base._current_insights(task.name))

    def snapshot(self, base, task) -> dict:
        tname = task.name
        return {
            "rules": copy.deepcopy(base._task_rule_items_with_count.get(tname, [])),
            "recent": copy.deepcopy(base._task_recent_success.get(tname, [])),
        }

    def restore(self, base, task, snap: dict) -> None:
        tname = task.name
        base._task_rule_items_with_count[tname] = list(snap.get("rules", []))
        base._task_recent_success[tname] = list(snap.get("recent", []))

    def patch_last_record(self, base, row: dict, chosen: str, deployed_text: str) -> None:
        mem = row.get("memory")
        if isinstance(mem, dict):
            mem["insights"] = deployed_text

    def solve_with_memory(self, base, task, entry, memory_text: str) -> str:
        gen_prompt = self.build_solver_prompt(base, task, entry, memory_text)
        return llm_generate(
            model_name=base.generation_model_name,
            prompt=gen_prompt,
            temperature=base.temperature,
            max_new_tokens=base.max_new_tokens,
        )

    def build_solver_prompt(self, base, task, entry, memory_text: str) -> str:
        prompt = task.build_prompt(entry)
        return base._build_solver_prompt(
            prompt=prompt,
            insights=memory_text,
            retrieved_cases=[],
            reflection_note="",
            try_idx=1,
        )


def _select_adapter(base_method) -> BaseJanusAdapter:
    cls_name = base_method.__class__.__name__
    if cls_name == "DC_RS":
        return DCRSJanusAdapter()
    if cls_name == "ExpeL":
        return ExpeLJanusAdapter()
    raise ValueError(
        f"Janus has no adapter for base method class: {cls_name}. "
        "Add one in methods/janus.py."
    )


# ---------------------------------------------------------------------------
# JanusMethod wrapper
# ---------------------------------------------------------------------------


class JanusMethod(BaseMethod):
    """Generic Janus controller over a base memory updater."""

    name = "Janus"

    def __init__(
        self,
        base_method,
        k: int = 20,
        k_prime: int = 12,
        l_fresh: int = 5,
        beta: float = 0.9,
        tau: float = 0.0,
        seed: int = 0,
        replay_limit=None,
        **_ignored,
    ):
        super().__init__()
        if base_method is None:
            raise ValueError("JanusMethod requires a base_method instance.")
        self.base = base_method
        self.adapter = _select_adapter(base_method)
        self.k = max(1, int(k))
        self.k_prime = max(1, min(int(k_prime), self.k))
        self.l_fresh = max(0, int(l_fresh))
        self.beta = float(beta)
        self.tau = float(tau)
        self.seed = int(seed)
        self.replay_limit = int(replay_limit) if replay_limit else None

        base_name = getattr(base_method, "name", base_method.__class__.__name__)
        self.name = f"Janus-{base_name}"

        self._task_state: dict = {}

    # ------------------------------------------------------------------ utils
    def _state(self, task) -> JanusState:
        return self._task_state.setdefault(task.name, JanusState())

    def _embed(self, texts):
        """Use the base method's embedder to keep spaces aligned."""
        if hasattr(self.base, "_encode"):
            vectors = self.base._encode(list(texts))
            return [to_vec(v) for v in vectors]
        raise RuntimeError(
            "Base method has no _encode(); Janus requires the base to expose "
            "a sentence-transformers-backed embedder."
        )

    # ----------------------------------------------- delegated BaseMethod API
    def reset_task_state(self, task):
        self._task_state.setdefault(task.name, JanusState())
        return self.base.reset_task_state(task)

    def get_current_memory(self, task):
        # After ``update_memory``, the base method's internal state already
        # reflects whichever memory Janus chose to deploy (new candidate or
        # restored old snapshot). The adapter knows how to materialize that
        # state as a string, which is exactly the full version of what is
        # otherwise truncated into ``janus_deployed_memory_preview``.
        try:
            return self.adapter.memory_text(self.base, task)
        except Exception:
            getter = getattr(self.base, "get_current_memory", None)
            if callable(getter):
                return getter(task)
            return None

    # Batched-test delegation: Janus is a memory-deployment controller; at
    # test time we don't mutate memory, so the solver prompt is entirely
    # determined by the base method's frozen state.
    def test_build_solver_prompt(self, task, entry, retrieved, inputs):
        return self.base.test_build_solver_prompt(task, entry, retrieved, inputs)

    def test_generation_model_name(self) -> str:
        return self.base.test_generation_model_name()

    def test_generation_params(self) -> dict:
        return self.base.test_generation_params()

    def supports_batched_test(self) -> bool:
        return self.base.supports_batched_test()

    def finalize_task_state(self, task):
        return self.base.finalize_task_state(task)

    def retrieve_memory(self, task, **kwargs):
        return self.base.retrieve_memory(task, **kwargs)

    def generate(self, task, prompt, **kwargs):
        return self.base.generate(task, prompt, **kwargs)

    def get_timing_adjustments(self, task):
        return self.base.get_timing_adjustments(task)

    def export_task_memory(self, task_name, output_path):
        return self.base.export_task_memory(task_name, output_path)

    def build_task_memory_snapshot(self, task_name):
        return self.base.build_task_memory_snapshot(task_name)

    def get_latest_task_record(self, task_name):
        return self.base.get_latest_task_record(task_name)

    def run_trial(self, task, entry, inputs, timer, update_memory=True):
        # Use the default runner path so Janus can intercept update_memory.
        # If the base method has an interactive run_trial we intentionally skip
        # it here to keep Janus semantics consistent across baselines (MVP).
        return None

    def _decide_trigger(self, z_t, prev_m, state) -> tuple:
        """Override in subclasses to swap out the MMT trigger rule."""
        return should_trigger_mmt(z_t, prev_m, self.tau)

    # -------------------------------------------------------- update override
    def update_memory(self, task, output, **kwargs):
        state = self._state(task)
        record = kwargs.get("record") or {}
        entry = kwargs.get("entry")

        # 1) Snapshot current base state; let base produce candidate new memory.
        snap = self.adapter.snapshot(self.base, task)
        self.base.update_memory(task, output, **kwargs)

        step_mem = self.base.get_step_memories(task) or {}
        old_memory_text = str(step_mem.get("old_memory") or "")
        new_memory_text = str(step_mem.get("new_memory") or "")

        # 2) Build current task payload for pending/seen stores.
        question = str(record.get("question", "") or "")
        gt = str(
            record.get("gold_final_number", "")
            or record.get("answer", "")
            or record.get("gold_final_answer", "")
            or ""
        )
        score = float(record.get("score", 0.0) or 0.0)
        feedback = record.get("feedback", "failure")

        # Batch embed the question + both memory texts in a single encoder
        # forward to cut ~3x encoder round-trips per step down to 1.
        texts_to_embed = []
        idx = {}
        if question:
            idx["q"] = len(texts_to_embed)
            texts_to_embed.append(question)
        if old_memory_text:
            idx["old"] = len(texts_to_embed)
            texts_to_embed.append(old_memory_text)
        if new_memory_text:
            idx["new"] = len(texts_to_embed)
            texts_to_embed.append(new_memory_text)

        try:
            vecs = self._embed(texts_to_embed) if texts_to_embed else []
        except Exception:
            vecs = []

        def _pick(key, default_shape):
            j = idx.get(key)
            if j is None or j >= len(vecs):
                return np.zeros(default_shape, dtype=float)
            v = np.asarray(vecs[j], dtype=float).reshape(-1)
            return v

        q_emb = _pick("q", 1)
        old_vec_cached = _pick("old", 1)
        new_vec_cached = _pick("new", 1)

        task_item = {
            "entry": entry,
            "question": question,
            "output": str(output),
            "gt": gt,
            "score": score,
            "feedback": feedback,
            "embedding": q_emb,
        }

        meta = {
            "enabled": True,
            "phase": "warmup",
            "triggered": False,
            "chosen_memory": "new",
            "reason": "warmup",
            "cosine_z_m_prev": None,
            "z_norm": None,
            "momentum_norm": None,
            "support_old_acc": None,
            "support_new_acc": None,
            "num_eval_tasks": 0,
            "num_flip_tasks": 0,
            "coverage_size": len(state.coverage_set),
            "boundary_size": len(state.boundary_set),
            "pending_size": len(state.pending_tasks),
            "seen_size": len(state.seen_tasks),
            "step_count": state.step_count,
        }

        state.step_count += 1

        # 3) Warm-up phase: accept new, store embeddings.
        if state.step_count <= self.k:
            state.pending_tasks.append(task_item)
            meta["pending_size"] = len(state.pending_tasks)
            if state.step_count == self.k and not state.warmup_initialized:
                self._initialize_support_sets(state)
                state.seen_tasks.extend(state.pending_tasks)
                state.pending_tasks = []
                state.warmup_initialized = True
                meta["reason"] = "warmup_end_init"
                meta["coverage_size"] = len(state.coverage_set)
                meta["boundary_size"] = len(state.boundary_set)
                meta["pending_size"] = 0
                meta["seen_size"] = len(state.seen_tasks)
            self._annotate_last_record(task, meta, old_memory_text, new_memory_text, chosen="new")
            return

        meta["phase"] = "controller"

        # 4) MMT check.
        # Re-use the vectors computed in the single batched embed above.
        old_vec = old_vec_cached
        new_vec = new_vec_cached

        if old_vec.shape != new_vec.shape:
            z_t = np.zeros_like(new_vec)
        else:
            z_t = new_vec - old_vec

        prev_m = state.momentum
        triggered, cos_val = self._decide_trigger(z_t, prev_m, state)

        meta["cosine_z_m_prev"] = float(cos_val) if prev_m is not None else None
        meta["z_norm"] = float(np.linalg.norm(z_t))
        meta["momentum_norm"] = float(np.linalg.norm(prev_m)) if prev_m is not None else None
        meta["triggered"] = bool(triggered)

        chosen = "new"

        if not triggered:
            state.pending_tasks.append(task_item)
            meta["reason"] = "no_trigger"
            meta["pending_size"] = len(state.pending_tasks)
        else:
            # Merge pending + current into seen, then clear pending.
            state.pending_tasks.append(task_item)
            fresh_set = list(state.pending_tasks)
            if self.l_fresh > 0 and len(fresh_set) > self.l_fresh:
                rng = random.Random(self.seed + state.trigger_count)
                fresh_set = rng.sample(fresh_set, self.l_fresh)
            state.seen_tasks.extend(state.pending_tasks)
            state.pending_tasks = []
            state.trigger_count += 1

            # Refresh coverage set on seen_tasks with warm-start centroids.
            try:
                self._refresh_coverage_set(state)
            except Exception:
                # Keep previous coverage if clustering fails.
                pass

            eval_entries = []
            for cov in state.coverage_set:
                eval_entries.append(cov["rep_task"])
            for bnd in state.boundary_set:
                eval_entries.append(bnd["task"])
            eval_entries.extend(fresh_set)

            if self.replay_limit is not None and len(eval_entries) > self.replay_limit:
                rng = random.Random(self.seed + state.trigger_count + 17)
                eval_entries = rng.sample(eval_entries, self.replay_limit)

            try:
                old_correct, new_correct = self._replay_eval(
                    task, eval_entries, old_memory_text, new_memory_text
                )
            except Exception as exc:
                meta["reason"] = f"replay_failed:{type(exc).__name__}"
                meta["chosen_memory"] = "new"
                # Momentum still updates even if replay failed.
                state.momentum = momentum_update(prev_m, z_t, self.beta)
                self._annotate_last_record(
                    task, meta, old_memory_text, new_memory_text, chosen="new"
                )
                return

            total = max(1, len(eval_entries))
            old_acc = sum(old_correct) / total
            new_acc = sum(new_correct) / total
            meta["support_old_acc"] = float(old_acc)
            meta["support_new_acc"] = float(new_acc)
            meta["num_eval_tasks"] = total

            if new_acc >= old_acc:
                chosen = "new"
                meta["chosen_memory"] = "new"
                meta["reason"] = (
                    "trigger_new_better" if new_acc > old_acc else "trigger_tie_new"
                )
            else:
                self.adapter.restore(self.base, task, snap)
                chosen = "old"
                meta["chosen_memory"] = "old"
                meta["reason"] = "trigger_old_better"

            flip_tasks = [
                ent
                for ent, co, cn in zip(eval_entries, old_correct, new_correct)
                if co != cn
            ]
            cov_ids = {id(cov["rep_task"]) for cov in state.coverage_set}
            flip_tasks = [t for t in flip_tasks if id(t) not in cov_ids]
            meta["num_flip_tasks"] = len(flip_tasks)
            try:
                self._refresh_boundary_set(state, flip_tasks)
            except Exception:
                pass

        # 5) Momentum update (always, after warm-up).
        state.momentum = momentum_update(prev_m, z_t, self.beta)

        meta["coverage_size"] = len(state.coverage_set)
        meta["boundary_size"] = len(state.boundary_set)
        meta["pending_size"] = len(state.pending_tasks)
        meta["seen_size"] = len(state.seen_tasks)

        self._annotate_last_record(
            task, meta, old_memory_text, new_memory_text, chosen=chosen
        )

    # ------------------------------------------------ warm-up / support sets
    def _initialize_support_sets(self, state: JanusState) -> None:
        if not state.pending_tasks:
            return
        X = np.stack(
            [np.asarray(e["embedding"], dtype=float).reshape(-1) for e in state.pending_tasks]
        )
        if X.ndim != 2 or X.shape[0] == 0:
            return
        k = min(self.k_prime, X.shape[0])
        centers, labels = run_kmeans(X, k, seed=self.seed)
        state.centroids = centers

        coverage = []
        used_idx = set()
        for ci in range(k):
            mask = labels == ci
            if not np.any(mask):
                continue
            idxs = np.where(mask)[0]
            dists = np.linalg.norm(X[idxs] - centers[ci], axis=1)
            rep_idx = int(idxs[int(np.argmin(dists))])
            rep_item = state.pending_tasks[rep_idx]
            coverage.append(
                {
                    "centroid": centers[ci],
                    "rep_task": rep_item,
                    "embedding": rep_item["embedding"],
                }
            )
            used_idx.add(rep_idx)

        state.coverage_set = coverage

        remaining = [
            t for i, t in enumerate(state.pending_tasks) if i not in used_idx
        ]
        k_b = max(0, self.k - self.k_prime)
        state.boundary_set = [
            {"task": t, "embedding": t["embedding"], "tag": "bootstrap"}
            for t in remaining[:k_b]
        ]

    def _refresh_coverage_set(self, state: JanusState) -> None:
        if not state.seen_tasks:
            return
        X = np.stack(
            [np.asarray(e["embedding"], dtype=float).reshape(-1) for e in state.seen_tasks]
        )
        if X.ndim != 2 or X.shape[0] == 0:
            return
        k = min(self.k_prime, X.shape[0])

        init = None
        if (
            state.centroids is not None
            and getattr(state.centroids, "shape", (0, 0))[0] == k
            and state.centroids.shape[-1] == X.shape[1]
        ):
            init = state.centroids

        centers, labels = run_kmeans(X, k, seed=self.seed, init_centers=init)
        state.centroids = centers

        coverage = []
        for ci in range(k):
            mask = labels == ci
            if not np.any(mask):
                continue
            idxs = np.where(mask)[0]
            dists = np.linalg.norm(X[idxs] - centers[ci], axis=1)
            rep_idx = int(idxs[int(np.argmin(dists))])
            rep_item = state.seen_tasks[rep_idx]
            coverage.append(
                {
                    "centroid": centers[ci],
                    "rep_task": rep_item,
                    "embedding": rep_item["embedding"],
                }
            )
        state.coverage_set = coverage

    def _refresh_boundary_set(self, state: JanusState, flip_tasks: list) -> None:
        k_b = max(0, self.k - self.k_prime)
        if k_b == 0:
            state.boundary_set = []
            return
        if len(flip_tasks) >= k_b:
            rng = random.Random(self.seed + state.trigger_count + 101)
            sampled = rng.sample(flip_tasks, k_b)
            state.boundary_set = [
                {"task": t, "embedding": t["embedding"], "tag": "flip"}
                for t in sampled
            ]
            return

        cov_ids = {id(cov["rep_task"]) for cov in state.coverage_set}
        flip_ids = {id(t) for t in flip_tasks}
        prev = [
            e
            for e in state.boundary_set
            if id(e["task"]) not in flip_ids and id(e["task"]) not in cov_ids
        ]
        new_entries = [
            {"task": t, "embedding": t["embedding"], "tag": "flip"} for t in flip_tasks
        ]
        slots_left = k_b - len(new_entries)
        state.boundary_set = new_entries + prev[: max(0, slots_left)]

    # ---------------------------------------------------------- replay eval
    def _replay_eval(self, task, eval_entries, old_memory_text, new_memory_text):
        """Score the eval set under ``old_memory`` and ``new_memory``.

        All prompts (2 x len(valid_entries)) are submitted to the model in
        a single batched call so the backend (vLLM) can use continuous
        batching. This is the hot path on every MMT trigger and was
        previously the main reason Janus was slow.
        """
        n = len(eval_entries)
        old_correct = [0] * n
        new_correct = [0] * n

        # Stage 1: build prompts (skip invalid items).
        valid_idx = []
        old_prompts = []
        new_prompts = []
        entry_payloads = []
        for i, item in enumerate(eval_entries):
            entry_payload = item.get("entry") if isinstance(item, dict) else None
            if entry_payload is None:
                continue
            try:
                p_old = self.adapter.build_solver_prompt(
                    self.base, task, entry_payload, old_memory_text
                )
                p_new = self.adapter.build_solver_prompt(
                    self.base, task, entry_payload, new_memory_text
                )
            except Exception:
                # Prompt construction itself failed (rare); treat as 0/0.
                continue
            valid_idx.append(i)
            old_prompts.append(p_old)
            new_prompts.append(p_new)
            entry_payloads.append(entry_payload)

        if not valid_idx:
            return old_correct, new_correct

        # Stage 2: one batched LLM call for all old+new prompts.
        model_name = self.adapter.solver_model_name(self.base)
        temperature = getattr(self.base, "temperature", 0.7)
        max_new_tokens = getattr(self.base, "max_new_tokens", 2048)

        prompts = old_prompts + new_prompts
        try:
            outputs = llm_generate_batch(
                model_name=model_name,
                prompts=prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception:
            # Fall back to per-entry sequential solve if batch fails.
            outputs = []
            for p in prompts:
                try:
                    outputs.append(
                        llm_generate(
                            model_name=model_name,
                            prompt=p,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                except Exception:
                    outputs.append("")

        m = len(valid_idx)
        old_outs = outputs[:m]
        new_outs = outputs[m:]

        # Stage 3: evaluate.
        for local_i, global_i in enumerate(valid_idx):
            entry_payload = entry_payloads[local_i]
            try:
                score_old = float(task.evaluate_entry(old_outs[local_i], entry_payload))
            except Exception:
                score_old = 0.0
            try:
                score_new = float(task.evaluate_entry(new_outs[local_i], entry_payload))
            except Exception:
                score_new = 0.0
            old_correct[global_i] = 1 if score_old >= 1.0 else 0
            new_correct[global_i] = 1 if score_new >= 1.0 else 0

        return old_correct, new_correct

    # -------------------------------------------------------- record tagging
    def _annotate_last_record(
        self, task, meta: dict, old_memory_text: str, new_memory_text: str, chosen: str
    ) -> None:
        if not getattr(self.base, "memory", None):
            return
        row = self.base.memory[-1]
        if not isinstance(row, dict):
            return
        mm = row.setdefault("memory_meta", {})
        if not isinstance(mm, dict):
            mm = {}
            row["memory_meta"] = mm
        mm["janus"] = meta
        row["janus_old_memory_preview"] = (old_memory_text or "")[:500]
        row["janus_new_memory_preview"] = (new_memory_text or "")[:500]
        deployed_text = old_memory_text if chosen == "old" else new_memory_text
        row["janus_deployed_memory_preview"] = (deployed_text or "")[:500]
        try:
            self.adapter.patch_last_record(self.base, row, chosen, deployed_text)
        except Exception:
            pass
