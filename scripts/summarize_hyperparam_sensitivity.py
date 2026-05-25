#!/usr/bin/env python3
"""Summarize hyperparameter sensitivity runs: accuracy + eval-cost proxy.

Eval-cost proxy (per user spec):
    total_eval_cost = trigger_times × (K + L)

``trigger_times`` = number of post-warmup steps with ``memory_meta.janus.triggered``
== true in the training ``*_train_memory.jsonl``.

Reads:
  - ``<run_dir>/train/*_train_memory.jsonl`` for triggers
  - ``<run_dir>/test/*_test_stats.json`` for test accuracy

Also scans reference dirs under ``outputs/GPQA_*`` when passed via ``--reference``.

Example:
    python scripts/summarize_hyperparam_sensitivity.py \\
        --ablation-dir outputs/ablation_hyperparam
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_one(parent: Path, pattern: str) -> Path | None:
    hits = sorted(parent.glob(pattern))
    return hits[0] if hits else None


def _trigger_stats(memory_jsonl: Path, k: int, l_fresh: int) -> dict:
    rows = _read_jsonl(memory_jsonl)
    triggered = 0
    controller_steps = 0
    for row in rows:
        meta = (row.get("memory_meta") or {}).get("janus") or {}
        if meta.get("phase") != "controller":
            continue
        controller_steps += 1
        if meta.get("triggered"):
            triggered += 1
    cost = triggered * (k + l_fresh)
    rate = triggered / controller_steps if controller_steps else 0.0
    return {
        "train_rows": len(rows),
        "controller_steps": controller_steps,
        "trigger_times": triggered,
        "trigger_rate": rate,
        "K": k,
        "L": l_fresh,
        "eval_cost_proxy": cost,
    }


def _test_accuracy(run_dir: Path) -> float | None:
    stats = _find_one(run_dir / "test", "*_test_stats.json")
    if stats is None:
        stats = _find_one(run_dir / "test", "*_testonly_stats.json")
    if stats is None:
        return None
    data = json.loads(stats.read_text(encoding="utf-8"))
    for key in ("accuracy", "acc", "test_accuracy"):
        if key in data:
            return float(data[key])
    # Some stats files nest under results
    if "results" in data and isinstance(data["results"], list) and data["results"]:
        r0 = data["results"][0]
        for key in ("accuracy", "acc"):
            if key in r0:
                return float(r0[key])
    return None


def _parse_run_label(name: str) -> dict:
    """Infer K / tau from output folder name."""
    out: dict = {"label": name}
    m = re.search(r"_K(\d+)$", name)
    if m:
        out["sweep"] = "K"
        out["K"] = int(m.group(1))
        out["tau"] = 0.0
        return out
    m = re.search(r"_tau([pm]?\d+p?\d*)$", name)
    if m:
        out["sweep"] = "tau"
        out["K"] = 20
        tag = m.group(1).replace("p", ".")
        tag = tag.replace("m", "-")
        out["tau"] = float(tag)
        return out
    return out


def _summarize_run(run_dir: Path, k_default: int, l_default: int) -> dict | None:
    mem = _find_one(run_dir / "train", "*_train_memory.jsonl")
    if mem is None:
        return None
    label_info = _parse_run_label(run_dir.name)
    k = int(label_info.get("K", k_default))
    l_fresh = l_default
    row = {
        "run_dir": str(run_dir),
        **label_info,
        **_trigger_stats(mem, k=k, l_fresh=l_fresh),
    }
    acc = _test_accuracy(run_dir)
    if acc is not None:
        row["test_accuracy"] = acc
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation-dir",
        type=str,
        default="outputs/ablation_hyperparam",
        help="Root dir containing GPQA_Janus_DC_RS_K* / GPQA_Janus_DC_RS_tau* runs.",
    )
    p.add_argument("--K-default", type=int, default=20)
    p.add_argument("--L-default", type=int, default=5)
    p.add_argument(
        "--reference",
        type=str,
        nargs="*",
        default=[
            "outputs/GPQA_DC_RS",
            "outputs/GPQA_Janus_DC_RS",
        ],
        help="Existing main-result dirs (default K=20 tau=0.0 for Janus).",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write full summary JSON.",
    )
    args = p.parse_args()

    ablation_root = Path(args.ablation_dir)
    rows: list[dict] = []

    if ablation_root.is_dir():
        for child in sorted(ablation_root.iterdir()):
            if not child.is_dir():
                continue
            rec = _summarize_run(child, args.K_default, args.L_default)
            if rec:
                rows.append(rec)

    for ref in args.reference:
        ref_path = Path(ref)
        if not ref_path.is_dir():
            continue
        rec = _summarize_run(ref_path, args.K_default, args.L_default)
        if rec:
            rec["reference"] = True
            if "sweep" not in rec:
                if "Janus" in ref_path.name:
                    rec["sweep"] = "default"
                    rec["K"] = args.K_default
                    rec["tau"] = 0.0
                else:
                    rec["sweep"] = "baseline"
                    rec["method"] = "DC-RS"
            rows.append(rec)

    # Print table
    header = (
        f"{'label':<32} {'sweep':<8} {'K':>4} {'tau':>6} "
        f"{'triggers':>9} {'tr_rate':>8} {'cost':>8} {'test_acc':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        acc = r.get("test_accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        print(
            f"{r.get('label', r['run_dir']):<32} "
            f"{r.get('sweep', ''):<8} "
            f"{r.get('K', ''):>4} "
            f"{r.get('tau', ''):>6.2f} "
            f"{r['trigger_times']:>9} "
            f"{r['trigger_rate']:>8.3f} "
            f"{r['eval_cost_proxy']:>8} "
            f"{acc_s:>9}"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
