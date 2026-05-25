#!/usr/bin/env python3
"""Mean ± std of HumanEval test accuracy across stream-order seeds.

Expects layout from ``run_ablation_stream_order.sh``:
  outputs/ablation_stream_order/<subdir>/seed_<N>/test/*_test_stats.json

Example:
    python scripts/summarize_stream_order.py \\
        --ablation-dir outputs/ablation_stream_order
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _accuracy_from_stats(path: Path) -> float | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "accuracy" in data:
        return float(data["accuracy"])
    if "results" in data and isinstance(data["results"], list) and data["results"]:
        r0 = data["results"][0]
        if "accuracy" in r0:
            return float(r0["accuracy"])
    return None


def _collect(ablation_dir: Path) -> dict[str, list[float]]:
    by_method: dict[str, list[float]] = {}
    for sub in sorted(ablation_dir.iterdir()):
        if not sub.is_dir():
            continue
        accs: list[float] = []
        for seed_dir in sorted(sub.glob("seed_*")):
            stats = sorted((seed_dir / "test").glob("*_test_stats.json"))
            if not stats:
                continue
            acc = _accuracy_from_stats(stats[0])
            if acc is not None:
                m = re.search(r"seed_(\d+)$", seed_dir.name)
                accs.append(acc)
        if accs:
            by_method[sub.name] = accs
    return by_method


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    if n == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation-dir",
        type=str,
        default="outputs/ablation_stream_order",
    )
    p.add_argument("--out-json", type=str, default=None)
    args = p.parse_args()

    root = Path(args.ablation_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)

    grouped = _collect(root)
    rows = []
    print(f"{'method':<24} {'n':>3} {'mean':>8} {'std':>8}  per-seed accuracies")
    print("-" * 72)
    for name, accs in grouped.items():
        mu, sd = _mean_std(accs)
        rows.append({"method": name, "n": len(accs), "mean": mu, "std": sd, "accuracies": accs})
        acc_str = ", ".join(f"{a:.4f}" for a in accs)
        print(f"{name:<24} {len(accs):>3} {mu:>8.4f} {sd:>8.4f}  [{acc_str}]")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
