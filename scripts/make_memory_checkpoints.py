"""Truncate a training-phase *_memory.jsonl into intermediate checkpoints.

Used by ablation 3 (Old-vs-New Deployment Decision): take the memory file
produced during training and write the first 20% / 40% / 60% / 80% of
rows to separate files. Each truncated file represents the memory state
the method would have reached if training had stopped at that point.

Example:
    python scripts/make_memory_checkpoints.py \\
        --memory-jsonl outputs/GPQA_Janus_DC_RS/train/GPQA_Janus-DC_RS_train_memory.jsonl \\
        --out-dir outputs/ablation3/GPQA_Janus_DC_RS_checkpoints \\
        --fractions 20,40,60,80

Writes ``<out-dir>/checkpoint_<pct>.jsonl`` for each percentage and prints
the resulting row counts so the bash caller can sanity-check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line.rstrip("\n"))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-jsonl", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--fractions",
        type=str,
        default="20,40,60,80",
        help="Comma-separated percentages of rows to keep, e.g. '20,40,60,80'.",
    )
    args = p.parse_args()

    src = Path(args.memory_jsonl).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Memory file not found: {src}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(src)
    n = len(rows)
    if n == 0:
        raise ValueError(f"Memory file is empty: {src}")

    fractions = [int(x.strip()) for x in args.fractions.split(",") if x.strip()]

    print(f"[ckpt] source rows: {n}  src: {src}")
    summary = {"source_rows": n, "checkpoints": {}}
    for pct in fractions:
        keep = max(1, round(n * pct / 100.0))
        keep = min(keep, n)
        out_path = out_dir / f"checkpoint_{pct:02d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows[:keep]:
                f.write(r + "\n")
        summary["checkpoints"][f"{pct}%"] = {"rows": keep, "path": str(out_path)}
        print(f"[ckpt] {pct:>3}% -> rows={keep:>5}  {out_path}")

    summary_path = out_dir / "checkpoints_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ckpt] summary written -> {summary_path}")


if __name__ == "__main__":
    main()
