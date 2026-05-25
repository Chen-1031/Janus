"""Subprocess-based Python executor for DC-RS-style code-in-the-loop.

No sandboxing beyond `subprocess.run` + cwd=tempdir + timeout + stdin=DEVNULL.
Used by methods that expect to emit Python and receive execution results
back into the prompt (matches the protocol of the original DC-RS paper).
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py|python3)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

EXEC_SENTINEL = "EXECUTE CODE!"


def has_exec_directive(text: str) -> bool:
    return EXEC_SENTINEL in (text or "")


def extract_last_python_block(text: str) -> str:
    if not text:
        return ""
    matches = _CODE_BLOCK_RE.findall(str(text))
    return matches[-1].strip() if matches else ""


def run_python(code: str, timeout: int = 15) -> Dict[str, Any]:
    if not code.strip():
        return {
            "stdout": "",
            "stderr": "(no python code block found)",
            "timed_out": False,
            "returncode": -1,
        }
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return {
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "timed_out": False,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            # On Python 3.10 `TimeoutExpired.stdout` / `.stderr` can come
            # back as bytes even when text=True; decode defensively.
            def _as_str(buf):
                if buf is None:
                    return ""
                if isinstance(buf, (bytes, bytearray)):
                    return buf.decode("utf-8", errors="replace")
                return buf
            return {
                "stdout": _as_str(exc.stdout),
                "stderr": _as_str(exc.stderr) + f"\n[TIMEOUT after {timeout}s]",
                "timed_out": True,
                "returncode": -1,
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "stdout": "",
                "stderr": f"[executor error: {type(exc).__name__}: {exc}]",
                "timed_out": False,
                "returncode": -1,
            }


def format_exec_result(result: Dict[str, Any], max_chars: int = 4000) -> str:
    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    if len(stdout) > max_chars:
        stdout = stdout[:max_chars] + f"\n... [truncated, {len(stdout) - max_chars} more chars]"
    if len(stderr) > max_chars:
        stderr = stderr[:max_chars] + f"\n... [truncated, {len(stderr) - max_chars} more chars]"
    lines = ["EXECUTION RESULT:", "stdout:"]
    lines.append(stdout.rstrip() if stdout.strip() else "(empty)")
    lines.append("stderr:")
    lines.append(stderr.rstrip() if stderr.strip() else "(empty)")
    if result.get("timed_out"):
        lines.append("(execution timed out)")
    rc = result.get("returncode")
    if rc is not None and rc != 0 and not result.get("timed_out"):
        lines.append(f"(non-zero returncode: {rc})")
    return "\n".join(lines)