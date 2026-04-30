"""Sandboxed Python execution for LLM-generated experiment code.

Public API:
    run_script(code, session_dir, *, timeout_s=30, ram_mb=1024, seed=42) -> SandboxResult

Flow:
    1. Render preamble + user code, write to ``session_dir/exp.py`` (always —
       even if the AST check rejects the user code, exp.py is preserved as an
       audit artifact).
    2. ``ast.parse`` the user code (preamble is trusted, not scanned). Reject
       on syntax errors.
    3. Walk the AST: deny imports outside ``ALLOWED_IMPORTS``, deny calls to
       names in ``DENIED_BUILTIN_CALLS``, deny attribute access on names in
       ``DENIED_ATTRIBUTE_PREFIXES``, deny bare references to ``DENIED_NAMES``.
    4. On POSIX, run via a tiny launcher that sets ``RLIMIT_AS`` then
       ``runpy.run_path("exp.py")``. On Windows, run exp.py directly and emit
       a one-time RuntimeWarning that the RAM cap is unavailable.
    5. Subprocess runs with a *scrubbed* environment — no API keys, no
       LANGFUSE_*, no PYTHONPATH leakage.
    6. Capture stdout (last 4096 chars), stderr (last 50 lines, capped at
       4096 chars), PNGs in session_dir, and ``metrics.json`` if present.

Stdlib-only — no new dependencies.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hypothesisloop.sandbox.allowlist import (
    ALLOWED_IMPORTS,
    DENIED_ATTRIBUTE_PREFIXES,
    DENIED_BUILTIN_CALLS,
    DENIED_NAMES,
)
from hypothesisloop.sandbox.preamble import USER_CODE_MARKER, render_preamble


_STDOUT_CAP = 4096
_STDERR_CAP = 4096
_STDERR_LINE_CAP = 50

_RAM_CAP_WARNED = False


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    figures: list[Path] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    blocked_reason: Optional[str] = None
    duration_s: float = 0.0
    timed_out: bool = False
    oom_killed: bool = False


# ---------------------------------------------------------------------------
# AST scan
# ---------------------------------------------------------------------------
def _ast_check(code: str) -> Optional[str]:
    """Return ``None`` if code passes the denylist; else a short reason string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    return f"denied: import of {alias.name!r} (not in allowlist)"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                return "denied: relative import"
            top = node.module.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                return f"denied: from {node.module!r} import (not in allowlist)"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DENIED_BUILTIN_CALLS:
                return f"denied: call to builtin {func.id!r}"
        elif isinstance(node, ast.Attribute):
            if node.attr in DENIED_ATTRIBUTE_PREFIXES:
                return f"denied: attribute access {node.attr!r}"
        elif isinstance(node, ast.Name):
            if node.id in DENIED_NAMES:
                return f"denied: reference to {node.id!r}"
    return None


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------
def _truncate_stdout(text: str) -> str:
    if len(text) <= _STDOUT_CAP:
        return text
    return text[-_STDOUT_CAP:]


def _truncate_stderr(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines(keepends=True)
    tail = "".join(lines[-_STDERR_LINE_CAP:])
    if len(tail) > _STDERR_CAP:
        tail = tail[-_STDERR_CAP:]
    return tail


# ---------------------------------------------------------------------------
# Env scrubbing
# ---------------------------------------------------------------------------
def _build_env() -> dict:
    """Subprocess environment with secrets stripped.

    The sandbox must not be able to phone home with our API keys, even if
    the LLM tries to read os.environ. We only forward variables Python itself
    needs to start (PATH, plus a small Windows-specific allowlist).
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if sys.platform == "win32":
        # Without these, CPython on Windows may fail to initialize.
        for k in ("SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE", "COMSPEC"):
            if k in os.environ:
                env[k] = os.environ[k]
    # Hard guarantees: no secrets, no proxy/tracing leakage.
    forbidden_substrings = ("API_KEY", "SECRET", "TOKEN", "LANGFUSE", "MOONSHOT", "OPENAI")
    for k in list(env):
        if any(s in k.upper() for s in forbidden_substrings):
            del env[k]
    return env


# ---------------------------------------------------------------------------
# POSIX launcher
# ---------------------------------------------------------------------------
def _write_posix_launcher(session_dir: Path, ram_mb: int) -> Path:
    launcher = session_dir / "_launcher.py"
    bytes_cap = int(ram_mb) * 1024 * 1024
    launcher.write_text(
        "import resource, runpy\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({bytes_cap}, {bytes_cap}))\n"
        "runpy.run_path('exp.py', run_name='__main__')\n",
        encoding="utf-8",
    )
    return launcher


def _warn_ram_cap_once() -> None:
    global _RAM_CAP_WARNED
    if not _RAM_CAP_WARNED:
        _RAM_CAP_WARNED = True
        warnings.warn("RAM cap unavailable on Windows", RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_script(
    code: str,
    session_dir: Path,
    *,
    timeout_s: int = 30,
    ram_mb: int = 1024,
    seed: int = 42,
) -> SandboxResult:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Always write exp.py first — audit trail even when AST blocks the run.
    full_script = render_preamble(seed) + "\n" + code + "\n"
    exp_path = session_dir / "exp.py"
    exp_path.write_text(full_script, encoding="utf-8")

    blocked = _ast_check(code)
    if blocked is not None:
        return SandboxResult(exit_code=0, blocked_reason=blocked)

    # Build subprocess command (POSIX adds the setrlimit launcher).
    if sys.platform == "win32":
        _warn_ram_cap_once()
        cmd = [sys.executable, str(exp_path.name)]
    else:
        launcher = _write_posix_launcher(session_dir, ram_mb)
        cmd = [sys.executable, str(launcher.name)]

    env = _build_env()
    start = time.monotonic()
    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = 0
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(session_dir),
            timeout=timeout_s,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = -1
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")

    duration_s = time.monotonic() - start

    # OOM detection (POSIX only — RLIMIT_AS triggers SIGKILL or MemoryError).
    oom_killed = False
    if sys.platform != "win32" and not timed_out and exit_code != 0:
        if "MemoryError" in stderr or exit_code == -9 or "Killed" in stderr:
            oom_killed = True

    # Capture artifacts.
    figures = sorted(p.resolve() for p in session_dir.glob("*.png"))
    metrics: dict = {}
    metrics_path = session_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}

    return SandboxResult(
        exit_code=exit_code,
        stdout=_truncate_stdout(stdout),
        stderr=_truncate_stderr(stderr),
        figures=figures,
        metrics=metrics,
        blocked_reason=None,
        duration_s=duration_s,
        timed_out=timed_out,
        oom_killed=oom_killed,
    )


__all__ = ["SandboxResult", "run_script", "USER_CODE_MARKER"]
