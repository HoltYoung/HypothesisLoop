"""Experiment step — LLM-driven codegen + sandboxed execution + retry-on-error.

Usage:
    step = ExperimentStep(
        llm=get_llm(model="moonshot-v1-128k", temperature=0.7),
        session_root="reports/<session_id>",
        dataset_path="data/adult.csv",
        schema_summary=profile_dataset(df, dataset_path="data/adult.csv"),
    )
    experiment = step(hypothesis)

Each call writes to ``<session_root>/iter_<NNN>/attempt_<KK>/`` so retries are
preserved as audit artifacts. Per SPEC §6.6, ``_generate_code`` is the
``evolving.codes`` span and ``_run_attempt`` is the ``evolving.feedbacks``
span — they pair up in the Langfuse UI.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from jinja2 import Template

from hypothesisloop.agent.state import (
    Experiment as ExperimentRecord,
    ExperimentAttempt,
    Hypothesis,
)
from hypothesisloop.sandbox.runner import SandboxResult, run_script
from hypothesisloop.trace.langfuse_client import (
    EVOLVING_CODES,
    EVOLVING_FEEDBACKS,
    observe,
)

# CRITICAL: override=True — the user's shell may carry a stale OPENAI_API_KEY.
load_dotenv(override=True)


_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "experiment.j2"
)
_RETRY_TAIL_CAP = 1500


# ---------------------------------------------------------------------------
# helpers (module-level so tests can hit them directly)
# ---------------------------------------------------------------------------
def extract_python_code(raw: str) -> str:
    """Strip markdown fences / chat decorations from an LLM response.

    Handles, in order:
      1. ``<think>...</think>`` reasoning blocks (Kimi sometimes emits these).
      2. fenced ```python ... ``` or ``` ... ``` blocks.
      3. plain text — falls through with ``.strip()``.
    """
    # Strip <think>...</think> blocks (case-insensitive, multi-line).
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL | re.IGNORECASE)

    fenced = re.search(
        r"```(?:python|py)?\s*\n(.*?)\n```", cleaned, flags=re.DOTALL
    )
    if fenced:
        return fenced.group(1).strip()
    return cleaned.strip()


def _check_ascii_identifiers(code: str) -> Optional[str]:
    """Return a retry-error string if any ``NAME`` token contains non-ASCII chars.

    Uses ``tokenize`` rather than a raw regex so string literals and comments
    (where unicode is fine) don't trip the check. Returns ``None`` on a
    syntax/tokenize error so the runner surfaces the SyntaxError normally
    rather than masking it with a lint failure.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        return None
    for tok in tokens:
        if tok.type == tokenize.NAME and any(ord(c) > 127 for c in tok.string):
            return (
                f"Identifier {tok.string!r} contains non-ASCII characters; "
                f"rewrite using only [A-Za-z_][A-Za-z0-9_]*."
            )
    return None


def format_error_for_retry(result: SandboxResult) -> str:
    """Build the ``last_error_summary`` block for the retry prompt."""
    lines = [f"exit_code: {result.exit_code}"]
    if result.blocked_reason:
        lines.append(f"blocked_reason: {result.blocked_reason}")
    if result.timed_out:
        lines.append("status: TIMED OUT")
    if result.oom_killed:
        lines.append("status: OOM KILLED")

    stderr_tail = (result.stderr or "")[-_RETRY_TAIL_CAP:] or "(empty)"
    stdout_tail = (result.stdout or "")[-_RETRY_TAIL_CAP:] or "(empty)"

    lines.append(f"\nstderr (last {_RETRY_TAIL_CAP} chars):\n{stderr_tail}")
    lines.append(f"\nstdout (last {_RETRY_TAIL_CAP} chars):\n{stdout_tail}")
    return "\n".join(lines)


def _content_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ---------------------------------------------------------------------------
# ExperimentStep
# ---------------------------------------------------------------------------
class ExperimentStep:
    """LLM codegen → sandbox run → retry. Stateless after construction."""

    def __init__(
        self,
        *,
        llm: Any,
        session_root: Path | str,
        dataset_path: Path | str,
        schema_summary: str,
        prompt_path: Optional[Path | str] = None,
        max_retries: int = 3,
        timeout_s: int = 30,
        ram_mb: int = 1024,
        seed: int = 42,
    ):
        self.llm = llm
        self.session_root = Path(session_root)
        # Absolute path so the prepended pd.read_csv works regardless of cwd
        # (the sandbox subprocess runs in attempt_dir, not the project root).
        self.dataset_path = Path(dataset_path).resolve()
        self.schema_summary = schema_summary

        if prompt_path is None:
            prompt_path = _DEFAULT_PROMPT_PATH
        self.prompt_template = Template(Path(prompt_path).read_text(encoding="utf-8"))

        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.ram_mb = ram_mb
        self.seed = seed

    # ---- public entry --------------------------------------------------
    def __call__(self, hypothesis: Hypothesis) -> ExperimentRecord:
        iter_dir = self.session_root / f"iter_{hypothesis.iteration:03d}"
        attempts: list[ExperimentAttempt] = []
        prior_code: Optional[str] = None
        last_error_summary: Optional[str] = None

        # max_retries = retries; total attempts = max_retries + 1.
        for k in range(self.max_retries + 1):
            attempt_dir = iter_dir / f"attempt_{k:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            llm_code = self._generate_code(hypothesis, prior_code, last_error_summary)

            # Pre-sandbox lint: catch non-ASCII identifiers (Kimi sometimes
            # emits CJK characters in variable names) without spending the
            # subprocess + sandbox cost. Counts as one attempt against the
            # retry budget so the next call sees the lint error.
            ascii_error = _check_ascii_identifiers(llm_code)
            if ascii_error is not None:
                sandbox_result = SandboxResult(
                    exit_code=0,
                    stdout="",
                    stderr=ascii_error,
                    figures=[],
                    metrics={},
                    blocked_reason=f"lint: {ascii_error}",
                    duration_s=0.0,
                    timed_out=False,
                    oom_killed=False,
                )
            else:
                full_code = self._wrap_with_data_loader(llm_code)
                sandbox_result = self._run_attempt(full_code, attempt_dir)

            attempt = ExperimentAttempt(
                attempt_idx=k,
                code=llm_code,  # store the raw LLM code, not the wrapped version
                exit_code=sandbox_result.exit_code,
                stdout=sandbox_result.stdout,
                stderr=sandbox_result.stderr,
                figures=[str(p) for p in sandbox_result.figures],
                metrics=sandbox_result.metrics,
                blocked_reason=sandbox_result.blocked_reason,
                duration_s=sandbox_result.duration_s,
                timed_out=sandbox_result.timed_out,
                oom_killed=sandbox_result.oom_killed,
            )
            attempts.append(attempt)

            succeeded = (
                sandbox_result.blocked_reason is None
                and sandbox_result.exit_code == 0
                and not sandbox_result.timed_out
                and not sandbox_result.oom_killed
            )
            if succeeded:
                return ExperimentRecord(
                    hypothesis_id=hypothesis.id,
                    attempts=attempts,
                    succeeded=True,
                )

            # Failure → set up retry context for the next iteration.
            prior_code = llm_code
            last_error_summary = format_error_for_retry(sandbox_result)

        # Exhausted retries.
        return ExperimentRecord(
            hypothesis_id=hypothesis.id,
            attempts=attempts,
            succeeded=False,
        )

    # ---- @observe-tagged inner steps -----------------------------------
    @observe(name=EVOLVING_CODES)
    def _generate_code(
        self,
        hypothesis: Hypothesis,
        prior_code: Optional[str],
        last_error_summary: Optional[str],
    ) -> str:
        rendered = self.prompt_template.render(
            hypothesis=hypothesis.__dict__,
            dataset_path=str(self.dataset_path),
            schema_summary=self.schema_summary,
            prior_code=prior_code,
            last_error_summary=last_error_summary,
        )
        response = self.llm.invoke(rendered)
        raw = _content_of(response)
        return extract_python_code(raw)

    @observe(name=EVOLVING_FEEDBACKS)
    def _run_attempt(self, full_code: str, attempt_dir: Path) -> SandboxResult:
        return run_script(
            full_code,
            attempt_dir,
            timeout_s=self.timeout_s,
            ram_mb=self.ram_mb,
            seed=self.seed,
        )

    # ---- code wrapping --------------------------------------------------
    def _wrap_with_data_loader(self, llm_code: str) -> str:
        """Prepend a deterministic data loader to the LLM's code.

        The LLM's prompt promises ``df`` is already loaded; that's only true
        because we prepend ``pd.read_csv(...)`` here. ``json.dumps`` quotes the
        path safely on every platform (no backslash-escape gotchas on Windows).
        """
        path_literal = json.dumps(str(self.dataset_path))
        loader = (
            "import pandas as pd\n"
            "import numpy as np\n"
            f"df = pd.read_csv({path_literal})\n"
        )
        return loader + "\n# ===== LLM CODE BELOW =====\n" + llm_code


__all__ = [
    "ExperimentStep",
    "extract_python_code",
    "format_error_for_retry",
    "_check_ascii_identifiers",
]
