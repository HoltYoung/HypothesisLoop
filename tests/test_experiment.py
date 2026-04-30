"""Phase 4 experiment-step unit tests (mocked LLM, real sandbox)."""

from __future__ import annotations

from pathlib import Path

from hypothesisloop.agent.state import Hypothesis, new_hypothesis_id
from hypothesisloop.sandbox.runner import SandboxResult
from hypothesisloop.steps.experiment import (
    ExperimentStep,
    _check_ascii_identifiers,
    extract_python_code,
    format_error_for_retry,
)


# ---------------------------------------------------------------------------
# stubs / helpers
# ---------------------------------------------------------------------------
class _StubLLM:
    """Returns canned responses in order; useful for testing retry sequences."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def invoke(self, _prompt):
        if not self._responses:
            raise AssertionError("StubLLM out of responses")

        class _Resp:
            content = self._responses.pop(0)

        return _Resp()


def _make_hypothesis(iteration: int = 1) -> Hypothesis:
    return Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=iteration,
        statement="age has a non-trivial mean",
        null="age has zero mean",
        test_type="custom",
        target_columns=["age"],
        expected_outcome="mean > 0",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
    )


_DATASET = Path("data/adult.csv").resolve()


def _make_step(*, llm, session_root: Path, max_retries: int = 3) -> ExperimentStep:
    return ExperimentStep(
        llm=llm,
        session_root=session_root,
        dataset_path=_DATASET,
        schema_summary="age: numeric column; income: categorical (>50K, <=50K).",
        max_retries=max_retries,
        timeout_s=30,
        ram_mb=1024,
        seed=42,
    )


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------
def test_extract_python_code_strips_fences():
    bare = "import pandas as pd\nprint(df)"
    assert extract_python_code(bare) == bare
    assert extract_python_code("  " + bare + "  \n") == bare

    fenced_lang = f"```python\n{bare}\n```"
    assert extract_python_code(fenced_lang) == bare

    fenced_plain = f"```\n{bare}\n```"
    assert extract_python_code(fenced_plain) == bare

    with_think = (
        "<think>let me reason about this</think>\n"
        f"```python\n{bare}\n```\n"
    )
    assert extract_python_code(with_think) == bare


def test_format_error_for_retry_includes_both_streams():
    long_text = "x" * 5000  # > 1500 cap
    fake = SandboxResult(
        exit_code=1,
        stdout="partial output before crash",
        stderr=long_text,
        figures=[],
        metrics={},
        blocked_reason=None,
        duration_s=0.5,
        timed_out=False,
        oom_killed=False,
    )
    out = format_error_for_retry(fake)
    assert "exit_code: 1" in out
    assert "stderr (last 1500 chars)" in out
    assert "stdout (last 1500 chars)" in out
    assert "partial output before crash" in out
    # stderr was 5000 chars; the slice that lands in `out` is at most 1500.
    stderr_section = out.split("stderr (last 1500 chars):\n", 1)[1].split(
        "\n\nstdout", 1
    )[0]
    assert len(stderr_section) <= 1500


def test_experiment_success_first_attempt(tmp_path: Path):
    code = (
        "mean_age = df['age'].mean()\n"
        "print(f'mean_age={mean_age:.4f}')\n"
        "hl_emit('mean_age', float(mean_age))\n"
        "hl_emit('n', int(len(df)))\n"
    )
    step = _make_step(llm=_StubLLM([code]), session_root=tmp_path / "session")
    h = _make_hypothesis()
    exp = step(h)

    assert exp.succeeded is True
    assert len(exp.attempts) == 1
    a = exp.attempts[0]
    assert a.exit_code == 0, a.stderr
    assert a.metrics.get("mean_age") is not None
    assert a.metrics.get("n") == len(_load_adult())  # full dataset row count


def test_experiment_retry_then_success(tmp_path: Path):
    broken = (
        "x = df['nonexistent_column'].mean()\n"
        "print(x)\n"
    )
    fixed = (
        "x = df['age'].mean()\n"
        "print(f'mean_age={x:.4f}')\n"
        "hl_emit('mean_age', float(x))\n"
    )
    step = _make_step(llm=_StubLLM([broken, fixed]), session_root=tmp_path / "session")
    exp = step(_make_hypothesis())

    assert exp.succeeded is True
    assert len(exp.attempts) == 2
    assert exp.attempts[0].exit_code != 0
    assert (
        "nonexistent_column" in exp.attempts[0].stderr
        or "KeyError" in exp.attempts[0].stderr
    )
    assert exp.attempts[1].exit_code == 0


def test_experiment_max_retries_exhausted(tmp_path: Path):
    broken = "x = df['nope'].mean()\nprint(x)\n"
    step = _make_step(
        llm=_StubLLM([broken] * 4),
        session_root=tmp_path / "session",
        max_retries=3,
    )
    exp = step(_make_hypothesis())

    assert exp.succeeded is False
    assert len(exp.attempts) == 4  # initial + 3 retries
    assert all(a.exit_code != 0 for a in exp.attempts)


def test_experiment_blocked_import_triggers_retry(tmp_path: Path):
    blocked = "import os\nprint(df.shape)\n"
    fixed = (
        "print(df.shape)\n"
        "hl_emit('rows', int(df.shape[0]))\n"
    )
    step = _make_step(llm=_StubLLM([blocked, fixed]), session_root=tmp_path / "session")
    exp = step(_make_hypothesis())

    assert exp.succeeded is True
    assert len(exp.attempts) == 2
    a0 = exp.attempts[0]
    assert a0.blocked_reason is not None
    assert "os" in a0.blocked_reason
    assert exp.attempts[1].exit_code == 0


# ---------------------------------------------------------------------------
# CJK identifier lint (Phase 8.1)
# ---------------------------------------------------------------------------
def test_extract_python_code_rejects_cjk_identifier():
    code = "df大概率 = df.mean()\n"
    err = _check_ascii_identifiers(code)
    assert err is not None
    assert "non-ASCII" in err


def test_check_ascii_identifiers_allows_unicode_strings():
    code = 'msg = "你好"\nprint(msg)\n'
    err = _check_ascii_identifiers(code)
    assert err is None


def test_experiment_cjk_identifier_triggers_retry(tmp_path: Path):
    """A CJK-tainted attempt should count against the retry budget; clean code on retry succeeds."""
    cjk = "df大概率 = df['age'].mean()\nprint(df大概率)\n"
    fixed = (
        "x = df['age'].mean()\n"
        "print(f'mean_age={x:.4f}')\n"
        "hl_emit('mean_age', float(x))\n"
    )
    step = _make_step(llm=_StubLLM([cjk, fixed]), session_root=tmp_path / "session")
    exp = step(_make_hypothesis())

    assert exp.succeeded is True
    assert len(exp.attempts) == 2
    a0 = exp.attempts[0]
    assert a0.blocked_reason is not None
    assert "non-ASCII" in a0.blocked_reason
    assert exp.attempts[1].exit_code == 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_adult() -> "pd.DataFrame":  # type: ignore[name-defined]
    import pandas as pd

    return pd.read_csv(_DATASET)
