"""Phase 1 sandbox tests.

Eight required cases per Phase 1 spec, plus three bonus cases. The RAM-limit
test is skipped on Windows (``setrlimit`` is POSIX-only); a one-time
``RuntimeWarning`` is emitted in that environment instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hypothesisloop.sandbox.runner import (
    SandboxResult,
    _build_env,
    run_script,
)


# ---------------------------------------------------------------------------
# Required cases (1-8)
# ---------------------------------------------------------------------------
def test_success_pandas(tmp_path: Path):
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, 2, 3]})\n"
        "print(df['a'].mean())\n"
    )
    result = run_script(code, tmp_path)
    assert result.blocked_reason is None
    assert result.exit_code == 0, result.stderr
    assert "2.0" in result.stdout


def test_timeout(tmp_path: Path):
    code = "while True:\n    pass\n"
    result = run_script(code, tmp_path, timeout_s=2)
    assert result.timed_out is True
    assert result.exit_code != 0


@pytest.mark.skipif(sys.platform == "win32", reason="setrlimit unavailable on Windows")
def test_ram_limit(tmp_path: Path):
    code = "x = bytearray(2 * 1024 * 1024 * 1024)\n"
    result = run_script(code, tmp_path, ram_mb=512, timeout_s=10)
    assert result.oom_killed or result.exit_code != 0


def test_blocked_import_os(tmp_path: Path):
    code = "import os\n"
    result = run_script(code, tmp_path)
    assert result.blocked_reason is not None
    assert "os" in result.blocked_reason
    assert (tmp_path / "exp.py").exists(), "exp.py must be written for audit trail even when blocked"


def test_blocked_eval(tmp_path: Path):
    code = "eval('1+1')\n"
    result = run_script(code, tmp_path)
    assert result.blocked_reason is not None
    assert "eval" in result.blocked_reason


def test_blocked_open_write(tmp_path: Path):
    code = "open('foo.txt', 'w').write('bad')\n"
    result = run_script(code, tmp_path)
    assert result.blocked_reason is not None
    assert "open" in result.blocked_reason


def test_error_capture_for_retry(tmp_path: Path):
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1]})\n"
        "df.nonexistent\n"
    )
    result = run_script(code, tmp_path)
    assert result.blocked_reason is None
    assert result.exit_code != 0
    assert "AttributeError" in result.stderr or "nonexistent" in result.stderr
    assert len(result.stderr) <= 4096


def test_allowed_scipy(tmp_path: Path):
    code = (
        "from scipy import stats\n"
        "print(stats.ttest_1samp([1, 2, 3, 4, 5], 3.0).pvalue)\n"
    )
    result = run_script(code, tmp_path)
    assert result.blocked_reason is None
    assert result.exit_code == 0, result.stderr
    # Stdout should be a parseable float (scipy returns 1.0 for this exact case).
    val = float(result.stdout.strip().splitlines()[-1])
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Bonus cases (9-11)
# ---------------------------------------------------------------------------
def test_hl_emit_writes_metrics(tmp_path: Path):
    code = (
        "hl_emit('p_value', 0.03)\n"
        "hl_emit('n', 100)\n"
    )
    result = run_script(code, tmp_path)
    assert result.blocked_reason is None
    assert result.exit_code == 0, result.stderr
    assert result.metrics == {"p_value": 0.03, "n": 100}


def test_seed_reproducibility(tmp_path: Path):
    code = "import random\nprint(random.random())\n"
    r1 = run_script(code, tmp_path / "run1", seed=42)
    r2 = run_script(code, tmp_path / "run2", seed=42)
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert r1.stdout.strip() == r2.stdout.strip()


def test_env_scrubbed_unit():
    """_build_env() must not pass any *_API_KEY / LANGFUSE_* / OPENAI / MOONSHOT vars."""
    env = _build_env()
    forbidden = ("API_KEY", "SECRET", "TOKEN", "LANGFUSE", "MOONSHOT", "OPENAI")
    for k in env:
        upper = k.upper()
        for f in forbidden:
            assert f not in upper, f"{k!r} leaked through env scrub"


def test_env_scrubbed_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The preamble's .sandbox_env.json should record no API keys present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak")
    monkeypatch.setenv("KIMI_API_KEY", "kimi-test-should-not-leak")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-should-not-leak")
    code = "x = 1\n"  # trivial user code; we only care about the preamble side-effect
    result = run_script(code, tmp_path)
    assert result.blocked_reason is None
    assert result.exit_code == 0, result.stderr
    env_check_path = tmp_path / ".sandbox_env.json"
    assert env_check_path.exists(), "preamble should have written .sandbox_env.json"
    env_check = json.loads(env_check_path.read_text())
    assert env_check["openai_api_key_present"] is False
    assert env_check["kimi_api_key_present"] is False
    assert env_check["moonshot_api_key_present"] is False
    assert env_check["langfuse_keys_present"] is False
