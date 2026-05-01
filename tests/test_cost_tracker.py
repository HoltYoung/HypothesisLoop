"""Phase 9 cost tracker unit tests."""

from __future__ import annotations

import threading

import pytest

from hypothesisloop.llm.cost_tracker import RATES, CostTracker, UsageRecord


class _UsageObj:
    """Mimics LangChain's ``usage_metadata`` (input_tokens / output_tokens)."""

    def __init__(self, input_tokens: int, output_tokens: int, cache: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache


def test_record_kimi_costing():
    t = CostTracker()
    rec = t.record("moonshot-v1-128k", _UsageObj(input_tokens=1000, output_tokens=500))
    expected = (1000 * 0.95 + 500 * 4.00) / 1_000_000
    assert rec.cost_usd == pytest.approx(expected, rel=1e-9)
    assert t.total_tokens == 1500
    assert t.total_calls == 1


def test_record_openai_costing():
    t = CostTracker()
    t.record("gpt-4o-mini", _UsageObj(input_tokens=2000, output_tokens=1000))
    expected = (2000 * 0.15 + 1000 * 0.60) / 1_000_000
    assert t.total_cost_usd == pytest.approx(expected, rel=1e-9)


def test_record_with_cache_hits_charges_cache_rate():
    t = CostTracker()
    # 1000 input tokens of which 400 came from cache → 400 at cache_hit, 600 at input.
    rec = t.record(
        "moonshot-v1-128k",
        _UsageObj(input_tokens=1000, output_tokens=200, cache=400),
    )
    expected = (600 * 0.95 + 400 * 0.16 + 200 * 4.00) / 1_000_000
    assert rec.cost_usd == pytest.approx(expected, rel=1e-9)
    assert rec.cache_tokens == 400


def test_record_handles_dict_usage_shape():
    """OpenAI's raw SDK uses prompt_tokens/completion_tokens dict shape."""
    t = CostTracker()
    rec = t.record(
        "gpt-4o-mini",
        {"prompt_tokens": 100, "completion_tokens": 50},
    )
    assert rec.input_tokens == 100
    assert rec.output_tokens == 50


def test_record_unknown_model_costs_zero():
    t = CostTracker()
    rec = t.record("some-future-model", _UsageObj(input_tokens=1_000_000, output_tokens=1_000_000))
    assert rec.cost_usd == 0.0
    assert t.total_tokens == 2_000_000


def test_by_model_aggregates_per_model():
    t = CostTracker()
    t.record("moonshot-v1-128k", _UsageObj(100, 50))
    t.record("moonshot-v1-128k", _UsageObj(200, 100))
    t.record("gpt-4o-mini", _UsageObj(50, 25))
    by = t.by_model()
    assert set(by.keys()) == {"moonshot-v1-128k", "gpt-4o-mini"}
    assert by["moonshot-v1-128k"]["calls"] == 2
    assert by["moonshot-v1-128k"]["input_tokens"] == 300
    assert by["moonshot-v1-128k"]["output_tokens"] == 150
    assert by["gpt-4o-mini"]["calls"] == 1


def test_thread_safe_concurrent_appends():
    t = CostTracker()
    n_threads = 8
    n_per_thread = 50

    def worker():
        for _ in range(n_per_thread):
            t.record("moonshot-v1-128k", _UsageObj(10, 5))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.total_calls == n_threads * n_per_thread
    assert t.total_tokens == n_threads * n_per_thread * 15


def test_rates_table_includes_known_models():
    for m in ("moonshot-v1-128k", "gpt-4o-mini", "gpt-4o", "text-embedding-3-small"):
        assert m in RATES
