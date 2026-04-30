"""Phase 7 bias scanner unit tests."""

from __future__ import annotations

from hypothesisloop.agent.state import (
    Hypothesis,
    HypothesisFeedback,
    TraceNode,
    new_hypothesis_id,
)
from hypothesisloop.safety.bias_scanner import (
    DISCLAIMER,
    add_disclaimers,
    scan_node,
    scan_text,
)


def test_scan_no_sensitive_var_no_flag():
    flags = scan_text("Higher education leads to higher income.")
    assert flags == []


def test_scan_sensitive_var_no_causal_no_flag():
    flags = scan_text("Race correlates with income.")
    assert flags == []


def test_scan_planted_causal_race_flags():
    flags = scan_text("Race causes income disparities.", source="hypothesis")
    assert len(flags) == 1
    f = flags[0]
    assert f.sensitive_var == "race"
    assert "cause" in f.causal_verb.lower()
    assert "Race causes income disparities" in f.snippet
    assert f.source == "hypothesis"


def test_scan_synonym_sex_gender():
    flags = scan_text("Gender leads to income gaps.")
    assert len(flags) == 1
    assert flags[0].sensitive_var == "sex"
    assert "leads" in flags[0].causal_verb.lower()


def test_scan_native_country():
    flags = scan_text("Being foreign-born results in lower wages.")
    assert len(flags) == 1
    assert flags[0].sensitive_var == "native_country"


def test_scan_multiple_flags():
    text = (
        "Race causes income disparities. "
        "Marital status drives household earnings as well."
    )
    flags = scan_text(text)
    assert len(flags) == 2
    vars_seen = {f.sensitive_var for f in flags}
    assert vars_seen == {"race", "relationship"}


def test_add_disclaimers_idempotent():
    text = (
        "## §3 Hypothesis chain\n\n"
        "Race causes income disparities, which is striking.\n\n"
        "Other unrelated paragraph."
    )
    flags = scan_text(text)
    once = add_disclaimers(text, flags)
    twice = add_disclaimers(once, flags)

    assert once.count(DISCLAIMER.split("\n", 1)[0]) == 1
    assert twice.count(DISCLAIMER.split("\n", 1)[0]) == 1, "applying twice must not duplicate"
    assert once == twice


def test_scan_node_mutates_feedback():
    h = Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=1,
        statement="Race causes income disparities.",
        null="not Race causes income disparities.",
        test_type="custom",
        target_columns=["race", "income"],
        expected_outcome="?",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
    )
    fb = HypothesisFeedback(
        hypothesis_id=h.id,
        decision="confirmed",
        reason="The income gap is due to gender-based factors.",
        observations="diff is large",
        novel_subhypotheses=[],
        confidence=0.7,
    )
    node = TraceNode(id=h.id, parent_id=None, iteration=1, hypothesis=h, feedback=fb)
    flags = scan_node(node)

    assert len(flags) >= 2  # statement + reason both flagged
    assert any(f.source == "hypothesis" for f in flags)
    assert any(f.source == "feedback_reason" for f in flags)
    assert len(node.feedback.bias_flags) == len(flags)
    # Flags are stored as plain dicts on the dataclass.
    assert all(isinstance(d, dict) for d in node.feedback.bias_flags)
