"""Bias scanner — flags causal claims about sensitive variables (SPEC §6.7).

Trigger: any sentence that mentions a sensitive variable AND a causal verb.

Sensitive variables (UCI Adult): race, sex/gender, native-country/nationality,
relationship/marital-status. Causal verbs include `causes`, `leads to`,
`results in`, `due to`, `drives`, `because of`, `makes`, `stems from`.

Three usage points (Phase 7 implements detection — not blocking):

1. In ``run_loop`` after evaluate, on hypothesis.statement +
   feedback.reason + feedback.observations. Mutates ``feedback.bias_flags``.
2. In :func:`hypothesisloop.steps.report.render_report`, on the rendered
   markdown. Prepends a ``DISCLAIMER`` blockquote above flagged paragraphs.
3. (Future, Phase 9) Pre-experiment, on ``hypothesis.statement``, to abort
   if a clearly causal claim slipped past the prompt's "no causation" rule.

Regex synonym groups are inclusive on purpose: false positives are cheap
(disclaimers are soft), misses on causal/sensitive claims are dangerous.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from hypothesisloop.agent.state import TraceNode


# Sensitive-variable synonym groups; canonicalized to one label per group.
_SENSITIVE_VAR_PATTERNS: dict[str, re.Pattern] = {
    "race": re.compile(r"\b(race|racial|ethnic(ity)?)\b", re.IGNORECASE),
    "sex": re.compile(r"\b(sex|gender|male|female|men|women)\b", re.IGNORECASE),
    "native_country": re.compile(
        r"\b(native[\s_-]?country|nationality|country of origin|immigrant|foreign(-?born)?)\b",
        re.IGNORECASE,
    ),
    "relationship": re.compile(
        r"\b(relationship status|marital(\s|-)?status|married|divorced|widowed|never[\s_-]?married|spouse)\b",
        re.IGNORECASE,
    ),
}

_CAUSAL_VERB_PATTERN = re.compile(
    r"\b(causes?|caused by|leads? to|due to|results? in|drives?|because of|"
    r"makes?|made by|stems? from|attributable to|responsible for)\b",
    re.IGNORECASE,
)

DISCLAIMER = (
    "> ⚠️ **Causal claim about a sensitive variable — interpret as correlational only.** "
    "This finding describes an association in the data; it does not establish causation."
)


@dataclass
class BiasFlag:
    sensitive_var: str
    causal_verb: str
    snippet: str
    source: str  # "hypothesis" | "feedback_reason" | "feedback_observations" | "report" | ...


# ---------------------------------------------------------------------------
# core scanner
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    """Best-effort paragraph-aware sentence split.

    Splits on blank-line paragraph boundaries first, then on
    ``. ! ?`` followed by whitespace + capital within each paragraph. Keeping
    paragraphs separate is what makes :func:`add_disclaimers` work — the
    flagged snippet has to be a substring of a single paragraph for the
    disclaimer to attach correctly.
    """
    if not text:
        return []
    sentences: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        flat = re.sub(r"\s+", " ", paragraph).strip()
        if not flat:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", flat)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def scan_text(text: str, *, source: str = "unknown") -> list[BiasFlag]:
    """Return one ``BiasFlag`` per ``(sentence, sensitive_var)`` pair."""
    flags: list[BiasFlag] = []
    for sentence in _split_sentences(text):
        causal_match = _CAUSAL_VERB_PATTERN.search(sentence)
        if not causal_match:
            continue
        causal_verb = causal_match.group(0).lower()
        for var_label, pattern in _SENSITIVE_VAR_PATTERNS.items():
            if pattern.search(sentence):
                flags.append(
                    BiasFlag(
                        sensitive_var=var_label,
                        causal_verb=causal_verb,
                        snippet=sentence,
                        source=source,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# disclaimer attachment (idempotent)
# ---------------------------------------------------------------------------
_DISCLAIMER_FIRST_LINE = DISCLAIMER.split("\n", 1)[0].strip()


def add_disclaimers(text: str, flags: list[BiasFlag]) -> str:
    """Prepend ``DISCLAIMER`` above each paragraph that contains a flagged sentence.

    Idempotent: applying twice does not duplicate disclaimers (we check whether
    the previous paragraph already starts with the disclaimer's signature line).
    """
    if not flags:
        return text

    flagged_snippets = [f.snippet.strip() for f in flags if f.snippet.strip()]
    if not flagged_snippets:
        return text

    # Split on blank-line-separated paragraphs; preserve the separator on output.
    paragraphs = re.split(r"\n{2,}", text)

    result: list[str] = []
    for para in paragraphs:
        # If THIS paragraph IS the disclaimer (idempotency: already-applied), keep
        # it but flag that the *next* paragraph already has a disclaimer above it.
        para_norm = para.strip()
        if para_norm.startswith(_DISCLAIMER_FIRST_LINE):
            result.append(para)
            continue

        contains_flag = any(snippet in para for snippet in flagged_snippets)
        if contains_flag:
            prev_is_disclaimer = (
                len(result) > 0
                and result[-1].strip().startswith(_DISCLAIMER_FIRST_LINE)
            )
            if not prev_is_disclaimer:
                result.append(DISCLAIMER)
        result.append(para)

    return "\n\n".join(result)


# ---------------------------------------------------------------------------
# trace-node helper (mutates feedback.bias_flags)
# ---------------------------------------------------------------------------
def scan_node(node: "TraceNode") -> list[BiasFlag]:
    """Scan a TraceNode's hypothesis + feedback. Mutates ``feedback.bias_flags``.

    Returns the flags newly added by this call (not the cumulative list).
    """
    flags: list[BiasFlag] = []
    flags.extend(scan_text(node.hypothesis.statement, source="hypothesis"))
    if node.feedback is not None:
        flags.extend(scan_text(node.feedback.reason, source="feedback_reason"))
        flags.extend(scan_text(node.feedback.observations, source="feedback_observations"))
        node.feedback.bias_flags.extend([asdict(f) for f in flags])
    return flags


__all__ = [
    "BiasFlag",
    "DISCLAIMER",
    "scan_text",
    "add_disclaimers",
    "scan_node",
]
