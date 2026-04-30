"""Report generator for HypothesisLoop runs.

Produces, side by side:
    reports/<session>/report.md   — Markdown with base64-embedded PNGs
    reports/<session>/report.txt  — plain text, no images, no markup

Section layout per SPEC §6.9:
    1. Run metadata
    2. Question & approach
    3. Hypothesis chain (the narrative)
    4. Key findings
    5. Rejections & dead-ends
    6. Bias flags raised
    7. Reasoning chain
    8. Limitations & caveats
    9. Reproduction
"""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path
from typing import Iterable, Literal, Optional

from dotenv import load_dotenv

from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    TraceNode,
)
from hypothesisloop.safety.bias_scanner import (
    BiasFlag,
    DISCLAIMER,
    add_disclaimers,
    scan_text,
)

# get_session_usage is lazy-imported inside _section_metadata so the report
# module doesn't pull langfuse on import. Keeps --report-only cold-start <5s.

load_dotenv(override=True)


DECISION_BADGES = {
    "confirmed": "✅",
    "rejected": "❌",
    "inconclusive": "⚠️",
    "invalid": "🚫",
}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def render_report(
    trace: DAGTrace,
    *,
    output_dir: Path | str,
    format: Literal["md", "txt", "both"] = "both",
    cli_command: Optional[str] = None,
    seed: int = 42,
    usage_override: Optional[dict] = None,
) -> dict:
    """Render Markdown + plain-text reports.

    Returns ``{"md": Path | None, "txt": Path | None, "bias_flags": list[dict]}``.
    The aggregate bias-flags list combines per-node ``feedback.bias_flags`` and
    any new flags surfaced when scanning the rendered Markdown.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_text = _render_markdown(
        trace, cli_command=cli_command, seed=seed, usage_override=usage_override
    )
    md_flags = scan_text(md_text, source="report")
    md_text = add_disclaimers(md_text, md_flags)

    md_path: Optional[Path] = None
    txt_path: Optional[Path] = None
    if format in ("md", "both"):
        md_path = output_dir / "report.md"
        md_path.write_text(md_text, encoding="utf-8")
    if format in ("txt", "both"):
        txt_path = output_dir / "report.txt"
        txt_path.write_text(_markdown_to_plain(md_text), encoding="utf-8")

    aggregate: list[dict] = []
    for node in trace.iter_nodes():
        if node.feedback is not None:
            aggregate.extend(node.feedback.bias_flags)
    aggregate.extend(
        [
            {
                "sensitive_var": f.sensitive_var,
                "causal_verb": f.causal_verb,
                "snippet": f.snippet,
                "source": f.source,
            }
            for f in md_flags
        ]
    )
    return {"md": md_path, "txt": txt_path, "bias_flags": aggregate}


# ---------------------------------------------------------------------------
# top-level renderer
# ---------------------------------------------------------------------------
def _render_markdown(
    trace: DAGTrace,
    *,
    cli_command: Optional[str],
    seed: int,
    usage_override: Optional[dict] = None,
) -> str:
    sections = [
        f"# HypothesisLoop run — `{trace.session_id}`",
        _section_metadata(trace, usage_override=usage_override),
        _section_question(trace),
        _section_hypothesis_chain(trace),
        _section_key_findings(trace),
        _section_rejections(trace),
        _section_bias_flags(trace),
        _section_reasoning_chain(trace),
        _section_limitations(trace),
        _section_reproduction(trace, cli_command=cli_command, seed=seed),
    ]
    return "\n\n".join(s.rstrip() for s in sections)


# ---------------------------------------------------------------------------
# §1 metadata
# ---------------------------------------------------------------------------
def _section_metadata(
    trace: DAGTrace, *, usage_override: Optional[dict] = None
) -> str:
    if usage_override is not None:
        usage = usage_override
    else:
        from hypothesisloop.trace.langfuse_client import get_session_usage  # lazy

        usage = get_session_usage(trace.session_id)
    iterations = trace.iteration_count()
    err = usage.get("_error")

    rows = [
        ("session_id", f"`{trace.session_id}`"),
        ("dataset", f"`{trace.dataset_path}`"),
        ("model", "see §9 (CLI invocation)"),
        ("iterations completed", str(iterations)),
        ("started_at", trace.created_at),
        ("total tokens", f"{usage.get('total_tokens', 0):,}"),
        ("input tokens", f"{usage.get('input_tokens', 0):,}"),
        ("output tokens", f"{usage.get('output_tokens', 0):,}"),
        ("total cost (USD)", f"${float(usage.get('total_cost_usd', 0.0)):.4f}"),
        ("wall time (s)", f"{float(usage.get('wall_time_s', 0.0)):.1f}"),
        ("Langfuse traces", str(usage.get("trace_count", 0))),
    ]
    body = "## 1. Run metadata\n\n| field | value |\n|---|---|\n"
    body += "\n".join(f"| {k} | {v} |" for k, v in rows)
    if err:
        body += (
            f"\n\n_Note: Langfuse usage rollup unavailable "
            f"(`{err}`); token / cost figures show 0._"
        )
    return body


# ---------------------------------------------------------------------------
# §2 question
# ---------------------------------------------------------------------------
def _section_question(trace: DAGTrace) -> str:
    return (
        "## 2. Question & approach\n\n"
        f"**Research question.** {trace.question}\n\n"
        "**Approach.** HypothesisLoop wraps the scientific method around the "
        "dataset above. Each iteration proposes a testable hypothesis, generates "
        "Python code in a sandbox, evaluates the result, and feeds the learning "
        "into the next round. Novelty checking and a soft-decay gate prevent "
        "the agent from re-testing what it has already settled."
    )


# ---------------------------------------------------------------------------
# §3 hypothesis chain
# ---------------------------------------------------------------------------
def _section_hypothesis_chain(trace: DAGTrace) -> str:
    nodes = trace.iter_nodes()
    parts = ["## 3. Hypothesis chain"]
    if not nodes:
        parts.append("\nNo iterations completed.")
        return "\n".join(parts)

    for node in nodes:
        parts.append("")
        parts.append(_render_node_block(node))
    return "\n".join(parts)


def _render_node_block(node: TraceNode) -> str:
    fb = node.feedback
    badge = DECISION_BADGES.get(fb.decision, "·") if fb else "·"
    decision = fb.decision if fb else "(no feedback yet)"
    confidence = f"{fb.confidence:.2f}" if fb else "—"
    re_explore = " · re-explored" if node.hypothesis.re_explore else ""
    head = f"### Iteration {node.iteration} {badge} {decision} (confidence: {confidence}){re_explore}"

    body = [head, "", node.hypothesis.statement]

    code_block = _render_code_block(node.experiment)
    if code_block:
        body.append("")
        body.append(code_block)

    figures_block = _render_figures(node.experiment)
    if figures_block:
        body.append("")
        body.append(figures_block)

    if fb is not None:
        body.append("")
        body.append(f"**Evaluation.** {fb.reason}")
        if fb.observations:
            body.append("")
            body.append(f"_Observations:_ {fb.observations}")

    metrics_block = _render_metrics(node.experiment)
    if metrics_block:
        body.append("")
        body.append(metrics_block)

    return "\n".join(body)


def _pick_attempt_for_display(experiment: Optional[Experiment]) -> Optional[ExperimentAttempt]:
    if experiment is None or not experiment.attempts:
        return None
    return experiment.attempts[-1]


def _render_code_block(experiment: Optional[Experiment]) -> str:
    attempt = _pick_attempt_for_display(experiment)
    if attempt is None or not attempt.code:
        return ""
    note = (
        f"Generated code (attempt {attempt.attempt_idx})"
        if (experiment is not None and experiment.succeeded)
        else f"Generated code (attempt {attempt.attempt_idx} — all retries failed)"
    )
    return (
        f"<details><summary>{note}</summary>\n\n"
        f"```python\n{attempt.code}\n```\n\n"
        f"</details>"
    )


def _render_figures(experiment: Optional[Experiment]) -> str:
    attempt = _pick_attempt_for_display(experiment)
    if attempt is None or not attempt.figures:
        return ""
    parts: list[str] = []
    for fig in attempt.figures:
        embedded = _embed_png(Path(fig))
        if embedded:
            parts.append(embedded)
    return "\n\n".join(parts)


def _render_metrics(experiment: Optional[Experiment]) -> str:
    attempt = _pick_attempt_for_display(experiment)
    if attempt is None or not attempt.metrics:
        return ""
    rows = ["| metric | value |", "|---|---|"]
    for k, v in attempt.metrics.items():
        rows.append(f"| `{k}` | {v} |")
    return "\n".join(rows)


def _embed_png(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return f"![{path.name}](data:image/png;base64,{b64})"


# ---------------------------------------------------------------------------
# §4 key findings
# ---------------------------------------------------------------------------
def _section_key_findings(trace: DAGTrace) -> str:
    confirmed = [
        node
        for node in trace.iter_nodes()
        if node.feedback is not None and node.feedback.decision == "confirmed"
    ]
    confirmed.sort(key=lambda n: n.feedback.confidence, reverse=True)
    top = confirmed[:3]

    parts = ["## 4. Key findings"]
    if not top:
        parts.append("\nNo confirmed hypotheses in this run.")
        return "\n".join(parts)
    parts.append("")
    for node in top:
        parts.append(
            f"- **Iter {node.iteration}** "
            f"(confidence {node.feedback.confidence:.2f}): "
            f"{node.hypothesis.statement} — "
            f"{_truncate(node.feedback.reason, 200)}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# §5 rejections & dead-ends
# ---------------------------------------------------------------------------
def _section_rejections(trace: DAGTrace) -> str:
    rejected_nodes = [
        node
        for node in trace.iter_nodes()
        if node.feedback is not None
        and node.feedback.decision in {"rejected", "inconclusive", "invalid"}
    ]
    parts = ["## 5. Rejections & dead-ends"]

    if rejected_nodes:
        parts.append("")
        parts.append("| iter | decision | statement | reason |")
        parts.append("|---|---|---|---|")
        for node in rejected_nodes:
            parts.append(
                f"| {node.iteration} | {node.feedback.decision} | "
                f"{_md_escape(node.hypothesis.statement)} | "
                f"{_md_escape(_truncate(node.feedback.reason, 220))} |"
            )
    else:
        parts.append("")
        parts.append("No rejected, inconclusive, or invalid iterations.")

    if trace.novelty_rejected:
        parts.append("")
        parts.append(f"### Novelty-rejected duplicates ({len(trace.novelty_rejected)})")
        parts.append("")
        for h in trace.novelty_rejected:
            parts.append(f"- {h.statement}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# §6 bias flags
# ---------------------------------------------------------------------------
def _section_bias_flags(trace: DAGTrace) -> str:
    rows: list[tuple[int, dict]] = []
    for node in trace.iter_nodes():
        if node.feedback is None:
            continue
        for flag in node.feedback.bias_flags:
            rows.append((node.iteration, flag))

    parts = ["## 6. Bias flags raised"]
    if not rows:
        parts.append("")
        parts.append("No bias flags raised in this run.")
        return "\n".join(parts)

    parts.append("")
    parts.append("| iter | sensitive_var | causal_verb | source | snippet |")
    parts.append("|---|---|---|---|---|")
    for iteration, flag in rows:
        parts.append(
            f"| {iteration} | {flag.get('sensitive_var', '')} | "
            f"{flag.get('causal_verb', '')} | {flag.get('source', '')} | "
            f"{_md_escape(_truncate(flag.get('snippet', ''), 200))} |"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# §7 reasoning chain
# ---------------------------------------------------------------------------
def _section_reasoning_chain(trace: DAGTrace) -> str:
    parts = ["## 7. Reasoning chain"]
    nodes = trace.iter_nodes()
    if not nodes:
        parts.append("")
        parts.append("No iterations completed.")
        return "\n".join(parts)
    parts.append("")
    for i, node in enumerate(nodes, start=1):
        justification = (node.hypothesis.concise_justification or "").strip()
        if not justification:
            justification = "_(no justification recorded)_"
        parts.append(f"{i}. **Iter {node.iteration}.** {justification}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# §8 limitations
# ---------------------------------------------------------------------------
def _section_limitations(trace: DAGTrace) -> str:
    bullets: list[str] = []
    nodes = trace.iter_nodes()

    failed_experiments = sum(
        1
        for n in nodes
        if n.experiment is not None and not n.experiment.succeeded
    )
    if failed_experiments:
        bullets.append(
            f"- **{failed_experiments}** iteration(s) had retries that hit the cap."
        )

    re_explored = sum(1 for n in nodes if n.hypothesis.re_explore)
    if re_explored:
        bullets.append(
            f"- **{re_explored}** hypothesis(es) marked `re_explore` — soft-decay engaged."
        )

    inconclusive = sum(
        1
        for n in nodes
        if n.feedback is not None and n.feedback.decision == "inconclusive"
    )
    if inconclusive:
        bullets.append(f"- **{inconclusive}** iteration(s) returned inconclusive verdicts.")

    bias_flag_count = sum(
        len(n.feedback.bias_flags) for n in nodes if n.feedback is not None
    )
    if bias_flag_count:
        bullets.append(
            f"- **{bias_flag_count}** bias flag(s) raised — interpret findings about "
            "sensitive variables as correlational only."
        )

    bullets.append(
        "- Run is deterministic only with `seed=42` and the same model + temperature; "
        "LLM outputs vary across runs."
    )

    parts = ["## 8. Limitations & caveats", ""] + bullets
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# §9 reproduction
# ---------------------------------------------------------------------------
def _section_reproduction(
    trace: DAGTrace, *, cli_command: Optional[str], seed: int
) -> str:
    sha = _git_sha()
    cmd = cli_command or "# command not recorded for this run"
    return (
        "## 9. Reproduction\n\n"
        f"```bash\n{cmd}\n```\n\n"
        f"- Git SHA: `{sha}`\n"
        f"- Seed: `{seed}`\n"
        f"- Dataset: `{trace.dataset_path}`\n"
        f"- Requirements pinned in `requirements.txt` "
        f"(install with `pip install -r requirements.txt && pip install -e .`).\n"
        f"- Trace artifact: `trace.json` lives next to this report."
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def _md_escape(text: str) -> str:
    """Escape characters that would break a Markdown table row."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ")


def _git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            .decode()
            .strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _markdown_to_plain(md: str) -> str:
    """Strip Markdown decorations for the .txt sibling. Crude but readable."""
    text = md
    # Drop image tags entirely.
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)
    # Drop <details>...</details> blocks.
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL)
    # Drop fenced code blocks (the raw code is in the trace + on disk).
    text = re.sub(r"```[a-zA-Z]*\n.*?\n```", "[code block omitted]", text, flags=re.DOTALL)
    # Strip ** and __ emphasis.
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    # Strip inline code backticks.
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Strip headings (# ## ### ...)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Strip blockquote markers.
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # Replace badges with words.
    text = (
        text.replace("✅", "[CONFIRMED]")
        .replace("❌", "[REJECTED]")
        .replace("⚠️", "[WARNING]")
        .replace("🚫", "[INVALID]")
    )
    # Collapse multiple blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


__all__ = ["DECISION_BADGES", "render_report"]
