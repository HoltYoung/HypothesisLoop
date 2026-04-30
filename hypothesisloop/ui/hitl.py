"""Human-in-the-loop CLI prompt + end-of-run summary printer.

Per locked decision row 2 of SPEC §4, the HITL gate fires once per iteration
with a ``[c]ontinue / [s]top / [r]edirect <text>`` prompt. ``stream_in`` and
``stream_out`` are injected so tests can drive the prompt with ``StringIO``.
"""

from __future__ import annotations

import sys
from typing import IO, Optional

from hypothesisloop.agent.state import DAGTrace, TraceNode


HITL_HELP = "[c]ontinue / [s]top / [r]edirect <text>"


def hitl_prompt(
    node: TraceNode,
    *,
    stream_in: Optional[IO] = None,
    stream_out: Optional[IO] = None,
) -> dict:
    """Print the iteration result and read the user's next-step decision.

    Loops until valid input. Returns one of:
        {"action": "continue"}
        {"action": "stop"}
        {"action": "redirect", "hypothesis": "..."}

    ``stream_in`` / ``stream_out`` resolve at call time (not as default args)
    so pytest's stdout capture and ``contextlib.redirect_stdout`` reach the
    live streams. Default-argument capture freezes the original handles.
    """
    if stream_in is None:
        stream_in = sys.stdin
    if stream_out is None:
        stream_out = sys.stdout
    fb = node.feedback
    print("", file=stream_out)
    print(f"=== iter {node.iteration} ===", file=stream_out)
    print(f"hypothesis: {node.hypothesis.statement}", file=stream_out)
    if fb is not None:
        print(f"  decision : {fb.decision}  (confidence={fb.confidence:.2f})", file=stream_out)
        print(f"  reason   : {fb.reason}", file=stream_out)
    if node.experiment is not None:
        attempts = len(node.experiment.attempts)
        print(
            f"  attempts : {attempts}  succeeded={node.experiment.succeeded}",
            file=stream_out,
        )

    while True:
        print(f"\n{HITL_HELP}: ", end="", file=stream_out, flush=True)
        line = stream_in.readline()
        if not line:  # EOF (Ctrl+D / closed pipe / closed StringIO)
            return {"action": "stop"}

        cmd = line.strip()
        if cmd == "" or cmd == "c":
            return {"action": "continue"}
        if cmd == "s":
            return {"action": "stop"}
        if cmd.startswith("r ") and len(cmd) > 2:
            return {"action": "redirect", "hypothesis": cmd[2:].strip()}
        print(f"  invalid input '{cmd}'. Try again.", file=stream_out)


def print_run_summary(
    trace: DAGTrace, *, stream_out: Optional[IO] = None
) -> None:
    """Phase 6 placeholder for the eventual Markdown report (Phase 7).

    Prints a tight text summary of the run to ``stream_out`` (resolved at
    call time so pytest / ``redirect_stdout`` capture works).
    """
    if stream_out is None:
        stream_out = sys.stdout
    print("", file=stream_out)
    print(f"=== HypothesisLoop run: {trace.session_id} ===", file=stream_out)
    print(f"dataset      : {trace.dataset_path}", file=stream_out)
    print(f"question     : {trace.question}", file=stream_out)
    print(f"iterations   : {trace.iteration_count()}", file=stream_out)

    accepted = trace.iter_nodes()
    rejected = trace.novelty_rejected
    print(f"accepted     : {len(accepted)}", file=stream_out)
    print(f"novelty rej. : {len(rejected)}", file=stream_out)
    print("", file=stream_out)

    for node in accepted:
        decision = node.feedback.decision if node.feedback else "(no feedback)"
        confidence = f"{node.feedback.confidence:.2f}" if node.feedback else "-"
        flag = " · re-explored" if node.hypothesis.re_explore else ""
        print(
            f"  iter {node.iteration:>2}{flag}: [{decision} c={confidence}] "
            f"{node.hypothesis.statement}",
            file=stream_out,
        )

    if rejected:
        print(f"\n  rejected as duplicates ({len(rejected)}):", file=stream_out)
        for h in rejected:
            print(f"    - {h.statement[:120]}", file=stream_out)
    print("", file=stream_out)


__all__ = ["HITL_HELP", "hitl_prompt", "print_run_summary"]
