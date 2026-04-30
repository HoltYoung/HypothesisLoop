"""Wires every loop component (LLMs, retriever, steps, novelty, pruner, scheduler).

Used by both the CLI (``hypothesisloop.cli``) and the Streamlit UI
(``hypothesisloop.ui.streamlit_app``) so behavior is identical across
front-ends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# CRITICAL: override=True — the user's shell may carry a stale OPENAI_API_KEY.
load_dotenv(override=True)

from hypothesisloop.agent.novelty import NoveltyChecker
from hypothesisloop.agent.pruner import Pruner
from hypothesisloop.agent.scheduler import LinearScheduler
from hypothesisloop.agent.state import DAGTrace
from hypothesisloop.llm.dispatch import get_llm
from hypothesisloop.llm.embed import embed_text
from hypothesisloop.primitives.rag import load_index, retrieve
from hypothesisloop.safety.bias_scanner import scan_node
from hypothesisloop.steps.evaluate import Evaluate
from hypothesisloop.steps.experiment import ExperimentStep
from hypothesisloop.steps.hypothesize import Hypothesize


def build_steps(
    *,
    trace: DAGTrace,
    session_root: Path | str,
    model: str = "moonshot-v1-128k",
    seed: int = 42,
    rag_index_path: str | Path = "knowledge/rag.index",
    rag_chunks_path: str | Path = "knowledge/rag_chunks.pkl",
    rag_k: int = 4,
) -> dict[str, Any]:
    """Construct every callable + helper the loop needs.

    Returns a dict with seven entries:
        hypothesize_fn, experiment_fn, evaluate_fn,
        novelty_fn, safety_fn, scheduler, pruner.

    Each is wired against the same ``trace`` (so step calls reference the
    correct dataset path / schema) and the shared scheduler (so HITL
    redirects from the front-end land in the next ``Hypothesize`` call).
    """
    llm_hyp = get_llm(model=model, temperature=0.7)
    llm_exp = get_llm(model=model, temperature=0.7)  # codegen also creative-ish
    llm_eval = get_llm(model=model, temperature=0.3)

    index, chunks = load_index(rag_index_path, rag_chunks_path)

    def retriever(q: str) -> list[dict]:
        return retrieve(q, index, chunks, k=rag_k)

    pruner = Pruner()
    scheduler = LinearScheduler()

    hypothesize_fn = Hypothesize(
        llm=llm_hyp,
        retriever=retriever,
        rag_k=rag_k,
        scheduler=scheduler,
        pruner=pruner,
    )
    experiment_fn = ExperimentStep(
        llm=llm_exp,
        session_root=session_root,
        dataset_path=trace.dataset_path,
        schema_summary=trace.schema_summary,
        seed=seed,
    )
    evaluate_fn = Evaluate(llm=llm_eval)
    novelty_fn = NoveltyChecker(embed_fn=embed_text)

    return {
        "hypothesize_fn": hypothesize_fn,
        "experiment_fn": experiment_fn,
        "evaluate_fn": evaluate_fn,
        "novelty_fn": novelty_fn,
        "safety_fn": scan_node,  # plain function — no construction needed
        "scheduler": scheduler,
        "pruner": pruner,
    }


__all__ = ["build_steps"]
