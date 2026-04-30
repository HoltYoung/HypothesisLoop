"""Build the HypothesisLoop RAG index.

Indexes exactly two files (per SPEC §8):

    knowledge/adult_codebook.md
    knowledge/test_selection.md

and writes:

    knowledge/rag.index           (FAISS, inner-product, normalized)
    knowledge/rag_chunks.pkl      (chunk metadata + embedding model name)

Embedding model: ``text-embedding-3-small`` (per SPEC §6.8). Cost is
estimated assuming OpenAI's published price of $0.02 / 1M tokens for that
model.

Usage:
    python scripts/build_rag_index.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# CRITICAL: override=True — the user's shell may have a stale OPENAI_API_KEY.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

from hypothesisloop.primitives.rag import (  # noqa: E402  (after dotenv)
    build_faiss_index,
    chunk_markdown_by_heading,
    make_embedding_text,
    save_index,
)


# OpenAI text-embedding-3-small list price as of 2026-04.
_EMBED_PRICE_PER_1M_TOKENS = 0.02


def _read_corpus(knowledge_dir: Path) -> list[tuple[str, str]]:
    sources = ["adult_codebook.md", "test_selection.md"]
    docs: list[tuple[str, str]] = []
    for name in sources:
        fp = knowledge_dir / name
        if not fp.exists():
            raise FileNotFoundError(f"Missing required RAG source: {fp}")
        docs.append((name, fp.read_text(encoding="utf-8")))
    return docs


def _estimate_cost(chunks) -> tuple[int, float]:
    # Crude tokens-per-char estimator (~4 chars/token for English prose).
    chars = sum(len(make_embedding_text(c)) for c in chunks)
    tokens = max(1, chars // 4)
    cost_usd = tokens / 1_000_000 * _EMBED_PRICE_PER_1M_TOKENS
    return tokens, cost_usd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--knowledge-dir",
        default=str(PROJECT_ROOT / "knowledge"),
        help="Directory holding adult_codebook.md and test_selection.md.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    args = parser.parse_args()

    knowledge_dir = Path(args.knowledge_dir)
    docs = _read_corpus(knowledge_dir)

    chunks = []
    for source, text in docs:
        chunks.extend(chunk_markdown_by_heading(source, text))

    if not chunks:
        raise ValueError("No chunks built — check the source markdown files.")

    tokens_est, cost_est = _estimate_cost(chunks)
    print(f"Sources       : {[s for s, _ in docs]}")
    print(f"Chunks built  : {len(chunks)}")
    print(f"Tokens (est.) : {tokens_est:,}")
    print(f"Cost (est.)   : ${cost_est:.6f}  (model={args.embedding_model})")

    index, _ = build_faiss_index(chunks, embedding_model=args.embedding_model)

    index_path = knowledge_dir / "rag.index"
    chunks_path = knowledge_dir / "rag_chunks.pkl"
    save_index(
        index,
        chunks,
        index_path=index_path,
        chunks_path=chunks_path,
        embedding_model=args.embedding_model,
    )

    print(f"\nWrote {index_path}")
    print(f"Wrote {chunks_path}")


if __name__ == "__main__":
    main()
