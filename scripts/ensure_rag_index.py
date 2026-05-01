"""Build the RAG index if it doesn't already exist.

Run at container startup / local-launch. Cheap (~5 sec, ~$0.0001) when needed,
no-op when the files are already there.
"""
from __future__ import annotations

import sys
from pathlib import Path

INDEX_PATH = Path("knowledge/rag.index")
CHUNKS_PATH = Path("knowledge/rag_chunks.pkl")


def main() -> int:
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        print(f"[ensure_rag_index] index already present at {INDEX_PATH} — skipping rebuild")
        return 0

    print("[ensure_rag_index] index missing — rebuilding…")
    from dotenv import load_dotenv
    load_dotenv(override=True)

    import runpy
    runpy.run_path("scripts/build_rag_index.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
