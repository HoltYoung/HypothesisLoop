from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader


@dataclass
class RagChunk:
    text: str
    source: str
    heading: str


def load_markdown_files(knowledge_dir: str | Path) -> list[tuple[str, str]]:
    """Return markdown docs as (relative_path, file_text)."""
    knowledge_dir = Path(knowledge_dir)
    docs: list[tuple[str, str]] = []

    for fp in sorted(knowledge_dir.rglob("*")):
        if fp.suffix.lower() != ".md":
            continue
        text = fp.read_text(encoding="utf-8")
        docs.append((str(fp.relative_to(knowledge_dir)), text))

    return docs


def load_pdf_files(
    knowledge_dir: str | Path,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Return PDF docs as:
    [
        (relative_pdf_path, [("Page 1", page_text), ("Page 2", page_text), ...]),
        ...
    ]
    """
    knowledge_dir = Path(knowledge_dir)
    pdf_docs: list[tuple[str, list[tuple[str, str]]]] = []

    for fp in sorted(knowledge_dir.rglob("*")):
        if fp.suffix.lower() != ".pdf":
            continue

        reader = PdfReader(str(fp))
        pages: list[tuple[str, str]] = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text(extraction_mode="layout") or ""
            text = text.strip()

            if text:
                pages.append((f"Page {i}", text))

        if pages:
            pdf_docs.append((str(fp.relative_to(knowledge_dir)), pages))

    return pdf_docs


def chunk_markdown_by_heading(source: str, text: str) -> list[RagChunk]:
    """Split markdown by headings. Fall back to paragraph chunks if needed."""
    lines = text.splitlines()
    chunks: list[RagChunk] = []

    current_heading = "Document Start"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_heading
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(
                RagChunk(
                    text=body,
                    source=source,
                    heading=current_heading,
                )
            )
        current_lines = []

    for line in lines:
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            flush()
            current_heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()
        else:
            current_lines.append(line)

    flush()

    if not chunks:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paras, start=1):
            chunks.append(
                RagChunk(
                    text=para,
                    source=source,
                    heading=f"Paragraph {i}",
                )
            )

    return chunks


def chunk_pdf_pages(
    source: str, pages: list[tuple[str, str]], max_chars: int = 1800
) -> list[RagChunk]:
    """
    Chunk PDF text page-by-page, with optional splitting of very long pages.
    """
    chunks: list[RagChunk] = []

    for page_label, page_text in pages:
        if len(page_text) <= max_chars:
            chunks.append(
                RagChunk(
                    text=page_text,
                    source=source,
                    heading=page_label,
                )
            )
            continue

        paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        current: list[str] = []
        current_len = 0
        part_num = 1

        def flush_part() -> None:
            nonlocal current, current_len, part_num
            if current:
                chunks.append(
                    RagChunk(
                        text="\n\n".join(current),
                        source=source,
                        heading=f"{page_label} - Part {part_num}",
                    )
                )
                current = []
                current_len = 0
                part_num += 1

        for para in paras:
            add_len = len(para) + 2
            if current and current_len + add_len > max_chars:
                flush_part()
            current.append(para)
            current_len += add_len

        flush_part()

    return chunks


def build_rag_chunks(knowledge_dir: str | Path) -> list[RagChunk]:
    """
    Build chunks from both markdown and PDF files.
    Markdown is chunked by headings.
    PDFs are chunked by page / page-part.
    """
    all_chunks: list[RagChunk] = []

    for source, text in load_markdown_files(knowledge_dir):
        all_chunks.extend(chunk_markdown_by_heading(source, text))

    for source, pages in load_pdf_files(knowledge_dir):
        all_chunks.extend(chunk_pdf_pages(source, pages))

    return all_chunks


def make_embedding_text(chunk: RagChunk) -> str:
    return f"SOURCE: {chunk.source}\nSECTION: {chunk.heading}\n\n{chunk.text}"


def build_faiss_index(
    chunks: list[RagChunk],
    embedding_model: str = "text-embedding-3-small",
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    if not chunks:
        raise ValueError(
            "No chunks found. Add .md or .pdf files to the knowledge directory first."
        )

    embeddings = OpenAIEmbeddings(model=embedding_model)
    texts = [make_embedding_text(c) for c in chunks]

    vectors = embeddings.embed_documents(texts)
    matrix = np.array(vectors, dtype="float32")

    faiss.normalize_L2(matrix)
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index, matrix


def save_faiss_index(
    knowledge_dir: str | Path,
    index: faiss.IndexFlatIP,
    chunks: list[RagChunk],
    embedding_model: str,
) -> tuple[Path, Path]:
    knowledge_dir = Path(knowledge_dir)
    index_path = knowledge_dir / "rag_faiss.index"
    meta_path = knowledge_dir / "rag_chunks.pkl"

    faiss.write_index(index, str(index_path))

    payload = {
        "embedding_model": embedding_model,
        "chunks": chunks,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(payload, f)

    return index_path, meta_path


def load_faiss_index(
    knowledge_dir: str | Path,
) -> tuple[faiss.IndexFlatIP, list[RagChunk], str]:
    knowledge_dir = Path(knowledge_dir)
    index_path = knowledge_dir / "rag_faiss.index"
    meta_path = knowledge_dir / "rag_chunks.pkl"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Missing saved RAG index files. Expected both "
            f"{index_path.name} and {meta_path.name} in {knowledge_dir}."
        )

    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        payload = pickle.load(f)

    chunks = payload["chunks"]
    embedding_model = payload.get("embedding_model", "unknown")

    return index, chunks, embedding_model


def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[RagChunk],
    k: int = 4,
    embedding_model: str = "text-embedding-3-small",
) -> list[tuple[RagChunk, float]]:
    if not chunks:
        return []

    embeddings = OpenAIEmbeddings(model=embedding_model)
    q = np.array([embeddings.embed_query(query)], dtype="float32")
    faiss.normalize_L2(q)

    k = min(k, len(chunks))
    scores, ids = index.search(q, k)

    results: list[tuple[RagChunk, float]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append((chunks[idx], float(score)))

    return results


def format_rag_context(results: list[tuple[RagChunk, float]]) -> str:
    if not results:
        return "No retrieved reference material."

    parts = []
    for i, (chunk, score) in enumerate(results, start=1):
        parts.append(
            f"[Reference {i} | source={chunk.source} | section={chunk.heading} | score={score:.3f}]\n"
            f"{chunk.text}"
        )

    return "\n\n---\n\n".join(parts)
