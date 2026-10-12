"""Turns the FHWA guidance PDFs into page-anchored chunks for retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # doc:page:index, the citation the agent has to use
    doc: str
    page: int
    text: str


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}")
_PHONE = re.compile(r"\(?\b\d{3}\)?[ -]\d{3}-\d{4}\b")


def clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    # contact details of the authors add nothing to retrieval and are personal data
    text = _PHONE.sub("[phone removed]", _EMAIL.sub("[email removed]", text))
    text = re.sub(r"-\n(\w)", r"\1", text)  # words broken across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def chunk_pages(doc: str, pages: list[str], words: int = 160, overlap: int = 40) -> list[Chunk]:
    """Fixed word windows inside a page. Chunks never cross pages so a citation is a page."""
    out: list[Chunk] = []
    step = words - overlap
    for page_no, raw in enumerate(pages, start=1):
        text = clean(raw)
        tokens = text.split()
        if len(tokens) < 25:  # covers, blank pages, figure-only pages
            continue
        if len(re.findall(r"\.{4,}", text)) > 5:  # tables of contents and lists of figures
            continue
        i = 0
        for start in range(0, max(1, len(tokens) - overlap), step):
            piece = tokens[start : start + words]
            if len(piece) < 25 and start > 0:
                break
            out.append(Chunk(f"{doc}:{page_no}:{i}", doc, page_no, " ".join(piece)))
            i += 1
    return out


def pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [p.extract_text() or "" for p in PdfReader(str(path)).pages]


def build_corpus(sources: dict[str, Path], out_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc, path in sources.items():
        chunks.extend(chunk_pages(doc, pdf_pages(path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")
    return chunks


def load_corpus(path: Path) -> list[Chunk]:
    with path.open() as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
