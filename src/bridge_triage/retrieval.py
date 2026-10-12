"""Hybrid retrieval over the FHWA chunks: BM25 + dense vectors, fused with reciprocal rank
fusion, optionally reranked by a cross-encoder.

Dense and rerank models run locally through fastembed (ONNX, CPU). A hashing embedder is
available so tests and CI run without downloading anything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Chunk

_STOP = set(
    "a an the of and or to in on for with by at from as is are was were be been it its this that "
    "these those which what how why when where who do does did can should would than then there "
    "their them they we you your our not no into over under between after before such each any".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag of hashed unigrams and bigrams. No model, no network."""

    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class FastEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.name = model
        self.model = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=2)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts, batch_size=16)), dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self.model = TextCrossEncoder(model_name=model, cache_dir=cache_dir, threads=2)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts, batch_size=16)]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    sources: tuple[str, ...]  # which retrievers ranked it

    @property
    def page_key(self) -> str:
        return f"{self.chunk.doc}:{self.chunk.page}"


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion. k=60 is the value from Cormack et al. 2009."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        cache_path: Path | None = None,
        depth: int = 30,
    ):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.embedder = embedder
        self.reranker = reranker
        self.depth = depth
        self.vectors = self._vectors(cache_path) if embedder else None

    def _vectors(self, cache_path: Path | None) -> np.ndarray:
        texts = [c.text for c in self.chunks]
        key = hashlib.sha1(("\n".join(texts) + self.embedder.name).encode()).hexdigest()[:16]
        if cache_path and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["key"]) == key:
                return cached["vectors"]
        vecs = self.embedder.embed(texts)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, key=np.array(key), vectors=vecs)
        return vecs

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[: self.depth]]

    def _dense_rank(self, query: str) -> list[int]:
        q = self.embedder.embed([query])[0]
        sims = self.vectors @ q
        return [int(i) for i in np.argsort(-sims)[: self.depth]]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        """mode: bm25, dense, hybrid or hybrid+rerank."""
        rankings, names = [], []
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            rankings.append(self._bm25_rank(query))
            names.append("bm25")
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            if self.vectors is None:
                raise ValueError("dense retrieval needs an embedder")
            rankings.append(self._dense_rank(query))
            names.append("dense")
        if not rankings:
            raise ValueError(f"unknown mode {mode}")
        fused = rrf(rankings)
        order = sorted(fused, key=lambda i: -fused[i])
        if mode == "hybrid+rerank":
            if self.reranker is None:
                raise ValueError("hybrid+rerank needs a reranker")
            pool = order[: self.depth]
            scores = self.reranker.scores(query, [self.chunks[i].text for i in pool])
            ranked = sorted(zip(pool, scores), key=lambda t: -t[1])
            fused = dict(ranked)
            order = [i for i, _ in ranked]
        out = []
        for idx in order[:k]:
            src = tuple(n for n, r in zip(names, rankings) if idx in r)
            out.append(Hit(self.chunks[idx], round(float(fused[idx]), 5), src))
        return out
