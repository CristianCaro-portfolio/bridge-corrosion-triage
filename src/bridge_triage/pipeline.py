"""Wiring: builds the dependencies of the agent and runs it over one or many photos."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .agent import Deps, ReviewDecision, build_graph
from .config import Settings
from .corpus import load_corpus
from .llm import get_provider
from .retrieval import CrossEncoderReranker, FastEmbedder, HashingEmbedder, HybridRetriever
from .segment import OnnxSegmenter, OracleSegmenter


def make_retriever(settings: Settings) -> HybridRetriever:
    chunks = load_corpus(Path(settings.corpus_path))
    needs_dense = settings.retrieval_mode != "bm25"
    embedder = None
    if needs_dense:
        embedder = FastEmbedder() if settings.embedder == "fastembed" else HashingEmbedder()
    reranker = CrossEncoderReranker() if settings.retrieval_mode == "hybrid+rerank" else None
    cache = Path(settings.vector_cache) if settings.embedder == "fastembed" else None
    return HybridRetriever(chunks, embedder, reranker, cache_path=cache)


def make_segmenter(settings: Settings, oracle_masks: Path | None = None):
    if oracle_masks is not None:
        return OracleSegmenter(oracle_masks)
    model = Path(settings.model_path)
    if not model.exists():
        raise FileNotFoundError(
            f"{model} not found. Train it with `bridge-triage train` or use --oracle-masks"
        )
    return OnnxSegmenter(model, settings.input_size)


def make_deps(settings: Settings, oracle_masks: Path | None = None, segmenter=None, provider=None) -> Deps:
    return Deps(
        settings=settings,
        segmenter=segmenter or make_segmenter(settings, oracle_masks),
        retriever=make_retriever(settings),
        provider=provider or get_provider(settings.provider, settings.provider_model),
    )


@contextmanager
def sqlite_checkpointer(path: str):
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(path) as saver:
        yield saver


def run_one(graph, image: Path, site: str, thread_id: str | None = None) -> dict:
    """Runs until the end or until the review interrupt. Returns a short summary."""
    thread_id = thread_id or f"{image.stem}-{site}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()
    result = graph.invoke({"image": str(image), "site": site}, config)
    seconds = round(time.perf_counter() - t0, 3)
    state = graph.get_state(config)
    interrupts = [i.value for i in result.get("__interrupt__", [])]
    return {
        "thread_id": thread_id,
        "paused_for_review": bool(interrupts),
        "review_request": interrupts[0] if interrupts else None,
        "values": state.values,
        "seconds": seconds,
    }


def resume(graph, thread_id: str, review: ReviewDecision) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(Command(resume=review.model_dump()), config)
    return graph.get_state(config).values


def in_memory_graph(deps: Deps):
    return build_graph(deps, checkpointer=InMemorySaver())
