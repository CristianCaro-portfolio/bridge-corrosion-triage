import pytest

from bridge_triage.corpus import chunk_pages, load_corpus
from bridge_triage.retrieval import HashingEmbedder, HybridRetriever, rrf


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == fused[2] > fused[3] > 0
    assert fused[4] == pytest.approx(1 / 63)


def test_chunks_stay_inside_a_page_and_skip_contents():
    toc = "Contents " + " ".join(f"Item {i} ........ {i}" for i in range(10))
    body = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_pages("DOC", [toc, body, "short"])
    assert {c.page for c in chunks} == {2}
    assert chunks[0].chunk_id == "DOC:2:0"
    assert all(len(c.text.split()) <= 160 for c in chunks)


@pytest.fixture(scope="module")
def corpus(request):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return load_corpus(root / "sample_data/fhwa_chunks.jsonl")


def test_bm25_finds_the_splash_zone_page(corpus):
    r = HybridRetriever(corpus)
    hits = r.search("splash zones on through trusses from traffic spray", k=3, mode="bm25")
    assert "FHWA-HIF-16-002:50" in {h.page_key for h in hits}


def test_hybrid_with_offline_embedder(corpus):
    r = HybridRetriever(corpus, embedder=HashingEmbedder())
    hits = r.search("cyclical and condition-based preventive maintenance", k=5, mode="hybrid")
    assert len(hits) == 5
    assert any(h.chunk.doc == "FHWA-BPG-2018" for h in hits)
    assert all(set(h.sources) <= {"bm25", "dense"} for h in hits)


def test_modes_without_their_models_fail_loudly(corpus):
    r = HybridRetriever(corpus)
    with pytest.raises(ValueError):
        r.search("x", mode="dense")
    with pytest.raises(ValueError):
        r.search("x", mode="hybrid+rerank")
