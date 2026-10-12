# ADR 0003: Hybrid retrieval with a cross-encoder reranker over FHWA guidance

## Context

Each note cites guidance so a reviewer can check why an action is reasonable. The corpus is
small and public domain: three FHWA documents (the Steel Bridge Design Handbook volume on
corrosion protection, the Bridge Preservation Guide 2018 and the 2022 report on preservation
for locally owned bridges), split into page anchored chunks so a citation is a page.

Engineering text mixes exact terms (pack rust, splash zone, zinc-rich primer, time of
wetness) with paraphrase. The question was whether dense vectors help on top of BM25, and
whether a reranker is worth its latency on a CPU.

## Decision

- Keep BM25 (rank-bm25) and dense vectors (BAAI/bge-small-en-v1.5 through fastembed, ONNX
  on CPU) side by side and fuse them with reciprocal rank fusion (k = 60).
- Rerank the top 30 fused chunks with a cross-encoder (ms-marco-MiniLM-L-6-v2, also through
  fastembed). This is the default in config.yaml.
- Keep the vectors in a NumPy file keyed by a hash of the corpus and model. A few hundred
  chunks do not need a vector database.
- Offer a hashing embedder so tests and CI run the hybrid path with no download.
- Measure every mode on a fixed set of 22 questions with page level relevance, written by
  hand from the documents, and keep the numbers in results/retrieval.json.

## What the measurement said

BM25 alone beat dense alone and plain fusion on recall@1 (0.86 against 0.73 and 0.77),
which is expected when the questions use the same vocabulary as the handbook. The reranker
was the only step that improved on BM25 (0.95 recall@1, 1.00 recall@3), and it costs about
1.2 seconds per query on two CPU threads against half a millisecond for BM25. In this workflow it runs
once per photo, next to a language model call, so the cost is acceptable. In an interactive
search box it would not be.

## Alternatives considered

- Dense only: lower recall@1 here and harder to explain when it misses an exact term.
- A vector database (Qdrant, LanceDB, pgvector): the right move at thousands of documents or
  with metadata filters. At 300 chunks it adds a service and no measurable benefit.
- GraphRAG over the documents: the guidance is not entity heavy, and the questions are about
  single passages, so a graph would add build cost without a clear gain.

## Consequences

- The evaluation set is small and written by the same person who wrote the code. It catches
  regressions and compares modes, it does not prove general retrieval quality.
- Adding documents means rebuilding the corpus (`bridge-triage fetch`), the vector cache is
  invalidated automatically by its hash.
