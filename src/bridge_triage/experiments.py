"""Measured runs behind the numbers in the README. Each writes JSON into results/."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .corpus import load_corpus
from .evaluate import confusion, mrr, recall_at_k, segmentation_scores, worst_state_scores
from .pipeline import in_memory_graph, make_deps, run_one
from .retrieval import CrossEncoderReranker, FastEmbedder, HashingEmbedder, HybridRetriever
from .segment import OnnxSegmenter, decode_mask, quantify


def _test_pairs(split_root: Path) -> list[tuple[Path, Path]]:
    imgs = sorted((split_root / "Test" / "images_512").glob("*.jpeg"), key=lambda p: int(p.stem))
    return [(p, split_root / "Test" / "mask_512" / f"{p.stem}.png") for p in imgs]


def vision(settings: Settings, split_root: Path, out: Path) -> dict:
    seg = OnnxSegmenter(settings.model_path, settings.input_size)
    cm = np.zeros((4, 4), dtype=np.int64)
    true_ws, pred_ws, abs_err, per_image, latency = (
        [],
        [],
        {"fair": [], "poor": [], "severe": [], "corroded": []},
        [],
        [],
    )
    for img_path, mask_path in _test_pairs(split_root):
        truth = decode_mask(mask_path)
        img = Image.open(img_path).convert("RGB")
        seg.predict(img)  # warm path, the first call allocates
        t0 = time.perf_counter()
        pred, probs = seg.predict(img)
        latency.append(time.perf_counter() - t0)
        cm += confusion(truth, pred, 4)
        qt = quantify(truth, settings.min_area_pct)
        qp = quantify(pred, settings.min_area_pct, probs)
        true_ws.append(qt.worst_state)
        pred_ws.append(qp.worst_state)
        for k, a, b in [
            ("fair", qt.fair_pct, qp.fair_pct), ("poor", qt.poor_pct, qp.poor_pct),
            ("severe", qt.severe_pct, qp.severe_pct), ("corroded", qt.corroded_pct, qp.corroded_pct),
        ]:  # fmt: skip
            abs_err[k].append(abs(a - b))
        per_image.append({"image": img_path.stem, "true": qt.as_dict(), "pred": qp.as_dict()})
    result = {
        "test_images": len(per_image),
        "input_size": settings.input_size,
        "pixels": segmentation_scores(cm),
        "area_mae_pct_points": {k: round(float(np.mean(v)), 2) for k, v in abs_err.items()},
        "worst_state": worst_state_scores(true_ws, pred_ws),
        "latency_ms": {
            "median": round(1000 * statistics.median(latency), 1),
            "p95": round(1000 * float(np.percentile(latency, 95)), 1),
        },
        "runtime": f"onnxruntime CPU, 2 threads, {platform.processor() or platform.machine()}",
        "per_image": per_image,
    }
    out.write_text(json.dumps(result, indent=2))
    return result


def retrieval(settings: Settings, eval_path: Path, out: Path, modes: list[str]) -> dict:
    chunks = load_corpus(Path(settings.corpus_path))
    rows = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    relevant = [set(r["relevant_pages"]) for r in rows]
    results = {"questions": len(rows), "chunks": len(chunks), "modes": {}}
    fast = FastEmbedder() if any(m != "bm25" for m in modes) else None
    rerank = CrossEncoderReranker() if "hybrid+rerank" in modes else None
    for mode in modes:
        embed = fast
        label = mode
        if mode == "hybrid-hashing":
            embed, mode = HashingEmbedder(), "hybrid"
        cache = Path(settings.vector_cache) if embed is fast and embed is not None else None
        retr = HybridRetriever(chunks, embed if mode != "bm25" else None, rerank, cache_path=cache)
        t0 = time.perf_counter()
        ranked = []
        for r in rows:
            hits = retr.search(r["question"], k=10, mode=mode)
            pages = []
            for h in hits:  # rank pages, a page counts once
                if h.page_key not in pages:
                    pages.append(h.page_key)
            ranked.append(pages)
        ms = 1000 * (time.perf_counter() - t0) / len(rows)
        results["modes"][label] = {
            "recall@1": recall_at_k(ranked, relevant, 1),
            "recall@3": recall_at_k(ranked, relevant, 3),
            "recall@5": recall_at_k(ranked, relevant, 5),
            "mrr@10": mrr(ranked, relevant, 10),
            "ms_per_query": round(ms, 1),
        }
    results["embedder"] = fast.name if fast else None
    results["reranker"] = rerank.name if rerank else None
    out.write_text(json.dumps(results, indent=2))
    return results


def agent(
    settings: Settings, split_root: Path, out: Path, site: str, oracle: bool, provider=None, limit=None
) -> dict:
    mask_dir = split_root / "Test" / "mask_512" if oracle else None
    deps = make_deps(settings, oracle_masks=mask_dir, provider=provider)
    graph = in_memory_graph(deps)
    runs = []
    pairs = _test_pairs(split_root)[:limit] if limit else _test_pairs(split_root)
    for img_path, _ in pairs:
        r = run_one(graph, img_path, site)
        v = r["values"]
        log = v.get("validation_log", [])
        runs.append(
            {
                "image": img_path.stem,
                "seconds": r["seconds"],
                "attempts": v.get("attempts"),
                "first_try_ok": bool(log) and not log[0]["problems"],
                "final_ok": bool(log) and not log[-1]["problems"],
                "fallback": v.get("note_source") == "template-fallback",
                "paused_for_review": r["paused_for_review"],
                "worst_state": v["decision"]["worst_state"],
                "priority": v["decision"]["priority"],
                "problems": [p for entry in log for p in entry["problems"]],
                "note": v.get("note"),
            }
        )
    n = len(runs)
    problem_counts: dict[str, int] = {}
    for r in runs:
        for p in r["problems"]:
            key = p.split(":")[0]
            problem_counts[key] = problem_counts.get(key, 0) + 1
    summary = {
        "provider": deps.provider.name,
        "segmenter": "annotated masks" if oracle else "onnx model",
        "site": site,
        "photos": n,
        "first_try_pass_rate": round(sum(r["first_try_ok"] for r in runs) / n, 4),
        "pass_rate_after_retries": round(sum(r["final_ok"] for r in runs) / n, 4),
        "fallback_rate": round(sum(r["fallback"] for r in runs) / n, 4),
        "review_rate": round(sum(r["paused_for_review"] for r in runs) / n, 4),
        "mean_attempts": round(sum(r["attempts"] for r in runs) / n, 2),
        "median_seconds_per_photo": round(statistics.median(r["seconds"] for r in runs), 3),
        "problem_counts": dict(sorted(problem_counts.items(), key=lambda kv: -kv[1])),
        "runs": runs,
    }
    out.write_text(json.dumps(summary, indent=2))
    return summary
