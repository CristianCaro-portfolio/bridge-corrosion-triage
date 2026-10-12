"""Command line entry point: bridge-triage --help"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import typer

from .config import load_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Steel bridge corrosion triage")

CONFIG = typer.Option("config.yaml", "--config", "-c", help="settings file")


def _print(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def fetch(
    config: Path = CONFIG,
    raw_dir: Path = typer.Option(Path("data/raw"), help="where downloads go"),
    year: int = typer.Option(2024, help="weather year"),
    weather: bool = typer.Option(False, help="also refresh the NOAA files of the configured sites"),
):
    """Download the corrosion dataset and FHWA PDFs, rebuild the guidance corpus."""
    from .corpus import build_corpus
    from .data import fetch_corrosion, fetch_fhwa, fetch_weather, sha256

    s = load_settings(config)
    root = fetch_corrosion(raw_dir)
    typer.echo(f"corrosion dataset ready at {root}")
    docs = fetch_fhwa(raw_dir)
    chunks = build_corpus(docs, Path(s.corpus_path))
    typer.echo(f"{len(chunks)} guidance chunks written to {s.corpus_path}")
    manifest = {doc: {"file": str(p), "sha256": sha256(p)} for doc, p in docs.items()}
    if weather:
        for name, site in s.sites.items():
            dest = Path(site.weather_file)
            fetch_weather(site.station, year, dest)
            manifest[f"weather:{name}"] = {"file": str(dest), "sha256": sha256(dest)}
            typer.echo(f"weather for {name} written to {dest}")
    Path("results").mkdir(exist_ok=True)
    Path("results/data_manifest.json").write_text(json.dumps(manifest, indent=2))


@app.command()
def train(
    data_dir: Path = typer.Option(Path("data/raw/vt_corrosion/512x512")),
    out_dir: Path = typer.Option(Path("data/model")),
    size: int = 320,
    epochs: int = 30,
    pretrained: bool = typer.Option(True, help="start from the COCO pretrained LR-ASPP weights"),
    limit: int = typer.Option(0, help="use only N training photos (smoke runs)"),
):
    """Fine-tune the segmentation model and export it to ONNX (needs the train extra)."""
    from .train import train as run

    info = run(data_dir, out_dir, size=size, epochs=epochs, pretrained=pretrained, limit=limit or None)
    _print({k: v for k, v in info.items() if k != "history"})


@app.command()
def inspect(
    image: Path,
    site: str = typer.Option(..., help="site name from config.yaml"),
    config: Path = CONFIG,
    oracle_masks: Path = typer.Option(None, help="use annotated masks from this folder instead of the model"),
):
    """Run one photo through the workflow. Pauses for review when the policy asks for it."""
    from .agent import build_graph
    from .pipeline import make_deps, run_one, sqlite_checkpointer

    s = load_settings(config)
    deps = make_deps(s, oracle_masks=oracle_masks)
    with sqlite_checkpointer(s.checkpoint_db) as saver:
        r = run_one(build_graph(deps, checkpointer=saver), image, site)
    v = r["values"]
    _print(
        {
            "thread_id": r["thread_id"],
            "status": v.get("status", "waiting_for_review"),
            "quantities": v["quantities"],
            "exposure": {k: v["exposure"][k] for k in ("exposure_class", "tow_hours", "year")},
            "decision": v["decision"],
            "note": v["note"],
            "review_reasons": v["review_reasons"],
            "attempts": v["attempts"],
        }
    )
    if r["paused_for_review"]:
        typer.echo(
            "\nWaiting for review. Resume with: "
            f"bridge-triage review {r['thread_id']} --decision approve --reviewer <name>"
        )


@app.command()
def review(
    thread_id: str,
    decision: str = typer.Option(..., help="approve, edit or reject"),
    reviewer: str = typer.Option(...),
    comment: str = "",
    note: str = typer.Option(None, help="replacement note for edit"),
    priority: str = typer.Option(None, help="override the priority, P1 to P4"),
    config: Path = CONFIG,
):
    """Answer a paused review. Works from a new process thanks to the SQLite checkpoint."""
    from pydantic import ValidationError

    from .agent import ReviewDecision, build_graph
    from .pipeline import resume, sqlite_checkpointer
    from .segment import OracleSegmenter

    s = load_settings(config)
    try:
        answer = ReviewDecision(
            decision=decision, reviewer=reviewer, comment=comment, note=note, priority=priority
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    class _Unused:
        name = "unused"

    # Resuming starts at the review node, so no photo, model or retriever is needed.
    from .agent import Deps

    deps = Deps(settings=s, segmenter=OracleSegmenter("."), retriever=None, provider=_Unused())
    with sqlite_checkpointer(s.checkpoint_db) as saver:
        graph = build_graph(deps, checkpointer=saver)
        config_ = {"configurable": {"thread_id": thread_id}}
        if not graph.get_state(config_).next:
            raise typer.BadParameter(f"thread {thread_id} is not waiting for a review")
        values = resume(graph, thread_id, answer)
    _print(
        {
            "thread_id": thread_id,
            "status": values["status"],
            "priority": values["decision"]["priority"],
            "note": values["note"],
        }
    )


@app.command()
def findings(config: Path = CONFIG):
    """List the stored findings."""
    from .store import read_findings

    for row in read_findings(load_settings(config).findings_db):
        typer.echo(
            f"{row['created_at']:%Y-%m-%d %H:%M}  {row['image']:>12}  {row['site']:<16} "
            f"CS{row['worst_state']} "
            f"{row['priority']} {row['due_days']:>4}d  {row['status']:<15} {row['provider']}"
        )


@app.command()
def demo(config: Path = CONFIG):
    """Offline demo on the bundled sample: annotated masks, BM25, template notes."""
    from .agent import ReviewDecision, build_graph
    from .pipeline import make_deps, resume, run_one, sqlite_checkpointer
    from .store import read_findings

    s = load_settings(config)
    s.retrieval_mode, s.provider = "bm25", "template"
    tmp = Path(tempfile.mkdtemp(prefix="triage-demo-"))
    s.checkpoint_db, s.findings_db = str(tmp / "ckpt.sqlite"), str(tmp / "findings.duckdb")
    deps = make_deps(s, oracle_masks=Path("sample_data/masks"))
    images = sorted(Path("sample_data/images").glob("*.jpeg"), key=lambda p: int(p.stem))
    with sqlite_checkpointer(s.checkpoint_db) as saver:
        graph = build_graph(deps, checkpointer=saver)
        for i, img in enumerate(images):
            site = list(s.sites)[i % len(s.sites)]
            r = run_one(graph, img, site)
            v = r["values"]
            typer.echo(
                f"{img.name:>10} {site:<16} CS{v['decision']['worst_state']} {v['decision']['priority']} "
                f"{v['decision']['due_days']:>4}d  corroded {v['quantities']['corroded_pct']:>5}%  "
                f"{'REVIEW' if r['paused_for_review'] else 'auto'}"
            )
            if r["paused_for_review"]:
                resume(graph, r["thread_id"], ReviewDecision(decision="approve", reviewer="demo"))
    rows = read_findings(s.findings_db)
    typer.echo(f"\n{len(rows)} findings stored, {sum(r['status'] == 'approved' for r in rows)} after review")
    typer.echo(f"example note: {rows[-1]['note']}")
    shutil.rmtree(tmp, ignore_errors=True)
    if len(rows) != len(images):
        raise typer.Exit(1)


@app.command()
def evaluate(
    what: str = typer.Argument(..., help="vision, retrieval or agent"),
    config: Path = CONFIG,
    split_root: Path = typer.Option(Path("data/raw/vt_corrosion/512x512")),
    site: str = "richmond-inland",
    oracle: bool = typer.Option(True, help="agent eval on annotated masks to isolate the language model"),
    provider: str = typer.Option(None, help="template, hf or openai"),
    model: str = typer.Option(None, help="model id for the hf provider"),
    out: Path = typer.Option(None),
):
    """Measured runs. Results go to results/*.json."""
    from . import experiments

    s = load_settings(config)
    Path("results").mkdir(exist_ok=True)
    if what == "vision":
        r = experiments.vision(s, split_root, out or Path("results/vision.json"))
        _print({k: v for k, v in r.items() if k != "per_image"})
    elif what == "retrieval":
        modes = ["bm25", "dense", "hybrid", "hybrid+rerank", "hybrid-hashing"]
        _print(
            experiments.retrieval(
                s, Path("sample_data/retrieval_eval.jsonl"), out or Path("results/retrieval.json"), modes
            )
        )
    elif what == "agent":
        from .llm import get_provider

        p = get_provider(provider or s.provider, model or s.provider_model)
        tag = p.name.split("/")[-1].replace(":", "-")
        r = experiments.agent(
            s, split_root, out or Path(f"results/agent_{tag}.json"), site, oracle, provider=p
        )
        _print({k: v for k, v in r.items() if k != "runs"})
    else:
        raise typer.BadParameter("choose vision, retrieval or agent")


@app.command()
def report():
    """Rebuild the figures in docs/ from results/."""
    from .report import build

    for path in build(Path("results"), Path("docs")):
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
