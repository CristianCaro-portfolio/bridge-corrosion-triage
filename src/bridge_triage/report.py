"""Figures for the README, rebuilt from results/ by `bridge-triage report`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK = "#1f2933"
MUTED = "#61707d"
BLUE = "#2f6db5"
ORANGE = "#d9822b"
GREEN = "#3f8f5a"
GREY = "#e6eaee"


def _box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=color, lw=1.8
        )
    )
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, color=INK, weight="bold")
    ax.text(
        x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8.2, color=MUTED, linespacing=1.3
    )


def _arrow(ax, x0, y0, x1, y1, color=MUTED, text=None):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=7.8, color=color)


def architecture(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.6, 5.6), dpi=150)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0.8, 5.6)
    ax.axis("off")
    w, h = 1.9, 1.05
    top = 3.9
    _box(ax, 0.2, top, w, h, "Photo", "inspection photo\nof a steel member", INK)
    _box(ax, 2.6, top, w, h, "Segment", "LR-ASPP MobileNetV3\nONNX Runtime, CPU", BLUE)
    _box(ax, 5.0, top, w, h, "Quantify", "% area in CS2, CS3, CS4\nworst state, confidence", BLUE)
    _box(ax, 7.4, top, w, h, "Exposure", "NOAA hourly 2024\nwet hours, freezing rain", BLUE)
    _box(ax, 9.8, top, w, h, "Policy", "priority and due days\nplain code, configurable", BLUE)
    for x in (2.1, 4.5, 6.9, 9.3):
        _arrow(ax, x, top + h / 2, x + 0.5, top + h / 2)

    mid = 2.1
    _box(ax, 9.8, mid, w, h, "Retrieve", "BM25 + bge-small\nfused, then reranked", ORANGE)
    _box(ax, 7.4, mid, w, h, "Draft note", "local Qwen2.5 or template\nor OpenAI compatible", ORANGE)
    _box(ax, 5.0, mid, w, h, "Check", "citations, numbers,\nstate, priority, due days", ORANGE)
    _box(ax, 2.6, mid, w, h, "Engineer review", "LangGraph interrupt\ntyped answer, resumable", GREEN)
    _box(ax, 0.2, mid, w, h, "Findings", "DuckDB table\nwith audit trail", GREEN)
    _arrow(ax, 10.75, top, 10.75, mid + h)
    _arrow(ax, 9.8, mid + h / 2, 9.3, mid + h / 2)
    _arrow(ax, 7.4, mid + h / 2, 6.9, mid + h / 2)
    _arrow(ax, 5.0, mid + h / 2, 4.5, mid + h / 2, GREEN)
    ax.text(
        3.55,
        mid + h + 0.12,
        "CS3, CS4, low confidence\nor a note that kept failing",
        ha="center",
        fontsize=7.8,
        color=GREEN,
    )
    _arrow(ax, 2.6, mid + h / 2, 2.1, mid + h / 2)
    ax.annotate(
        "",
        (8.35, mid),
        (5.95, mid),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2, connectionstyle="arc3,rad=0.45"),
    )
    ax.text(
        7.15,
        mid - 0.68,
        "retry with the problems, max 3, then template + review",
        ha="center",
        fontsize=7.8,
        color=ORANGE,
    )
    ax.annotate(
        "",
        (1.15, mid),
        (5.95, mid),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2, connectionstyle="arc3,rad=-0.32"),
    )
    ax.text(
        3.55,
        1.08,
        "CS1 and CS2 with good confidence publish directly",
        ha="center",
        fontsize=7.8,
        color=GREEN,
    )
    ax.text(
        0.2,
        5.35,
        "bridge-corrosion-triage: every decision is code, the language model only writes the note",
        fontsize=12,
        color=INK,
        weight="bold",
    )
    ax.text(
        12.55,
        2.62,
        "LangGraph state\ncheckpointed in\nSQLite at every\nstep",
        fontsize=8,
        color=MUTED,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", fc=GREY, ec="none"),
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def results_figure(results: Path, path: Path, split_root: Path | None = None) -> Path | None:
    vision_p, retr_p = results / "vision.json", results / "retrieval.json"
    agents = sorted(p for p in results.glob("agent_*.json") if "prompt-v" not in p.name)
    if not (vision_p.exists() and retr_p.exists() and agents):
        return None
    vision = json.loads(vision_p.read_text())
    retr = json.loads(retr_p.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)

    ax = axes[0]
    classes = ["fair", "poor", "severe"]
    ious = [vision["pixels"]["iou"][c] for c in classes]
    f1s = [vision["pixels"]["f1"][c] for c in classes]
    xs = range(len(classes))
    ax.bar([x - 0.2 for x in xs], ious, 0.4, color=BLUE, label="IoU")
    ax.bar([x + 0.2 for x in xs], f1s, 0.4, color="#9ec1e8", label="F1")
    for x, v in zip(xs, ious):
        ax.text(x - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for x, v in zip(xs, f1s):
        ax.text(x + 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs), ["CS2 fair", "CS3 poor", "CS4 severe"])
    ax.set_ylim(0, 1)
    ws = vision["worst_state"]
    ax.set_title(
        f"Segmentation on {vision['test_images']} held out photos\n"
        f"escalation recall {ws['escalation_recall']:.2f}, worst state exact {ws['exact']:.2f}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    modes = list(retr["modes"])
    r1 = [retr["modes"][m]["recall@1"] for m in modes]
    r3 = [retr["modes"][m]["recall@3"] for m in modes]
    xs = range(len(modes))
    ax.bar([x - 0.2 for x in xs], r1, 0.4, color=ORANGE, label="recall@1")
    ax.bar([x + 0.2 for x in xs], r3, 0.4, color="#f2c08f", label="recall@3")
    for x, v in zip(xs, r1):
        ax.text(x - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs), [m.replace("+", "+\n").replace("-", "-\n") for m in modes], fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_title(f"Retrieval, {retr['questions']} questions over {retr['chunks']} FHWA chunks", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    ax = axes[2]
    rows = sorted((json.loads(p.read_text()) for p in agents), key=lambda r: -r["photos"])
    names = [r["provider"].replace("hf:Qwen/", "").replace("-Instruct", "") for r in rows]
    labels = [f"{n}\n{r['photos']} photos" for n, r in zip(names, rows)]
    first = [r["first_try_pass_rate"] for r in rows]
    final = [r["pass_rate_after_retries"] for r in rows]
    xs = range(len(rows))
    ax.bar([x - 0.2 for x in xs], first, 0.4, color=GREEN, label="passes checks first try")
    ax.bar([x + 0.2 for x in xs], final, 0.4, color="#a7d3b4", label="after up to 3 tries")
    for x, v in zip(xs, first):
        ax.text(x - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for x, v in zip(xs, final):
        ax.text(x + 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs), labels, fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_title("Notes that pass the code checks", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def examples_figure(
    results: Path, split_root: Path, model_path: Path, size: int, path: Path, names: list[str]
) -> Path:
    import numpy as np
    from PIL import Image

    from .segment import OnnxSegmenter, colorize, decode_mask, quantify

    seg = OnnxSegmenter(model_path, size)
    fig, axes = plt.subplots(3, len(names), figsize=(3.0 * len(names), 9), dpi=130)
    for j, name in enumerate(names):
        img = Image.open(split_root / "Test" / "images_512" / f"{name}.jpeg").convert("RGB")
        truth = decode_mask(split_root / "Test" / "mask_512" / f"{name}.png")
        pred, probs = seg.predict(img)
        qt, qp = quantify(truth, 0.5), quantify(pred, 0.5, probs)
        base = np.asarray(img, dtype=float)
        for i, (title, m) in enumerate(
            [("photo", None), (f"annotated CS{qt.worst_state}", truth), (f"model CS{qp.worst_state}", pred)]
        ):
            ax = axes[i, j]
            if m is None:
                ax.imshow(img)
            else:
                over = base.copy()
                hit = m > 0
                over[hit] = 0.45 * base[hit] + 0.55 * colorize(m)[hit]
                ax.imshow(over.astype("uint8"))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Held out test photos. Red CS2 fair, green CS3 poor, yellow CS4 severe", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # 256 colour palette keeps the file under 1 MB in the repo
    rgb = Image.open(path).convert("RGB")
    small = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    small.save(path, optimize=True)
    return path


def build(
    results: Path,
    docs: Path,
    split_root: Path = Path("data/raw/vt_corrosion/512x512"),
    model_path: Path = Path("data/model/corrosion_seg.onnx"),
    size: int = 320,
) -> list[Path]:
    docs.mkdir(exist_ok=True)
    out = [architecture(docs / "architecture.png")]
    r = results_figure(results, docs / "results.png")
    if r:
        out.append(r)
    if model_path.exists() and (split_root / "Test").exists():
        # one photo per worst state, including the CS4 photo the model calls CS3
        names = ["21", "18", "24", "28", "17"]
        out.append(examples_figure(results, split_root, model_path, size, docs / "examples.png", names))
    return out
