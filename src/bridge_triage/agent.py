"""LangGraph workflow: photo -> condition states -> exposure -> policy -> guidance -> note ->
checks -> engineer review when needed -> findings table.

Every step that decides something is plain code. The language model only writes the note and
its output is checked before it can be stored. State is checkpointed in SQLite, so a review
can be answered hours later from another process and the run resumes where it stopped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from . import guardrail, store
from .config import Settings
from .exposure import Exposure, ExposureDataError, from_lcd_csv
from .llm import Provider, TemplateProvider
from .policy import decide
from .retrieval import HybridRetriever
from .segment import quantify

STATE_LABELS = {1: "CS1", 2: "CS2 (fair)", 3: "CS3 (poor)", 4: "CS4 (severe)"}
STATE_WORDS = {
    1: "no corrosion",
    2: "surface rust freckled corrosion",
    3: "pack rust section loss",
    4: "perforation severe section loss",
}


class ReviewDecision(BaseModel):
    """What the engineer sends back to resume a paused run."""

    decision: Literal["approve", "edit", "reject"]
    reviewer: str = Field(min_length=1)
    comment: str = ""
    note: str | None = Field(None, description="replacement note when decision is edit")
    priority: Literal["P1", "P2", "P3", "P4"] | None = None


class TriageState(TypedDict, total=False):
    image: str
    site: str
    quantities: dict
    exposure: dict
    decision: dict
    review_reasons: list[str]
    excerpts: list[dict]
    note: str
    note_source: str
    attempts: int
    problems: list[str]
    validation_log: list[dict]
    review: dict
    status: str
    error: str


@dataclass
class Deps:
    settings: Settings
    segmenter: Any  # OnnxSegmenter or OracleSegmenter
    retriever: HybridRetriever
    provider: Provider
    weather_root: Path = Path(".")
    _exposure_cache: dict[str, Exposure] = field(default_factory=dict)

    def exposure_for(self, site: str) -> Exposure:
        if site not in self._exposure_cache:
            cfg = self.settings.sites[site]
            self._exposure_cache[site] = from_lcd_csv(self.weather_root / cfg.weather_file, marine=cfg.marine)
        return self._exposure_cache[site]


def _facts(state: TriageState) -> dict:
    q, e, d = state["quantities"], state["exposure"], state["decision"]
    return {
        "worst_state": d["worst_state"],
        "worst_state_label": STATE_LABELS[d["worst_state"]],
        "fair_pct": q["fair_pct"],
        "poor_pct": q["poor_pct"],
        "severe_pct": q["severe_pct"],
        "corroded_pct": q["corroded_pct"],
        "exposure_class": e["exposure_class"],
        "tow_hours": e["tow_hours"],
        "year": e["year"],
        "priority": d["priority"],
        "action": d["action"],
        "due_days": d["due_days"],
    }


def build_graph(deps: Deps, checkpointer=None):
    s = deps.settings

    def assess(state: TriageState) -> dict:
        path = Path(state["image"])
        try:
            if hasattr(deps.segmenter, "predict_path"):
                mask, probs = deps.segmenter.predict_path(path)
            else:
                mask, probs = deps.segmenter.predict(Image.open(path))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            # fail closed: an unreadable photo is treated as the worst case and goes to a person
            return {
                "quantities": {
                    "fair_pct": 0.0,
                    "poor_pct": 0.0,
                    "severe_pct": 0.0,
                    "corroded_pct": 0.0,
                    "worst_state": 4,
                    "mean_confidence": 0.0,
                },
                "error": f"could not read image: {exc}",
            }
        return {"quantities": quantify(mask, s.min_area_pct, probs).as_dict()}

    def exposure(state: TriageState) -> dict:
        try:
            return {"exposure": deps.exposure_for(state["site"]).as_dict()}
        except (KeyError, ExposureDataError, FileNotFoundError) as exc:
            return {
                "exposure": {"exposure_class": "severe", "tow_hours": 0, "year": 0, "station": "unknown"},
                "error": f"exposure unavailable, assumed severe: {exc}",
            }

    def policy(state: TriageState) -> dict:
        q = state["quantities"]
        d = decide(q["worst_state"], state["exposure"]["exposure_class"], q["mean_confidence"], s.policy)
        reasons = list(d.review_reasons)
        if state.get("error"):
            reasons.append(state["error"])
        return {"decision": d.as_dict(), "review_reasons": reasons, "attempts": 0, "validation_log": []}

    def retrieve(state: TriageState) -> dict:
        d, e = state["decision"], state["exposure"]
        query = (
            f"steel bridge girder corrosion {STATE_WORDS[d['worst_state']]} "
            f"{'deicing salt' if e['exposure_class'] != 'mild' else ''} time of wetness "
            f"maintenance {d['action']}"
        )
        hits = deps.retriever.search(query, k=s.top_k, mode=s.retrieval_mode)
        return {
            "excerpts": [
                {"cite": h.page_key, "chunk_id": h.chunk.chunk_id, "text": h.chunk.text} for h in hits
            ]
        }

    def draft(state: TriageState) -> dict:
        attempts = state.get("attempts", 0) + 1
        try:
            gen = deps.provider.generate(_facts(state), state["excerpts"], state.get("problems") or None)
            return {"note": gen.text, "note_source": gen.model, "attempts": attempts}
        except Exception as exc:  # provider down or timed out: count the attempt, checks will fail it
            return {"note": "", "note_source": f"error: {type(exc).__name__}", "attempts": attempts}

    def validate(state: TriageState) -> dict:
        allowed = {e["cite"] for e in state["excerpts"]}
        problems = guardrail.check(state["note"], _facts(state), allowed)
        log = list(state.get("validation_log", []))
        log.append({"attempt": state["attempts"], "source": state["note_source"], "problems": problems})
        return {"problems": problems, "validation_log": log}

    def after_validate(state: TriageState) -> str:
        if not state["problems"]:
            return "route"
        return "draft" if state["attempts"] < s.max_attempts else "fallback"

    def fallback(state: TriageState) -> dict:
        gen = TemplateProvider().generate(_facts(state), state["excerpts"])
        reasons = list(state["review_reasons"]) + [f"generated note failed checks {state['attempts']} times"]
        return {"note": gen.text, "note_source": "template-fallback", "review_reasons": reasons}

    def route(state: TriageState) -> dict:
        return {}

    def needs_review(state: TriageState) -> str:
        return "human_review" if state["review_reasons"] else "publish"

    def human_review(state: TriageState) -> dict:
        payload = {
            "image": state["image"],
            "site": state["site"],
            "quantities": state["quantities"],
            "decision": state["decision"],
            "note": state["note"],
            "reasons": state["review_reasons"],
            "excerpts": [e["cite"] for e in state["excerpts"]],
        }
        answer = interrupt(payload, response_schema=ReviewDecision)
        review = ReviewDecision.model_validate(answer)
        update: dict = {"review": review.model_dump()}
        if review.decision == "edit" and review.note:
            update["note"] = review.note
            update["note_source"] = f"edited by {review.reviewer}"
        if review.priority:
            decision = dict(state["decision"])
            decision["priority"] = review.priority
            update["decision"] = decision
        return update

    def publish(state: TriageState, config: RunnableConfig) -> dict:
        review = state.get("review")
        if review is None:
            status = "auto_published"
        else:
            status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[review["decision"]]
        q, e, d = state["quantities"], state["exposure"], state["decision"]
        row = {
            "thread_id": config["configurable"]["thread_id"],
            "image": Path(state["image"]).name,
            "site": state["site"],
            "worst_state": d["worst_state"],
            "fair_pct": q["fair_pct"],
            "poor_pct": q["poor_pct"],
            "severe_pct": q["severe_pct"],
            "confidence": q["mean_confidence"],
            "exposure_class": e["exposure_class"],
            "tow_hours": e["tow_hours"],
            "priority": d["priority"],
            "action": d["action"],
            "due_days": d["due_days"],
            "note": state["note"],
            "citations": sorted(set(guardrail.CITE.findall(state["note"]))),
            "status": status,
            "reviewer": review["reviewer"] if review else None,
            "review_comment": review["comment"] if review else None,
            "provider": state["note_source"],
            "attempts": state["attempts"],
            "validation_log": state["validation_log"],
        }
        # upsert keyed by thread id, so a replayed publish step never writes a second row
        store.write_finding(s.findings_db, row)
        return {"status": status}

    g = StateGraph(TriageState)
    for name, fn in [
        ("assess", assess), ("exposure", exposure), ("policy", policy), ("retrieve", retrieve),
        ("draft", draft), ("validate", validate), ("fallback", fallback), ("route", route),
        ("human_review", human_review), ("publish", publish),
    ]:  # fmt: skip
        g.add_node(name, fn)
    g.add_edge(START, "assess")
    g.add_edge("assess", "exposure")
    g.add_edge("exposure", "policy")
    g.add_edge("policy", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "validate")
    g.add_conditional_edges("validate", after_validate, ["route", "draft", "fallback"])
    g.add_edge("fallback", "route")
    g.add_conditional_edges("route", needs_review, ["human_review", "publish"])
    g.add_edge("human_review", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
