import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from bridge_triage.agent import Deps, ReviewDecision, build_graph
from bridge_triage.llm import Generation, TemplateProvider
from bridge_triage.pipeline import make_retriever, resume, run_one, sqlite_checkpointer
from bridge_triage.segment import OracleSegmenter
from bridge_triage.store import read_findings


class InventsNumbers:
    name = "liar"

    def __init__(self):
        self.calls = 0

    def generate(self, facts, excerpts, feedback=None):
        self.calls += 1
        good = TemplateProvider().generate(facts, excerpts).text
        return Generation(good.replace(str(facts["corroded_pct"]), "99.9"), self.name, 0.0)


class Broken:
    name = "broken"

    def generate(self, facts, excerpts, feedback=None):
        raise TimeoutError("endpoint did not answer")


@pytest.fixture
def make(settings, root):
    def _make(provider=None):
        deps = Deps(
            settings=settings,
            segmenter=OracleSegmenter(root / "sample_data/masks"),
            retriever=make_retriever(settings),
            provider=provider or TemplateProvider(),
        )
        return build_graph(deps, checkpointer=InMemorySaver())

    return _make


def test_clean_photo_is_published_without_review(make, root, settings):
    r = run_one(make(), root / "sample_data/images/21.jpeg", "richmond-inland")
    assert not r["paused_for_review"]
    assert r["values"]["status"] == "auto_published"
    rows = read_findings(settings.findings_db)
    assert rows[0]["worst_state"] == 1 and rows[0]["priority"] == "P4"
    assert rows[0]["citations"]


def test_poor_photo_waits_for_an_engineer_then_publishes(make, root, settings):
    graph = make()
    r = run_one(graph, root / "sample_data/images/24.jpeg", "roanoke-valley")
    assert r["paused_for_review"] and read_findings(settings.findings_db) == []
    assert "CS3 always goes to an engineer" in r["review_request"]["reasons"]
    values = resume(
        graph,
        r["thread_id"],
        ReviewDecision(decision="edit", reviewer="ana", note="Checked on site.", priority="P1"),
    )
    assert values["status"] == "edited" and values["note"] == "Checked on site."
    row = read_findings(settings.findings_db)[0]
    assert row["reviewer"] == "ana" and row["priority"] == "P1"


def test_review_answer_is_validated(make, root):
    graph = make()
    r = run_one(graph, root / "sample_data/images/2.jpeg", "richmond-inland")
    with pytest.raises(ValidationError):
        graph.invoke(
            __import__("langgraph.types", fromlist=["Command"]).Command(
                resume={"decision": "maybe", "reviewer": "x"}
            ),
            {"configurable": {"thread_id": r["thread_id"]}},
        )


def test_bad_notes_are_retried_then_replaced(make, root, settings):
    liar = InventsNumbers()
    r = run_one(make(liar), root / "sample_data/images/18.jpeg", "richmond-inland")
    v = r["values"]
    assert liar.calls == settings.max_attempts
    assert v["note_source"] == "template-fallback"
    assert r["paused_for_review"]  # a CS2 photo would normally auto publish
    assert all(entry["problems"] for entry in v["validation_log"])


def test_provider_outage_does_not_lose_the_photo(make, root):
    r = run_one(make(Broken()), root / "sample_data/images/18.jpeg", "richmond-inland")
    assert r["values"]["note_source"] == "template-fallback" and r["paused_for_review"]


def test_unreadable_photo_fails_closed(make, tmp_path):
    bad = tmp_path / "99.jpeg"
    bad.write_text("not an image")
    r = run_one(make(), bad, "richmond-inland")
    assert r["values"]["decision"]["worst_state"] == 4
    assert any("could not read image" in x for x in r["values"]["review_reasons"])


def test_unknown_site_assumes_severe_exposure(make, root):
    r = run_one(make(), root / "sample_data/images/18.jpeg", "nowhere")
    v = r["values"]
    assert v["exposure"]["exposure_class"] == "severe" and r["paused_for_review"]


def test_review_resumes_from_another_process(settings, root):
    """The SQLite checkpoint is what lets an engineer answer later."""

    def graph(saver):
        deps = Deps(
            settings,
            OracleSegmenter(root / "sample_data/masks"),
            make_retriever(settings),
            TemplateProvider(),
        )
        return build_graph(deps, checkpointer=saver)

    with sqlite_checkpointer(settings.checkpoint_db) as saver:
        r = run_one(graph(saver), root / "sample_data/images/17.jpeg", "norfolk-harbor")
    assert r["paused_for_review"]
    with sqlite_checkpointer(settings.checkpoint_db) as saver:  # new connection, new graph
        values = resume(graph(saver), r["thread_id"], ReviewDecision(decision="approve", reviewer="luis"))
    assert values["status"] == "approved"
    assert values["decision"]["due_days"] == 14  # CS4 at a marine site
