import pytest

from bridge_triage import guardrail
from bridge_triage.llm import TemplateProvider
from bridge_triage.policy import DEFAULT_POLICY, decide

FACTS = {
    "worst_state": 3, "worst_state_label": "CS3 (poor)", "fair_pct": 14.37, "poor_pct": 23.36,
    "severe_pct": 0.0, "corroded_pct": 37.73, "exposure_class": "moderate", "tow_hours": 2912,
    "year": 2024, "priority": "P2", "action": "Measure remaining thickness", "due_days": 120,
}  # fmt: skip
EXCERPTS = [{"cite": "FHWA-HIF-16-002:53", "text": "bottom flanges"}]
ALLOWED = {"FHWA-HIF-16-002:53"}


def test_template_note_passes():
    note = TemplateProvider().generate(FACTS, EXCERPTS).text
    assert guardrail.check(note, FACTS, ALLOWED) == []


@pytest.mark.parametrize(
    "edit,expected",
    [
        (("37.73%", "41.5%"), "numbers not in the facts"),
        (("[FHWA-HIF-16-002:53]", "[FHWA-HIF-16-002:99]"), "not retrieved"),
        (("CS3", "CS4"), "worse than"),
        (("P2", "soon"), "priority code"),
        (("120 days", "a few months"), "due time"),
    ],
)
def test_guardrail_catches_errors(edit, expected):
    note = TemplateProvider().generate(FACTS, EXCERPTS).text.replace(*edit)
    problems = guardrail.check(note, FACTS, ALLOWED)
    assert any(expected in p for p in problems), problems


def test_guardrail_rejects_empty_and_uncited():
    assert guardrail.check("", FACTS, ALLOWED) == ["note is empty or too short"]
    note = TemplateProvider().generate(FACTS, []).text
    assert "no citation in [DOC:PAGE] form" in guardrail.check(note, FACTS, ALLOWED)


def test_policy_table():
    d = decide(4, "moderate", 0.9, DEFAULT_POLICY)
    assert (d.priority, d.due_days, d.needs_review) == ("P1", 30, True)
    d = decide(2, "severe", 0.9, DEFAULT_POLICY)
    assert (d.priority, d.due_days, d.needs_review) == ("P3", 180, False)
    d = decide(2, "mild", 0.5, DEFAULT_POLICY)
    assert d.needs_review and "confidence" in d.review_reasons[0]
