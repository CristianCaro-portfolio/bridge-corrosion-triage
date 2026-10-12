"""Deterministic checks on a generated note before it can reach the inspection record."""

from __future__ import annotations

import re

CITE = re.compile(r"\[([A-Z][A-Z0-9-]+:\d+)\]")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_STATE = re.compile(r"\bCS\s?([1-4])\b")


def allowed_numbers(facts: dict) -> set[float]:
    keys = ("fair_pct", "poor_pct", "severe_pct", "corroded_pct", "due_days", "tow_hours", "worst_state")
    vals = {float(facts[k]) for k in keys if k in facts}
    vals |= {1.0, 2.0, 3.0, 4.0}  # condition state and priority digits
    vals |= {float(facts.get("year", 0))}
    return vals


def check(note: str, facts: dict, allowed_cites: set[str], tol: float = 0.15) -> list[str]:
    """Returns the list of problems. Empty means the note can be used."""
    problems: list[str] = []
    if not note or len(note.split()) < 12:
        return ["note is empty or too short"]
    if len(note.split()) > 140:
        problems.append("note is longer than 140 words")

    cites = set(CITE.findall(note))
    if not cites:
        problems.append("no citation in [DOC:PAGE] form")
    unknown = sorted(cites - allowed_cites)
    if unknown:
        problems.append(f"cites pages that were not retrieved: {', '.join(unknown)}")

    worst = facts["worst_state"]
    states = {int(s) for s in _STATE.findall(note)}
    if worst not in states:
        problems.append(f"does not state the worst condition state CS{worst}")
    worse = [s for s in states if s > worst]
    if worse:
        problems.append(f"mentions CS{max(worse)}, worse than the measured CS{worst}")

    if facts["priority"] not in note:
        problems.append(f"does not give the priority code {facts['priority']}")
    if str(facts["due_days"]) not in note:
        problems.append(f"does not give the due time of {facts['due_days']} days")

    body = CITE.sub(" ", note)
    body = re.sub(r"\b[A-Z]+-[A-Z0-9-]+\b", " ", body)  # document ids such as FHWA-HIF-16-002
    allowed = allowed_numbers(facts)
    invented = []
    for raw in _NUM.findall(body):
        value = float(raw)
        if not any(abs(value - a) <= tol for a in allowed):
            invented.append(raw)
    if invented:
        problems.append(f"numbers not in the facts: {', '.join(sorted(set(invented)))}")
    return problems
