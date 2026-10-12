"""Maintenance policy: maps measured condition states and site exposure to an action.

The action is decided here, in plain code, and never by the language model. The model only
writes the note that explains it. The table is configurable in config.yaml. The default
values are a reasonable starting point, not numbers taken from AASHTO or FHWA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field


class Rule(BaseModel):
    priority: str
    action: str
    due_days: dict[str, int]  # by exposure class
    always_review: bool


class PolicyConfig(BaseModel):
    rules: dict[int, Rule]  # keyed by worst condition state 1..4
    min_confidence: float = Field(0.75, description="below this mean pixel confidence, a person reviews")


DEFAULT_POLICY = PolicyConfig(
    rules={
        4: Rule(
            priority="P1",
            action="Structural evaluation of section loss by a qualified engineer before any coating work",
            due_days={"mild": 30, "moderate": 30, "severe": 14},
            always_review=True,
        ),
        3: Rule(
            priority="P2",
            action="Measure remaining thickness, clean to bare steel and zone paint the affected area",
            due_days={"mild": 180, "moderate": 120, "severe": 90},
            always_review=True,
        ),
        2: Rule(
            priority="P3",
            action="Clean and spot paint the corroded area as preservation work",
            due_days={"mild": 365, "moderate": 365, "severe": 180},
            always_review=False,
        ),
        1: Rule(
            priority="P4",
            action="No corrosion repair, keep cyclical washing and recheck at the next routine inspection",
            due_days={"mild": 730, "moderate": 730, "severe": 365},
            always_review=False,
        ),
    }
)


@dataclass(frozen=True)
class Decision:
    worst_state: int
    priority: str
    action: str
    due_days: int
    needs_review: bool
    review_reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["review_reasons"] = list(self.review_reasons)
        return d


def decide(worst_state: int, exposure_class: str, confidence: float, policy: PolicyConfig) -> Decision:
    rule = policy.rules[worst_state]
    reasons = []
    if rule.always_review:
        reasons.append(f"CS{worst_state} always goes to an engineer")
    if confidence < policy.min_confidence:
        reasons.append(f"model confidence {confidence:.2f} below {policy.min_confidence:.2f}")
    return Decision(
        worst_state=worst_state,
        priority=rule.priority,
        action=rule.action,
        due_days=rule.due_days[exposure_class],
        needs_review=bool(reasons),
        review_reasons=tuple(reasons),
    )
