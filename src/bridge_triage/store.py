"""Findings table in DuckDB: one row per photo per run, with the audit trail of the agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DDL = """
CREATE TABLE IF NOT EXISTS findings (
    thread_id        VARCHAR PRIMARY KEY,
    image            VARCHAR,
    site             VARCHAR,
    worst_state      INTEGER,
    fair_pct         DOUBLE,
    poor_pct         DOUBLE,
    severe_pct       DOUBLE,
    confidence       DOUBLE,
    exposure_class   VARCHAR,
    tow_hours        INTEGER,
    priority         VARCHAR,
    action           VARCHAR,
    due_days         INTEGER,
    note             VARCHAR,
    citations        VARCHAR[],
    status           VARCHAR,  -- auto_published, approved, edited, rejected
    reviewer         VARCHAR,
    review_comment   VARCHAR,
    provider         VARCHAR,
    attempts         INTEGER,
    validation_log   JSON,
    created_at       TIMESTAMP
)
"""


def connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(DDL)
    return con


def write_finding(path: str | Path, row: dict) -> None:
    con = connect(path)
    try:
        cols = [
            "thread_id", "image", "site", "worst_state", "fair_pct", "poor_pct", "severe_pct",
            "confidence", "exposure_class", "tow_hours", "priority", "action", "due_days", "note",
            "citations", "status", "reviewer", "review_comment", "provider", "attempts",
            "validation_log",
        ]  # fmt: skip
        values = [row.get(c) for c in cols]
        values[cols.index("validation_log")] = json.dumps(row.get("validation_log", []))
        placeholders = ", ".join("?" for _ in cols)
        con.execute(
            f"INSERT OR REPLACE INTO findings ({', '.join(cols)}, created_at) VALUES ({placeholders}, ?)",
            values + [datetime.now(timezone.utc).replace(tzinfo=None)],
        )
    finally:
        con.close()


def read_findings(path: str | Path) -> list[dict]:
    con = connect(path)
    try:
        cur = con.execute("SELECT * FROM findings ORDER BY created_at")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
    finally:
        con.close()
