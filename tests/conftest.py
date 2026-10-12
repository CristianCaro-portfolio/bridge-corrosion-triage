from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def settings(tmp_path):
    from bridge_triage.config import load_settings

    s = load_settings(ROOT / "config.yaml")
    s.corpus_path = str(ROOT / "sample_data/fhwa_chunks.jsonl")
    s.retrieval_mode = "bm25"
    s.provider = "template"
    s.checkpoint_db = str(tmp_path / "ckpt.sqlite")
    s.findings_db = str(tmp_path / "findings.duckdb")
    for site in s.sites.values():
        site.weather_file = str(ROOT / site.weather_file)
    return s
