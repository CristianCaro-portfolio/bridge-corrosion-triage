"""Settings loaded from config.yaml, with environment overrides for the provider."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from .policy import DEFAULT_POLICY, PolicyConfig


class Site(BaseModel):
    station: str
    weather_file: str
    marine: bool = False
    note: str = ""


class Settings(BaseModel):
    model_path: str = "data/model/corrosion_seg.onnx"
    input_size: int = 320
    min_area_pct: float = 0.5
    corpus_path: str = "sample_data/fhwa_chunks.jsonl"
    vector_cache: str = "data/cache/vectors.npz"
    retrieval_mode: str = "hybrid"  # bm25, dense, hybrid, hybrid+rerank
    embedder: str = "fastembed"  # fastembed or hashing
    top_k: int = 4
    provider: str = "template"
    provider_model: str | None = None
    max_attempts: int = 3
    checkpoint_db: str = "data/checkpoints.sqlite"
    findings_db: str = "data/findings.duckdb"
    sites: dict[str, Site] = {}
    policy: PolicyConfig = DEFAULT_POLICY


def load_settings(path: str | Path = "config.yaml") -> Settings:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    settings = Settings(**(data or {}))
    if os.environ.get("TRIAGE_PROVIDER"):
        settings.provider = os.environ["TRIAGE_PROVIDER"]
    if os.environ.get("TRIAGE_PROVIDER_MODEL"):
        settings.provider_model = os.environ["TRIAGE_PROVIDER_MODEL"]
    return settings
