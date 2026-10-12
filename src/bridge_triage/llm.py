"""Swappable text generation providers for the inspection note.

template  deterministic note built from the facts, used in tests and CI
hf        a small open model run locally on CPU with transformers (default Qwen2.5-0.5B-Instruct)
openai    any OpenAI compatible endpoint (OpenAI, Ollama, vLLM, LM Studio) set by environment
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

SYSTEM = (
    "You write short bridge inspection notes for steel members. Use only the facts given. "
    "Do not invent measurements. Cite guidance with the exact tag shown, for example "
    "[FHWA-HIF-16-002:45]. Plain text, no lists, no markdown."
)


def build_prompt(facts: dict, excerpts: list[dict], feedback: list[str] | None = None) -> str:
    lines = [
        "Facts:",
        json.dumps(facts, indent=1),
        "",
        "Guidance excerpts:",
    ]
    for e in excerpts:
        lines.append(f"[{e['cite']}] {e['text'][:600]}")
    lines += [
        "",
        "Write a note of 2 to 4 sentences for the inspection record. It must:",
        f"- state the worst condition state as {facts['worst_state_label']}",
        "- give the corroded share of the photo using the percentages in the facts",
        f"- state the action with its priority code {facts['priority']} and the due time in days",
        "- cite at least one excerpt using its tag in square brackets",
    ]
    if feedback:
        lines += ["", "Your previous note was rejected because:"] + [f"- {f}" for f in feedback]
        lines.append("Write the note again and fix these problems.")
    return "\n".join(lines)


@dataclass
class Generation:
    text: str
    model: str
    seconds: float


class Provider(Protocol):
    name: str

    def generate(self, facts: dict, excerpts: list[dict], feedback: list[str] | None) -> Generation: ...


class TemplateProvider:
    name = "template"

    def generate(self, facts: dict, excerpts: list[dict], feedback: list[str] | None = None) -> Generation:
        t0 = time.perf_counter()
        cite = excerpts[0]["cite"] if excerpts else ""
        text = (
            f"Worst condition state is {facts['worst_state_label']}: {facts['corroded_pct']}% of the "
            f"photo shows corrosion ({facts['fair_pct']}% fair, {facts['poor_pct']}% poor, "
            f"{facts['severe_pct']}% severe). Site exposure is {facts['exposure_class']} with "
            f"{facts['tow_hours']} wet hours a year. Action {facts['priority']} within "
            f"{facts['due_days']} days: {facts['action'].lower()}."
        )
        if cite:
            text += f" Guidance: [{cite}]."
        return Generation(text, self.name, time.perf_counter() - t0)


class HFLocalProvider:
    """Runs an instruct model on CPU. Weights are downloaded once into the Hugging Face cache."""

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 180):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model}"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype="float32")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, facts: dict, excerpts: list[dict], feedback: list[str] | None = None) -> Generation:
        import torch

        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(facts, excerpts, feedback)},
        ]
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        text = self.tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        return Generation(text.strip(), self.name, time.perf_counter() - t0)


class OpenAICompatibleProvider:
    """Reads LLM_BASE_URL, LLM_MODEL and LLM_API_KEY. Works with Ollama at http://localhost:11434/v1."""

    def __init__(self):
        self.base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "qwen2.5:0.5b-instruct")
        self.key = os.environ.get("LLM_API_KEY", "")
        self.name = f"openai:{self.model}"

    def generate(self, facts: dict, excerpts: list[dict], feedback: list[str] | None = None) -> Generation:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(facts, excerpts, feedback)},
            ],
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.key}"} if self.key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        return Generation(text, self.name, time.perf_counter() - t0)


def get_provider(name: str, model: str | None = None) -> Provider:
    if name == "template":
        return TemplateProvider()
    if name == "hf":
        return HFLocalProvider(model) if model else HFLocalProvider()
    if name == "openai":
        return OpenAICompatibleProvider()
    raise ValueError(f"unknown provider {name}")
