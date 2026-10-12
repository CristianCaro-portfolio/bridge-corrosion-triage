"""Corrosion condition state segmentation: mask decoding, ONNX inference and element quantities.

Classes follow the VT corrosion dataset and the AASHTO element condition states:
0 = background or good steel (CS1), 1 = fair (CS2), 2 = poor (CS3), 3 = severe (CS4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("background", "fair", "poor", "severe")
MASK_COLORS = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3}
PALETTE = np.array([[0, 0, 0], [220, 60, 50], [40, 160, 70], [235, 200, 40]], dtype=np.uint8)

# ImageNet statistics, the backbone was pretrained with them
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_mask(path: str | Path) -> np.ndarray:
    """Read an RGB annotation mask and return an (H, W) array of class ids."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    seen = np.zeros(rgb.shape[:2], dtype=bool)
    for color, cls in MASK_COLORS.items():
        hit = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        out[hit] = cls
        seen |= hit
    if not seen.all():
        raise ValueError(f"{path}: {int((~seen).sum())} pixels with an unknown colour")
    return out


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """RGB image to a normalised NCHW float32 tensor."""
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    arr = (arr / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


@dataclass(frozen=True)
class ConditionQuantities:
    """Share of the photo in each condition state, the unit AASHTO element inspection uses."""

    fair_pct: float
    poor_pct: float
    severe_pct: float
    worst_state: int  # 1..4, CS1 means no corrosion above the noise floor
    mean_confidence: float

    @property
    def corroded_pct(self) -> float:
        return round(self.fair_pct + self.poor_pct + self.severe_pct, 2)

    def as_dict(self) -> dict:
        return {
            "fair_pct": self.fair_pct,
            "poor_pct": self.poor_pct,
            "severe_pct": self.severe_pct,
            "corroded_pct": self.corroded_pct,
            "worst_state": self.worst_state,
            "mean_confidence": self.mean_confidence,
        }


def quantify(mask: np.ndarray, min_area_pct: float, probs: np.ndarray | None = None) -> ConditionQuantities:
    """Area share per state. A state counts for the worst state only above min_area_pct,
    so a few stray pixels do not escalate a photo."""
    total = mask.size
    pct = [round(100.0 * float((mask == c).sum()) / total, 2) for c in (1, 2, 3)]
    worst = 1
    for cls, p in zip((1, 2, 3), pct):
        if p >= min_area_pct:
            worst = cls + 1
    conf = 1.0 if probs is None else round(float(probs.max(axis=0).mean()), 3)
    return ConditionQuantities(pct[0], pct[1], pct[2], worst, conf)


class OnnxSegmenter:
    """Runs the exported model with ONNX Runtime on CPU."""

    def __init__(self, model_path: str | Path, size: int, threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.size = size

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return (mask, probs) at the original image resolution."""
        w, h = image.size
        logits = self.session.run(None, {self.input_name: preprocess(image, self.size)})[0][0]
        logits = logits - logits.max(axis=0, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=0, keepdims=True)
        mask_small = probs.argmax(axis=0).astype(np.uint8)
        mask = np.asarray(Image.fromarray(mask_small).resize((w, h), Image.NEAREST))
        return mask, probs


class OracleSegmenter:
    """Returns the annotated mask. Used in tests and to isolate the agent from vision errors."""

    def __init__(self, mask_dir: str | Path):
        self.mask_dir = Path(mask_dir)

    def predict_path(self, image_path: str | Path) -> tuple[np.ndarray, None]:
        stem = Path(image_path).stem
        return decode_mask(self.mask_dir / f"{stem}.png"), None
