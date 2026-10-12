"""Fine-tunes LR-ASPP MobileNetV3 on the corrosion condition state masks and exports to ONNX.

Needs the optional extra: pip install -e ".[train]"
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from .evaluate import confusion, segmentation_scores
from .segment import MEAN, STD, decode_mask

NUM_CLASSES = 4


def _pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted((split_dir / "images_512").glob("*.jpeg"), key=lambda p: int(p.stem))
    out = []
    for img in images:
        mask = split_dir / "mask_512" / f"{img.stem}.png"
        if mask.exists():
            out.append((img, mask))
    return out


def _load(pair: tuple[Path, Path], size: int, augment: bool, rng: random.Random):
    img = Image.open(pair[0]).convert("RGB")
    mask = Image.fromarray(decode_mask(pair[1]))
    if augment:
        # random crop between 60% and 100% of the side, keeps the scale of the defects plausible
        side = img.size[0]
        crop = int(side * rng.uniform(0.6, 1.0))
        x0, y0 = rng.randint(0, side - crop), rng.randint(0, side - crop)
        box = (x0, y0, x0 + crop, y0 + crop)
        img, mask = img.crop(box), mask.crop(box)
        if rng.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
    img = img.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1), np.asarray(mask, dtype=np.int64)


def build_model(pretrained: bool):
    from torch import nn
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
        lraspp_mobilenet_v3_large,
    )

    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights)
    head = model.classifier
    head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, NUM_CLASSES, 1)
    head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, NUM_CLASSES, 1)
    return model


def train(
    data_dir: Path,
    out_dir: Path,
    size: int = 256,
    epochs: int = 14,
    batch_size: int = 8,
    lr: float = 6e-4,
    val_fraction: float = 0.1,
    pretrained: bool = True,
    limit: int | None = None,
    seed: int = 7,
    threads: int = 2,
) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = random.Random(seed)
    pairs = _pairs(data_dir / "Train")
    rng.shuffle(pairs)
    if limit:
        pairs = pairs[:limit]
    n_val = max(1, int(len(pairs) * val_fraction))
    val, tr = pairs[:n_val], pairs[n_val:]
    batch_size = max(1, min(batch_size, len(tr)))

    # class weights from pixel frequency, square root to avoid over-weighting the rare class
    counts = np.zeros(NUM_CLASSES)
    for _, m in tr:
        counts += np.bincount(decode_mask(m).ravel(), minlength=NUM_CLASSES)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
    weights = weights / weights.mean()

    model = build_model(pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(tr) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    val_xy = [_load(p, size, False, rng) for p in val]
    best, history = -1.0, []
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best.pt"
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        rng.shuffle(tr)
        total = 0.0
        for i in range(0, len(tr) - batch_size + 1, batch_size):
            batch = [_load(p, size, True, rng) for p in tr[i : i + batch_size]]
            x = torch.from_numpy(np.stack([b[0] for b in batch]))
            y = torch.from_numpy(np.stack([b[1] for b in batch]))
            opt.zero_grad()
            loss = loss_fn(model(x)["out"], y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_xy:
                pred = model(torch.from_numpy(x[None]))["out"][0].argmax(0).numpy()
                cm += confusion(y, pred, NUM_CLASSES)
        scores = segmentation_scores(cm)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(total / max(1, len(tr)), 4),
                "val_miou": scores["miou"],
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if scores["miou"] > best:
            best = scores["miou"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    onnx_path = export_onnx(model, out_dir / "corrosion_seg.onnx", size)
    info = {
        "architecture": "lraspp_mobilenet_v3_large",
        "pretrained": "COCO_WITH_VOC_LABELS_V1" if pretrained else None,
        "input_size": size,
        "epochs": epochs,
        "train_images": len(tr),
        "val_images": len(val),
        "class_weights": [round(float(w), 3) for w in weights],
        "best_val_miou": best,
        "train_seconds": round(time.time() - t0, 1),
        "torch_threads": threads,
        "history": history,
        "onnx": onnx_path.name,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    return info


class _OutOnly:
    """Wraps the torchvision model so the exported graph has a single tensor output."""

    def __new__(cls, model):
        from torch import nn

        class Wrapped(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)["out"]

        return Wrapped(model)


def export_onnx(model, path: Path, size: int) -> Path:
    import torch

    model.eval()
    wrapped = _OutOnly(model)
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    return path
