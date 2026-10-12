import numpy as np
import pytest
from PIL import Image

from bridge_triage.segment import decode_mask, preprocess, quantify


def _mask_file(tmp_path, rgb):
    path = tmp_path / "m.png"
    Image.fromarray(rgb).save(path)
    return path


def test_decode_mask_maps_the_four_colours(tmp_path):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 1] = (128, 0, 0)
    rgb[1, 0] = (0, 128, 0)
    rgb[1, 1] = (128, 128, 0)
    assert decode_mask(_mask_file(tmp_path, rgb)).tolist() == [[0, 1], [2, 3]]


def test_decode_mask_rejects_unknown_colours(tmp_path):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    with pytest.raises(ValueError, match="unknown colour"):
        decode_mask(_mask_file(tmp_path, rgb))


def test_quantify_ignores_states_below_the_noise_floor():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:10, :] = 1  # 10% fair
    mask[10:11, :20] = 3  # 0.2% severe, below 0.5%
    q = quantify(mask, min_area_pct=0.5)
    assert q.fair_pct == 10.0 and q.severe_pct == 0.2
    assert q.worst_state == 2
    assert q.corroded_pct == 10.2


def test_quantify_reports_worst_state_and_confidence():
    mask = np.full((10, 10), 2, dtype=np.uint8)
    probs = np.full((4, 10, 10), 0.1)
    probs[2] = 0.7
    q = quantify(mask, 0.5, probs)
    assert q.worst_state == 3
    assert q.mean_confidence == pytest.approx(0.7)


def test_preprocess_shape_and_normalisation():
    x = preprocess(Image.new("RGB", (40, 30), (124, 116, 104)), 32)
    assert x.shape == (1, 3, 32, 32)
    assert abs(float(x.mean())) < 0.05  # the ImageNet mean colour maps to about zero
