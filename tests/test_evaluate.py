import numpy as np
import pytest

from bridge_triage.evaluate import confusion, mrr, recall_at_k, segmentation_scores, worst_state_scores


def test_segmentation_scores_on_a_known_case():
    truth = np.array([0, 0, 1, 1, 2, 3])
    pred = np.array([0, 1, 1, 1, 2, 2])
    s = segmentation_scores(confusion(truth, pred, 4))
    assert s["iou"]["background"] == pytest.approx(0.5, abs=1e-3)
    assert s["iou"]["fair"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["iou"]["severe"] == 0.0
    assert s["pixel_accuracy"] == pytest.approx(4 / 6, abs=1e-3)


def test_worst_state_scores_count_missed_escalations():
    s = worst_state_scores([1, 2, 3, 4, 4], [1, 3, 2, 4, 3])
    assert s["exact"] == pytest.approx(0.4, abs=1e-3)
    assert s["escalation_true"] == 3 and s["escalation_missed"] == 1
    assert s["escalation_recall"] == pytest.approx(2 / 3, abs=1e-3)


def test_retrieval_metrics():
    ranked = [["a", "b"], ["c", "a"], ["d", "e"]]
    rel = [{"a"}, {"a"}, {"z"}]
    assert recall_at_k(ranked, rel, 1) == pytest.approx(1 / 3, abs=1e-3)
    assert recall_at_k(ranked, rel, 2) == pytest.approx(2 / 3, abs=1e-3)
    assert mrr(ranked, rel) == pytest.approx(0.5, abs=1e-3)
