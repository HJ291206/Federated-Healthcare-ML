import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from ml.evaluation import classification_metrics, select_thresholds, threshold_for_recall, threshold_max_f1


def test_metrics_match_hand_computed_example():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.6, 0.2, 0.9, 0.3, 0.7, 0.8])
    m = classification_metrics(y, p, threshold=0.5)
    # predictions: 0 0 1 0 1 0 1 1 -> TN=3, FP=2, FN=1, TP=2
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 2, 1, 2)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["precision_class1"] == pytest.approx(2 / 4)
    assert m["recall_class1"] == pytest.approx(2 / 3)
    assert m["precision_class0"] == pytest.approx(3 / 4)
    assert m["recall_class0"] == pytest.approx(3 / 5)
    assert m["f1_class1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert m["precision"] == m["precision_class1"] and m["recall"] == m["recall_class1"]
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 5) / 2)


def test_ranking_metrics_bounds():
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert classification_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0


def test_all_negative_predictions_do_not_crash():
    m = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.99)
    assert m["precision_class1"] == 0 and m["recall_class1"] == 0 and m["tp"] == 0


@pytest.fixture
def scores():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.15).astype(int)
    p = np.clip(0.3 * y + rng.normal(0.3, 0.15, len(y)), 0, 1)
    return y, p


def test_threshold_max_f1_matches_brute_force(scores):
    y, p = scores
    t = threshold_max_f1(y, p)
    best = max(f1_score(y, p >= c) for c in np.unique(p))
    assert f1_score(y, p >= t) == pytest.approx(best)


def test_threshold_for_recall_is_highest_meeting_target(scores):
    y, p = scores
    t = threshold_for_recall(y, p, 0.8)
    assert recall_score(y, p >= t) >= 0.8
    higher = np.unique(p[p > t])
    assert higher.size == 0 or recall_score(y, p >= higher.min()) < 0.8


def test_select_thresholds_keys(scores):
    y, p = scores
    assert set(select_thresholds(y, p, 0.8)) == {"default_0.5", "val_max_f1", "val_recall_0.80"}
