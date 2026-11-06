"""The CARE score (analysis/care_score.py) on hand-checkable cases."""
import numpy as np
import pytest

from analysis.care_score import EventResult, care_score, criticality, earliness_weights, evaluate_event, fbeta


def test_fbeta_matches_the_formula():
    # tp 2, fp 1, fn 1: precision 2/3, recall 2/3 -> any beta gives 2/3
    assert fbeta([1, 1, 0, 1], [1, 1, 1, 0]) == pytest.approx(2 / 3)
    assert fbeta([1, 1], [0, 0]) == 0.0


def test_criticality_counts_only_normal_rows_and_floors_at_zero():
    flags = np.array([0, 1, 1, 1, 0, 1], bool)
    normal = np.array([1, 1, 1, 0, 1, 1], bool)
    assert criticality(flags, normal).tolist() == [0, 1, 2, 2, 1, 2]


def test_earliness_weights_hold_then_fall():
    w = earliness_weights(101)
    assert w[0] == 1 and w[25] == 1 and w[-1] == 0 and w[50] == pytest.approx(2 / 3, abs=0.01)


def test_detection_needs_twelve_hours_of_net_anomalies():
    ids = np.arange(200)
    flags = np.zeros(200, bool)
    flags[100:171] = True  # 71 rows: one short
    r = evaluate_event(1, "anomaly", ids, flags, np.ones(200, bool), 100, 199)
    assert not r.detected and r.max_criticality == 71
    flags[171] = True
    r = evaluate_event(1, "anomaly", ids, flags, np.ones(200, bool), 100, 199)
    assert r.detected and r.first_detection_id == 171 and r.lead_rows == 28


def test_special_cases_and_the_weighted_mean():
    base = dict(max_criticality=0, coverage=0.4, earliness=0.2, first_detection_id=None, lead_rows=None)
    a = EventResult(1, "anomaly", accuracy=None, detected=False, **base)
    n = EventResult(2, "normal", accuracy=1.0, detected=False, **{**base, "coverage": None, "earliness": None})
    assert care_score([a, n])["care"] == 0.0  # nothing detected
    a.detected = True
    s = care_score([a, n])
    assert s["reliability"] == 1.0 and s["care"] == pytest.approx((0.4 + 2 * 1.0 + 1.0 + 0.2) / 5, abs=1e-3)
    n.accuracy = 0.4
    assert care_score([a, n])["care"] == 0.4  # accuracy at most 0.5 caps the score
