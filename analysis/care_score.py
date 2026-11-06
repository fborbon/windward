"""The CARE score for early fault detection, implemented from its published definition: Gück,
Roelofs & Faulstich 2024, "CARE to Compare: A Real-World Benchmark Dataset for Early Fault
Detection in Wind Turbine Data", MDPI Data 9(12):138 (doi:10.3390/data9120138), and the
benchmark's own documentation (Fraunhofer IEE EnergyFaultDetector, CAREScore and the
CARE2Compare FAQ). Written here rather than installed because that package pulls in TensorFlow.

Per event (one CARE to Compare dataset, scored on its prediction rows):
- ground truth: True inside [event_start, event_end] of an anomaly event, and wherever the
  turbine is not in normal operation; False otherwise. Only rows in normal operation are scored
  pointwise (for Wind Farm A the status is not used at prediction time: every row counts).
- Coverage (anomaly events): F-beta (beta 0.5) of the pointwise predictions.
- Accuracy (normal events): share of scored rows predicted normal.
- Earliness (anomaly events): weighted share of the event window flagged, with weight 1 for the
  first quarter of the window falling linearly to 0 at its end.
- Criticality: a counter over the prediction rows up to event_end, +1 for a flagged row in normal
  operation, -1 for an unflagged one, unchanged otherwise, floored at 0 and capped at 1000. The
  event counts as detected when it reaches 72 (12 hours of net anomalies at 10 minutes).
Over all events:
- Reliability: event-wise F-beta (beta 0.5) of detected vs labelled anomaly.
- CARE = (Coverage + 2 x Accuracy + Reliability + Earliness) / 5 with the averages taken over
  anomaly events (Coverage, Earliness) and normal events (Accuracy); except CARE = mean Accuracy
  when that is at most 0.5, and CARE = 0 when no event is detected at all.
Reference points from the paper: random predictions 0.5, a per-farm autoencoder 0.66.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BETA = 0.5
CRITICALITY_THRESHOLD = 72
MAX_CRITICALITY = 1000
WEIGHTS = {"coverage": 1.0, "accuracy": 2.0, "reliability": 1.0, "earliness": 1.0}


def fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = BETA) -> float:
    """F-beta with sklearn's zero_division=0 convention (0 when precision and recall are both 0)."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / denom) if denom else 0.0


def criticality(flags: np.ndarray, normal: np.ndarray) -> np.ndarray:
    flags, normal = np.asarray(flags, bool), np.asarray(normal, bool)
    delta = np.where(normal & flags, 1, np.where(normal & ~flags, -1, 0))
    out = np.empty(len(delta), dtype=np.int64)
    c = 0
    for i, d in enumerate(delta):
        c = max(0, min(MAX_CRITICALITY, c + int(d)))
        out[i] = c
    return out


def earliness_weights(n: int) -> np.ndarray:
    """1 over the first quarter of an n-row event window, then linearly down to 0 at its end."""
    x = np.linspace(0, 1, n)
    return np.clip(np.where(x <= 0.25, 1.0, (1 - x) / 0.75), 0, 1)


@dataclass
class EventResult:
    event_id: int
    label: str
    detected: bool
    max_criticality: int
    coverage: float | None
    accuracy: float | None
    earliness: float | None
    first_detection_id: int | None  # row id where criticality first reached the threshold
    lead_rows: int | None  # rows between first detection and event_end (10-minute rows)


def evaluate_event(event_id: int, label: str, ids: np.ndarray, flags: np.ndarray, normal: np.ndarray,
                   event_start_id: int, event_end_id: int) -> EventResult:
    """ids, flags, normal: the prediction rows of one dataset, in order. normal: rows in normal
    operation (pass all True to ignore the status, as for Wind Farm A)."""
    ids, flags, normal = np.asarray(ids), np.asarray(flags, bool), np.asarray(normal, bool)
    in_event = (ids >= event_start_id) & (ids <= event_end_id)
    truth = ~normal | (in_event if label == "anomaly" else False)
    scored = normal
    coverage = fbeta(truth[scored], flags[scored]) if label == "anomaly" else None
    accuracy = float(np.mean(truth[scored] == flags[scored])) if scored.any() else None
    earliness = None
    if label == "anomaly" and in_event.any():
        w = earliness_weights(int(in_event.sum()))
        earliness = float(np.sum(flags[in_event] * w) / np.sum(w))
    upto = ids <= event_end_id
    crit = criticality(flags[upto], normal[upto])
    max_c = int(crit.max()) if len(crit) else 0
    first = None
    lead = None
    if label == "anomaly" and (crit >= CRITICALITY_THRESHOLD).any():
        k = int(np.argmax(crit >= CRITICALITY_THRESHOLD))
        first = int(ids[upto][k])
        lead = int(event_end_id - first)
    return EventResult(event_id, label, max_c >= CRITICALITY_THRESHOLD, max_c, coverage, accuracy, earliness, first, lead)


def care_score(results: list[EventResult]) -> dict:
    df = pd.DataFrame([r.__dict__ for r in results])
    anomaly = df["label"] == "anomaly"
    if anomaly.all() or (~anomaly).all():
        raise ValueError("the CARE score needs at least one anomaly and one normal event")
    acc = float(df.loc[~anomaly, "accuracy"].mean())
    cov = float(df.loc[anomaly, "coverage"].mean())
    ear = float(df.loc[anomaly, "earliness"].mean())
    rel = fbeta(anomaly.to_numpy(), df["detected"].to_numpy())
    parts = {"coverage": round(cov, 3), "accuracy": round(acc, 3), "reliability": round(rel, 3), "earliness": round(ear, 3)}
    if not df["detected"].any():
        score, rule = 0.0, "no event detected: CARE = 0"
    elif acc <= 0.5:
        score, rule = acc, "accuracy on normal events at most 0.5: CARE = accuracy"
    else:
        score = sum(WEIGHTS[k] * v for k, v in {"coverage": cov, "accuracy": acc, "reliability": rel, "earliness": ear}.items()) / sum(WEIGHTS.values())
        rule = "(coverage + 2 x accuracy + reliability + earliness) / 5"
    tp = int((anomaly & df["detected"]).sum())
    fp = int((~anomaly & df["detected"]).sum())
    return {"care": round(float(score), 3), "rule": rule, **parts,
            "anomaly_events": int(anomaly.sum()), "normal_events": int((~anomaly).sum()),
            "anomalies_detected": tp, "false_alarms": fp}
