"""Normal-behaviour model for CARE to Compare (data_sources/care_scada.py), scored with the CARE
score (analysis/care_score.py), plus everything the dashboard's case-study tab shows, written
once to dashboard/data/care_payload.json so the API never needs the 5.5 GB archive.

The detector, fixed before looking at any score:
1. Inputs: each case's 10-minute averages (angles as sin/cos, counters dropped), columns with
   more than 20% missing or no variance in training left out, gaps filled with the training
   median, standardised with the training mean and spread.
2. Training rows: the case's own "train" rows in normal operation (status 0). One model per
   case, i.e. per turbine and year, so it learns that turbine's own normal.
3. Model: PCA keeping 95% of the variance, fitted on the first 80% of those rows. A row's anomaly
   score is its mean squared reconstruction error: how far the sensors are from the
   relationships they normally keep with each other.
4. Threshold: the 99th percentile of the score on the last 20% (held out, still normal).
5. Event decision: the benchmark's criticality counter, detected at 72.

The paper's per-farm autoencoder scores 0.66 and random guessing 0.5. A grid of other settings
is also scored and written as `sensitivity`; it is tuned on the test labels, so it shows how much
the settings matter, not a better result.

    python -m analysis.care_nbm            # all 95 cases, ~15 minutes
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.care_score import care_score, evaluate_event
from analysis.efficiency import binned_power_curve, smooth_power_curve
from data_sources import care_scada as cs

PAYLOAD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "care_payload.json"
VARIANCE = 0.95
QUANTILE = 0.99
CALIBRATION_SHARE = 0.2
MAX_MISSING = 0.2
GRID_VARIANCE = (0.90, 0.95, 0.99)
GRID_QUANTILE = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)


def _features(df: pd.DataFrame, cols: dict[str, list[str]]) -> pd.DataFrame:
    out = df[cols["plain"]].copy()
    for c in cols["angles"]:
        r = np.deg2rad(df[c])
        out[f"{c}_sin"], out[f"{c}_cos"] = np.sin(r), np.cos(r)
    return out


def _clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def detect(df: pd.DataFrame, farm: str) -> dict:
    """Fit on the case's normal training rows; return per-variance scores and thresholds."""
    cols = cs.sensor_columns(farm, list(df.columns))
    feats = _features(df, cols)
    train_mask = (df["train_test"] == "train") & (df["status_type_id"] == 0)
    tr = feats[train_mask]
    keep = (tr.isna().mean() <= MAX_MISSING) & (tr.std() > 1e-6)
    feats, tr = feats.loc[:, keep], tr.loc[:, keep]
    med, mu, sd = tr.median(), tr.mean(), tr.std().replace(0, 1)
    z = _clean(((feats.fillna(med) - mu) / sd).to_numpy(dtype="float64"))
    ztr = z[train_mask.to_numpy()]
    n_cal = int(len(ztr) * CALIBRATION_SHARE)
    fit, cal = ztr[:-n_cal], ztr[-n_cal:]
    out = {"features": list(feats.columns), "by_variance": {}}
    for var in GRID_VARIANCE:
        pca = PCA(n_components=var, svd_solver="full").fit(fit)
        recon = pca.inverse_transform(pca.transform(z))
        err = (z - recon) ** 2
        score = err.mean(axis=1)
        cal_score = score[train_mask.to_numpy()][-n_cal:]
        out["by_variance"][var] = {"n_components": int(pca.n_components_), "score": score, "err": err,
                                   "thresholds": {q: float(np.quantile(cal_score, q)) for q in GRID_QUANTILE}}
    return out


def _top_sensors(err: np.ndarray, rows: np.ndarray, features: list[str], fd: pd.DataFrame, k: int = 3) -> list[dict]:
    """Which inputs carry most of the reconstruction error over the given rows: a pointer for the
    engineer, compared with the operator's root cause where one is given."""
    share = err[rows].mean(axis=0)
    share = share / share.sum() if share.sum() > 0 else share
    desc = dict(zip(fd["sensor_name"], fd["description"]))
    unit = dict(zip(fd["sensor_name"], fd["unit"]))
    out = []
    for i in np.argsort(share)[::-1][:k]:
        name = features[i]
        base = name.removesuffix("_sin").removesuffix("_cos").removesuffix("_avg")
        out.append({"sensor": name, "description": str(desc.get(base, "")), "unit": str(unit.get(base, "")),
                    "share_of_error_pct": round(float(100 * share[i]), 1)})
    return out


def _case(farm: str, event_id: int, ev: pd.Series) -> tuple[dict, dict]:
    df = cs.load_case(farm, event_id)
    meta = cs.FARMS[farm]
    series = df[[meta["wind"], meta["power"]]].rename(columns={meta["wind"]: "wind_speed_ms", meta["power"]: "power_frac"})
    curve = binned_power_curve(series, power_col="power_frac")
    smooth = smooth_power_curve(series, curve.index.values, power_col="power_frac")
    status = df["status_type_id"].value_counts().sort_index()
    start, end = pd.to_datetime(ev["event_start"]), pd.to_datetime(ev["event_end"])

    det = detect(df, farm)
    pred = (df["train_test"] == "prediction").to_numpy()
    ids = df["id"].to_numpy()[pred]
    normal = (df["status_type_id"] == 0).to_numpy()[pred] if meta["status_at_prediction"] else np.ones(pred.sum(), bool)
    label = ev["event_label"]
    grid = {}
    for var, d in det["by_variance"].items():
        for q, thr in d["thresholds"].items():
            grid[(var, q)] = evaluate_event(int(event_id), label, ids, d["score"][pred] > thr, normal,
                                            int(ev["event_start_id"]), int(ev["event_end_id"]))
    main = det["by_variance"][VARIANCE]
    res = grid[(VARIANCE, QUANTILE)]
    in_event = (ids >= ev["event_start_id"]) & (ids <= ev["event_end_id"])
    pred_rows = np.flatnonzero(pred)
    top = _top_sensors(main["err"], pred_rows[in_event] if in_event.any() else pred_rows, det["features"],
                       cs.load_feature_description(farm))
    # hourly series for the chart: score mean and criticality max, hours relative to event end
    flags = main["score"][pred] > main["thresholds"][QUANTILE]
    from analysis.care_score import criticality
    crit = criticality(flags, normal)
    hours = (ids - int(ev["event_end_id"])) / 6.0
    hourly = pd.DataFrame({"h": np.floor(hours), "score": main["score"][pred] / main["thresholds"][QUANTILE], "crit": crit})
    hourly = hourly.groupby("h").agg(score=("score", "mean"), crit=("crit", "max")).reset_index()

    case = {
        "event_id": int(event_id), "farm": farm, "turbine_id": str(ev["asset_id"]), "label": label,
        "description": ev["event_description"] if pd.notna(ev["event_description"]) else None,
        "duration_hours": round((end - start).total_seconds() / 3600, 1),
        "rows_observed": int(len(df)), "prediction_rows": int(pred.sum()),
        "power_curve": [{"wind_speed_bin": float(i), "mean_power_frac": float(r["mean_power_kw"]), "sample_count": int(r["sample_count"])}
                        for i, r in curve.iterrows() if not pd.isna(r["mean_power_kw"])],
        "power_curve_smooth": [{"wind_speed_bin": float(i), "mean_power_frac": float(v)} for i, v in smooth.items() if not pd.isna(v)],
        "status_breakdown": {cs.STATUS_LABELS.get(int(k), str(k)): int(v) for k, v in status.items()},
        "detector": {
            "detected": bool(res.detected), "max_criticality": res.max_criticality,
            "lead_hours": round(res.lead_rows / 6, 1) if res.lead_rows is not None else None,
            "coverage": None if res.coverage is None else round(res.coverage, 3),
            "accuracy": None if res.accuracy is None else round(res.accuracy, 3),
            "earliness": None if res.earliness is None else round(res.earliness, 3),
            "n_features": len(det["features"]), "n_components": main["n_components"],
            "top_sensors": top,
            "event_window_hours": [round(float((ev["event_start_id"] - ev["event_end_id"]) / 6), 1), 0.0],
            "hourly": {"hours_to_event_end": hourly["h"].astype(int).tolist(),
                       "score_vs_threshold": hourly["score"].round(2).tolist(),
                       "criticality": hourly["crit"].astype(int).tolist()},
        },
    }
    return case, grid


def build() -> dict:
    cases, grids = [], {}
    t0 = time.time()
    for farm in cs.FARMS:
        for event_id, ev in cs.load_events(farm).iterrows():
            case, grid = _case(farm, int(event_id), ev)
            cases.append(case)
            grids[int(event_id)] = grid
            print(f"[{time.time() - t0:5.0f}s] farm {farm} case {event_id:>3} {case['label']:<7} detected={case['detector']['detected']} "
                  f"crit={case['detector']['max_criticality']}", flush=True)

    def scores(params) -> dict:
        out = {}
        for farm in [*cs.FARMS, "all"]:
            ids = [c["event_id"] for c in cases if farm == "all" or c["farm"] == farm]
            out[farm] = care_score([grids[i][params] for i in ids])
        return out

    main = scores((VARIANCE, QUANTILE))
    sensitivity = [{"variance": v, "quantile": q, "care_all": scores((v, q))["all"]["care"]} for v in GRID_VARIANCE for q in GRID_QUANTILE]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"doi": "10.5281/zenodo.15846963", "paper_doi": "10.3390/data9120138", "licence": "CC BY-SA 4.0"},
        "farms": {k: {kk: vv for kk, vv in v.items() if kk not in ("wind", "power")} | {"cases": sum(c["farm"] == k for c in cases)}
                  for k, v in cs.FARMS.items()},
        "detector": {
            "method": (f"PCA normal-behaviour model per case: 10-minute averages, fitted on the case's normal training rows, "
                       f"{int(VARIANCE * 100)}% of variance kept; alarm when reconstruction error exceeds the {QUANTILE:.0%} "
                       "percentile of held-out normal rows; event detected when the benchmark's criticality counter reaches 72."),
            "variance": VARIANCE, "quantile": QUANTILE,
            "scores": main,
            "reference": {"random": 0.5, "autoencoder_per_farm_paper": 0.66,
                          "note": "Gück et al. 2024 (arXiv 2404.10320v2), all 95 cases; scores can vary on reproduction"},
            "sensitivity": sensitivity,
            "sensitivity_note": "scored on the same labels, so it shows how much the settings matter, not a tuned result",
        },
        "cases": cases,
    }


if __name__ == "__main__":
    payload = build()
    PAYLOAD_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps(payload["detector"]["scores"], indent=1))
    print(json.dumps(payload["detector"]["sensitivity"], indent=1))
    print(f"wrote {PAYLOAD_PATH} ({PAYLOAD_PATH.stat().st_size / 1e6:.2f} MB)")
