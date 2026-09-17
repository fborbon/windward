"""Builds the static dashboard payload for the Wind Prediction tab: the taxonomy table, the
full metrics comparison, and a plotting-ready actual-vs-predicted sample for a handful of
representative techniques (one per "shape" of result, not all 11 - the metrics table already
covers all 11 for anyone who wants the full comparison).

Run manually (`python -m wind_prediction.export`) whenever a technique changes - this is a
demonstrative comparison over a fixed historical SCADA period, not something that needs to
refresh on a schedule (see README.md §13 for why, and api/main.py's live weather endpoint for
the one part of this project that *does* refresh live).
"""
import json
from pathlib import Path

import pandas as pd

from wind_prediction.evaluate import run
from wind_prediction.taxonomy import FAMILIES

EXPORT_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "wind_prediction_payload.json"

# One representative line per "shape" of result for the comparison chart, so it stays readable -
# the results table below carries the full 11-technique comparison regardless of this choice.
CHART_TECHNIQUES = [
    "Seasonal naive (t-24h)",
    "SARIMA",
    "Gradient Boosting (recursive)",
    "Chronos-Bolt-Tiny (zero-shot)",
    "Hybrid (ETS + GBM residual)",
]
CHART_DAYS = 14


def export(farm_id: str = "kelmarsh") -> dict:
    out = run(farm_id=farm_id)
    results_df: pd.DataFrame = out["results"]
    predictions: dict = out["predictions"]
    y_true: pd.Series = out["y_true"]

    tail_idx = y_true.index[-24 * CHART_DAYS:]
    series_out = {"actual": [None if pd.isna(v) else round(float(v), 3) for v in y_true.loc[tail_idx]]}
    for name in CHART_TECHNIQUES:
        pred = predictions[name].reindex(tail_idx)
        series_out[name] = [None if pd.isna(v) else round(float(v), 3) for v in pred]

    best_name = results_df["mae_mw"].idxmin()
    payload = {
        "farm_id": farm_id,
        "taxonomy": FAMILIES,
        "results": [
            {
                "name": name,
                "family": row["family"],
                **{k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items() if k != "family"},
            }
            for name, row in results_df.iterrows()
        ],
        "best_technique": best_name,
        "chart_series": series_out,
        "chart_techniques": CHART_TECHNIQUES,
        "chart_days": CHART_DAYS,
        "meta": {
            "hours_evaluated": int(len(y_true)),
            "horizon_hours": 24,
            "n_windows": int(len(y_true) / 24),
        },
    }

    EXPORT_PATH.parent.mkdir(exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"wrote {EXPORT_PATH} ({EXPORT_PATH.stat().st_size / 1024:.0f} KB)")
    print(results_df.sort_values("mae_mw"))
    return payload


if __name__ == "__main__":
    export()
