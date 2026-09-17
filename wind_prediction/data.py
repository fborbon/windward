"""Shared target series, split and evaluation-window setup for every technique in models.py -
one place so every technique is scored on an identical protocol.

Evaluation protocol: 24h-ahead forecasts from non-overlapping windows spanning the chronological
test period (same 80/20 chronological split as demo_notebook/, and for the same documented
reason: a random split leaks seasons across train/test and overstates real forward accuracy).
Each technique forecasts hours [origin, origin+24) using only real data strictly before origin -
never another technique's predictions, and never true values from inside the forecast window
itself. This is a standard sliding-window backtest, not a full walk-forward *retrain* (classical
models are updated with true observations between windows via `.append(refit=False)`, which is
cheap and realistic; nothing is retrained from scratch per window) - see README.md §13 for the
trade-off against a fully retrained walk-forward.
"""
import numpy as np
import pandas as pd

from data_sources.farms import FARMS
from forecasting.features import TARGET_COLUMN
from forecasting.pipeline import build_training_frame

HORIZON = 24 # hours forecast ahead per window
STRIDE = 24 # non-overlapping windows
LOOKBACK = 72 # hours of history the DL/Transformer models condition on
TRAIN_FRACTION = 0.8


def load_series(farm_id: str = "kelmarsh") -> pd.DataFrame:
    """Full feature+target frame (forecasting.pipeline.build_training_frame, reused as-is),
    with a clean integer RangeIndex - statsmodels' state-space models need a regular index to
    support incremental `.append()`, and the real SCADA gaps this data drops (see demo_notebook
    §2) mean the real timestamp index isn't evenly spaced."""
    df = build_training_frame(FARMS[farm_id]).reset_index()
    df = df.rename(columns={"index": "timestamp"} if "index" in df.columns else {})
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    return df


def chronological_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def window_origins(test_len: int, horizon: int = HORIZON, stride: int = STRIDE) -> list[int]:
    """Indices (into the test frame) where each 24h forecast window starts."""
    return list(range(0, test_len - horizon + 1, stride))


def evaluated_index(test_df: pd.DataFrame, horizon: int = HORIZON, stride: int = STRIDE) -> pd.Index:
    """The subset of test_df's index actually covered by the window grid above - a technique's
    prediction Series should be reindexed to exactly this before scoring, so every technique is
    compared on the same hours regardless of how it internally windows."""
    idx = []
    for origin in window_origins(len(test_df), horizon, stride):
        idx.extend(range(origin, origin + horizon))
    return test_df.index[idx]
