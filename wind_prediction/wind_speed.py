"""Wind-speed forecasting at Kelmarsh with five time-series models, graded against the measured wind.

Everywhere else in this project the wind resource comes from ERA5 reanalysis (Open-Meteo archive, 10 m).
This module is a model demonstration on the question underneath that: how well can the wind itself
be predicted at a farm? Kelmarsh is used because it has a reference to grade against: the hub-height
anemometers of its six turbines (2016 SCADA), averaged into one farm wind speed.

Five forecasting models, each a different paradigm, forecast the next 24 h from the measured history
only (univariate, no weather-forecast input):
  seasonal naive      repeat the last 24 hours                                   (baseline)
  SARIMA              (1,1,1)(1,0,0)[24], statsmodels                            (classical statistical)
  gradient boosting   HistGradientBoosting, direct multi-output on 72 h windows  (machine learning)
  GRU                 PyTorch, trained on changes relative to the last value     (deep learning)
  LSTM                TensorFlow/Keras, same set-up as the GRU                   (deep learning)
plus persistence (last value repeated), the bar any short-range forecast has to clear.

Next to them, the reference the project actually uses: ERA5 at 10 m, ERA5 at 100 m (closer to the
~79 m hub), and ERA5 100 m linearly calibrated on the training period. ERA5 is not a forecast (it is
reanalysis, produced after the fact), so those rows answer a different question: how far is the
current wind source from the measured wind at the site.

Protocol, identical for every row (see wind_prediction/data.py): chronological 80/20 split,
non-overlapping 24 h windows over the test period, each forecast uses only data before its window.
Short SCADA gaps (up to 6 h) are interpolated so the models see a continuous series; scores are
computed only on hours that were really measured.

    .venv-tsdemo/bin/python -m wind_prediction.wind_speed        # writes dashboard/data/wind_speed_payload.json
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import greenbyte_scada
from data_sources.farms import FARMS
# Same protocol constants as wind_prediction/data.py (not imported: that module pulls in the whole
# production pipeline, which this offline demo does not need)
HORIZON, STRIDE, LOOKBACK, TRAIN_FRACTION = 24, 24, 72, 0.8


def window_origins(test_len: int, horizon: int = HORIZON, stride: int = STRIDE) -> list[int]:
    return list(range(0, test_len - horizon + 1, stride))

ROOT = Path(__file__).resolve().parent.parent
EXPORT_PATH = ROOT / "dashboard" / "data" / "wind_speed_payload.json"
CACHE_DIR = ROOT / "data" / "wind_speed_cache"
SEED = 42
MIN_TURBINES = 3          # hours with fewer turbines reporting are treated as missing
MAX_GAP_HOURS = 6
CHART_DAYS = 14


# ── Data ───────────────────────────────────────────────────────────────────────
def measured_wind(farm_id: str = "kelmarsh") -> pd.Series:
    """Hourly farm-mean hub-height wind speed from the turbines' own anemometers."""
    farm = FARMS[farm_id]
    long = greenbyte_scada.load_turbine_hourly_series(farm.scada_zips)
    g = long.groupby("timestamp")["wind_speed_ms"]
    mean = g.mean()[g.count() >= MIN_TURBINES]
    mean.index = pd.to_datetime(mean.index)
    return mean.sort_index().rename("measured")


def era5_wind(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """ERA5 wind speed at 10 m and 100 m for the farm (Open-Meteo archive), cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"era5_{lat:.3f}_{lon:.3f}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                "hourly": "wind_speed_10m,wind_speed_100m", "wind_speed_unit": "ms"})
    with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=120) as r:
        h = json.loads(r.read())["hourly"]
    df = pd.DataFrame({"era5_10m": h["wind_speed_10m"], "era5_100m": h["wind_speed_100m"]},
                      index=pd.to_datetime(h["time"]))
    df.to_csv(cache)
    return df


def load_frame(farm_id: str = "kelmarsh") -> pd.DataFrame:
    farm = FARMS[farm_id]
    meas = measured_wind(farm_id)
    full = pd.date_range(meas.index.min(), meas.index.max(), freq="h")
    df = pd.DataFrame(index=full)
    df["measured"] = meas.reindex(full)
    df["observed"] = df["measured"].notna()          # score only real measurements
    df["measured"] = df["measured"].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    era = era5_wind(farm.lat, farm.lon, full.min().date().isoformat(), full.max().date().isoformat())
    df = df.join(era, how="left")
    # trim leading/trailing hours the interpolation could not fill, then keep the longest clean stretch
    df = df.dropna(subset=["measured", "era5_10m", "era5_100m"])
    return df


# ── Models (settings carried over from the time-series model comparison) ─────────
def _windows(values: np.ndarray, lookback: int, horizon: int):
    n = len(values) - lookback - horizon + 1
    X = np.stack([values[i:i + lookback] for i in range(n)])
    y = np.stack([values[i + lookback:i + lookback + horizon] for i in range(n)])
    return X, y


def _history_before(train: np.ndarray, test: np.ndarray, origin: int) -> np.ndarray:
    return np.concatenate([train, test[:origin]])


def persistence(train, test, origins):
    out = np.full(len(test), np.nan)
    for o in origins:
        out[o:o + HORIZON] = _history_before(train, test, o)[-1]
    return out


def seasonal_naive(train, test, origins, period=24):
    out = np.full(len(test), np.nan)
    for o in origins:
        last = _history_before(train, test, o)[-period:]
        out[o:o + HORIZON] = np.tile(last, int(np.ceil(HORIZON / period)))[:HORIZON]
    return out


def sarima(train, test, origins):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    res = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 24),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
    out, pos = np.full(len(test), np.nan), 0
    for o in origins:
        if o > pos:                                   # condition on the true values up to the origin
            res = res.append(test[pos:o], refit=False)
            pos = o
        out[o:o + HORIZON] = res.forecast(HORIZON)
    return out


class _Scaler:
    def __init__(self, train):
        self.lo, self.hi = float(train.min()), float(train.max())

    def f(self, v):
        return (v - self.lo) / (self.hi - self.lo)

    def inv(self, v):
        return v * (self.hi - self.lo) + self.lo


def gradient_boosting(train, test, origins):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    sc = _Scaler(train)
    X, y = _windows(sc.f(train), LOOKBACK, HORIZON)
    gbm = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=SEED)).fit(X, y)
    out = np.full(len(test), np.nan)
    for o in origins:
        w = sc.f(_history_before(train, test, o)[-LOOKBACK:]).reshape(1, -1)
        out[o:o + HORIZON] = sc.inv(gbm.predict(w).ravel())
    return out


def _delta_windows(train):
    sc = _Scaler(train)
    X, y = _windows(sc.f(train), LOOKBACK, HORIZON)
    last = X[:, -1:]
    return sc, (X - last).reshape(-1, LOOKBACK, 1).astype("float32"), (y - last).astype("float32")


def gru_pytorch(train, test, origins, epochs=20):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    sc, Xd, yd = _delta_windows(train)

    class GRUForecaster(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, HORIZON)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])

    net, Xt, yt = GRUForecaster(), torch.tensor(Xd), torch.tensor(yd)
    opt, loss_fn = torch.optim.Adam(net.parameters(), lr=1e-2), nn.MSELoss()
    for _ in range(epochs):                                    # full batch, as in the source comparison
        opt.zero_grad()
        loss = loss_fn(net(Xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    out = np.full(len(test), np.nan)
    with torch.no_grad():
        for o in origins:
            w = sc.f(_history_before(train, test, o)[-LOOKBACK:])
            pred = net(torch.tensor((w - w[-1]).reshape(1, LOOKBACK, 1), dtype=torch.float32)).numpy().ravel()
            out[o:o + HORIZON] = sc.inv(pred + w[-1])
    return out


def lstm_keras(train, test, origins, epochs=10):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.random.set_seed(SEED)
    sc, Xd, yd = _delta_windows(train)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(LOOKBACK, 1)),
                                 tf.keras.layers.LSTM(32), tf.keras.layers.Dense(HORIZON)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xd, yd, epochs=epochs, batch_size=64, verbose=0)
    batch = []
    for o in origins:
        w = sc.f(_history_before(train, test, o)[-LOOKBACK:])
        batch.append((w - w[-1], w[-1]))
    preds = model.predict(np.stack([b[0] for b in batch]).reshape(-1, LOOKBACK, 1), verbose=0)
    out = np.full(len(test), np.nan)
    for o, p, (_, last) in zip(origins, preds, batch):
        out[o:o + HORIZON] = sc.inv(p + last)
    return out


# ── Scoring ────────────────────────────────────────────────────────────────────
def _metrics(y, p):
    err = p - y
    return {"mae_ms": float(np.mean(np.abs(err))), "rmse_ms": float(np.sqrt(np.mean(err ** 2))),
            "bias_ms": float(np.mean(err)), "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}


def run(farm_id: str = "kelmarsh", verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    log = (lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)) if verbose else (lambda m: None)

    log("loading measured wind (SCADA) + ERA5")
    df = load_frame(farm_id)
    split = int(len(df) * TRAIN_FRACTION)
    tr, te = df.iloc[:split], df.iloc[split:]
    train, test = tr["measured"].to_numpy(), te["measured"].to_numpy()
    origins = window_origins(len(te), HORIZON, STRIDE)
    covered = np.zeros(len(te), bool)
    for o in origins:
        covered[o:o + HORIZON] = True
    mask = covered & te["observed"].to_numpy()
    lead = np.zeros(len(te), int)
    for o in origins:
        lead[o:o + HORIZON] = np.arange(1, HORIZON + 1)

    # ERA5 100 m calibrated with a linear fit on the training period (simple model-output statistics)
    a, b = np.polyfit(tr["era5_100m"].to_numpy(), train, 1)

    rows = [  # name, kind, family, predictions
        ("Persistence (last value)", "forecast", "Naive / baseline", None, lambda: persistence(train, test, origins)),
        ("Seasonal naive (24 h)", "forecast", "Naive / baseline", None, lambda: seasonal_naive(train, test, origins)),
        ("SARIMA (1,1,1)(1,0,0)[24]", "forecast", "Classical statistical", "statsmodels", lambda: sarima(train, test, origins)),
        ("Gradient boosting (direct, 72 h window)", "forecast", "Machine learning", "scikit-learn",
         lambda: gradient_boosting(train, test, origins)),
        ("GRU", "forecast", "Deep learning", "PyTorch", lambda: gru_pytorch(train, test, origins)),
        ("LSTM", "forecast", "Deep learning", "TensorFlow / Keras", lambda: lstm_keras(train, test, origins)),
        ("ERA5 reanalysis, 10 m (used by this project)", "reference", "Reanalysis", None, lambda: te["era5_10m"].to_numpy()),
        ("ERA5 reanalysis, 100 m", "reference", "Reanalysis", None, lambda: te["era5_100m"].to_numpy()),
        ("ERA5 100 m, calibrated on training period", "reference", "Reanalysis + linear calibration", None,
         lambda: a * te["era5_100m"].to_numpy() + b),
    ]
    results, preds = [], {}
    y = test[mask]
    for name, kind, family, library, fn in rows:
        log(name)
        t0 = time.time()
        p = fn()
        secs = time.time() - t0
        preds[name] = p
        by_lead = [float(np.mean(np.abs(p[mask & (lead == h)] - test[mask & (lead == h)]))) for h in range(1, HORIZON + 1)]
        results.append({"name": name, "kind": kind, "family": family, "library": library,
                        **_metrics(y, p[mask]), "mae_by_lead_ms": by_lead, "seconds": round(secs, 1)})

    forecasts = sorted([r for r in results if r["kind"] == "forecast"], key=lambda r: r["mae_ms"])
    refs = sorted([r for r in results if r["kind"] == "reference"], key=lambda r: r["mae_ms"])
    best = forecasts[0]["name"]
    best_ml = next(r["name"] for r in forecasts if r["family"] not in ("Naive / baseline",))

    n_chart = CHART_DAYS * 24
    chart_idx = np.where(covered)[0][-n_chart:]
    chart_names = ["Persistence (last value)", best_ml, "ERA5 100 m, calibrated on training period"]
    meas_chart = np.where(te["observed"].to_numpy(), test, np.nan)[chart_idx]
    payload = {
        "farm_id": farm_id,
        "meta": {
            "period": f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}",
            "test_period": f"{te.index.min():%Y-%m-%d} to {te.index.max():%Y-%m-%d}",
            "n_windows": len(origins), "hours_scored": int(mask.sum()), "horizon_h": HORIZON,
            "lookback_h": LOOKBACK, "reference": "farm-mean hub-height anemometer wind speed, 6 turbines (SCADA)",
            "measured_mean_ms": round(float(y.mean()), 2),
            "era5_calibration": {"slope": round(float(a), 3), "intercept_ms": round(float(b), 3)},
        },
        "best_forecast": best,
        "forecasts": forecasts,
        "references": refs,
        "chart": {
            "timestamps": [t.isoformat() for t in te.index[chart_idx]],
            "series": {"Measured": [None if np.isnan(v) else round(float(v), 2) for v in meas_chart],
                       **{n: [round(float(v), 2) for v in preds[n][chart_idx]] for n in chart_names}},
        },
    }
    EXPORT_PATH.write_text(json.dumps(payload, indent=1))
    log(f"wrote {EXPORT_PATH}")
    return payload


if __name__ == "__main__":
    out = run()
    for r in out["forecasts"] + out["references"]:
        print(f"{r['name']:<48} MAE {r['mae_ms']:.3f}  RMSE {r['rmse_ms']:.3f}  bias {r['bias_ms']:+.3f}  R2 {r['r2']:.3f}")
