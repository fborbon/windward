"""Runs every technique in models.py against the same farm/split, scores each against the true
test-period production, and returns a tidy results table + the raw prediction series (used by
export.py for the dashboard payload). Also usable standalone: `python -m wind_prediction.evaluate`.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wind_prediction import models
from wind_prediction.data import chronological_split, evaluated_index, load_series


def _mape(y_true, y_pred):
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def _score(name, family, y_true, y_pred, results):
    y_pred = y_pred.reindex(y_true.index)
    results.append({
        "name": name, "family": family,
        "mae_mw": mean_absolute_error(y_true, y_pred),
        "rmse_mw": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
        "mape": _mape(y_true.values, y_pred.values),
    })


def run(farm_id: str = "kelmarsh", dl_epochs: int = 25, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")

    def log(msg):
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"loading + splitting series for {farm_id}")
    df = load_series(farm_id)
    train_df, test_df = chronological_split(df)
    idx = evaluated_index(test_df)
    y_true = test_df.loc[idx, "output_mw"]

    results, predictions = [], {}

    log("naive / baseline")
    for tname, series in models.naive_baselines(train_df, test_df).items():
        predictions[tname] = series
        _score(tname, "Naive / baseline", y_true, series, results)

    log("classical statistical: SARIMA (this one takes a while - one fit + 68 incremental updates)")
    series = models.sarima_forecast(train_df, test_df)
    predictions["SARIMA"] = series
    _score("SARIMA", "Classical statistical", y_true, series, results)

    log("exponential smoothing: Holt-Winters")
    ets_series, ets_models = models.ets_forecast(train_df, test_df)
    predictions["Holt-Winters (ETS)"] = ets_series
    _score("Holt-Winters (ETS)", "Exponential smoothing", y_true, ets_series, results)

    log("state-space: structural time series (Kalman filter)")
    series = models.state_space_forecast(train_df, test_df)
    predictions["Structural TS (Kalman)"] = series
    _score("Structural TS (Kalman)", "State-space", y_true, series, results)

    log("multivariate statistical: VAR")
    series = models.var_forecast(train_df, test_df)
    predictions["VAR"] = series
    _score("VAR (output + wind speed)", "Multivariate statistical", y_true, series, results)

    log("classical ML: HistGradientBoosting (recursive)")
    series = models.ml_recursive_forecast(train_df, test_df)
    predictions["Gradient Boosting (recursive)"] = series
    _score("Gradient Boosting (recursive)", "Classical ML", y_true, series, results)

    log("deep learning: LSTM (training...)")
    series = models.lstm_forecast(train_df, test_df, epochs=dl_epochs)
    predictions["LSTM (seq2seq)"] = series
    _score("LSTM (seq2seq)", "Deep learning", y_true, series, results)

    log("attention/transformer: compact Transformer encoder (training...)")
    series = models.transformer_forecast(train_df, test_df, epochs=dl_epochs)
    predictions["Transformer (encoder)"] = series
    _score("Transformer (encoder)", "Attention / Transformer", y_true, series, results)

    log("foundation model: Chronos-Bolt-Tiny (zero-shot, no training)")
    series = models.chronos_forecast(train_df, test_df)
    predictions["Chronos-Bolt-Tiny (zero-shot)"] = series
    _score("Chronos-Bolt-Tiny (zero-shot)", "Foundation model", y_true, series, results)

    log("hybrid: Holt-Winters + Gradient Boosting residual correction")
    series = models.hybrid_forecast(train_df, test_df, ets_series, ets_models)
    predictions["Hybrid (ETS + GBM residual)"] = series
    _score("Hybrid (ETS + GBM residual)", "Hybrid", y_true, series, results)

    log("done")
    results_df = pd.DataFrame(results).set_index("name").sort_values("mae_mw")
    return {"results": results_df, "predictions": predictions, "y_true": y_true, "test_df": test_df, "train_df": train_df}


if __name__ == "__main__":
    out = run()
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(out["results"])
