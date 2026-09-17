"""One function per taxonomy family (see taxonomy.py) - each takes the same (train_df, test_df)
frames (wind_prediction.data.load_series + chronological_split) and returns a pandas Series of
24h-ahead forecasts, indexed to test_df's positions, covering wind_prediction.data.evaluated_index.

Every function follows the same no-leakage rule: forecasting hours [origin, origin+24) may use
real data strictly before `origin` (and, for weather-feature-driven models, the true weather
*at* the forecast hours themselves - standing in for forecast weather inputs, exactly like
forecasting/pipeline.py's production model), but never true output_mw values from inside the
window, and never another technique's predictions.
"""
import numpy as np
import pandas as pd

from forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN
from wind_prediction.data import HORIZON, LOOKBACK, evaluated_index, window_origins

TARGET = TARGET_COLUMN


def _combined(train_df: pd.DataFrame, test_df: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.concat([train_df[cols], test_df[cols]], ignore_index=True)


def _empty_pred() -> pd.Series:
    return pd.Series(dtype=float)


# --------------------------------------------------------------------------- naive / baseline
def naive_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.Series]:
    combined = _combined(train_df, test_df, [TARGET])[TARGET]
    n_train = len(train_df)
    persistence, seasonal_naive, moving_average = {}, {}, {}
    for origin in window_origins(len(test_df)):
        g = n_train + origin # global position of the forecast origin
        last_value = combined.iloc[g - 1]
        trailing_mean = combined.iloc[g - 24:g].mean()
        for h in range(HORIZON):
            pos = origin + h
            persistence[pos] = last_value
            seasonal_naive[pos] = combined.iloc[g - 24 + h] # value exactly 24h before this hour
            moving_average[pos] = trailing_mean
    idx = evaluated_index(test_df)
    return {
        "Persistence (last value)": pd.Series(persistence).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx],
        "Seasonal naive (t-24h)": pd.Series(seasonal_naive).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx],
        "Moving average (24h)": pd.Series(moving_average).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx],
    }


# --------------------------------------------------------------------------- classical statistical
def sarima_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    import statsmodels.api as sm

    n_train = len(train_df)
    y_train = train_df[TARGET].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)
    y_test.index = y_test.index + n_train

    model = sm.tsa.SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 24),
                            enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=50)

    preds = {}
    observed_upto = 0
    for origin in window_origins(len(test_df)):
        new_segment = y_test.iloc[observed_upto:origin]
        if len(new_segment):
            res = res.append(new_segment, refit=False)
        forecast = res.get_forecast(HORIZON).predicted_mean.values
        for h in range(HORIZON):
            preds[origin + h] = forecast[h]
        observed_upto = origin
    idx = evaluated_index(test_df)
    return pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]


# --------------------------------------------------------------------------- exponential smoothing
def ets_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, refit_every: int = 1) -> tuple[pd.Series, dict]:
    """Returns (predictions, {origin: fitted_model}) - the fitted models are reused by
    hybrid_forecast so the hybrid technique doesn't refit the statistical half from scratch."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    n_train = len(train_df)
    combined = _combined(train_df, test_df, [TARGET])[TARGET]

    preds, models_by_origin = {}, {}
    fitted = None
    for i, origin in enumerate(window_origins(len(test_df))):
        history = combined.iloc[: n_train + origin].reset_index(drop=True)
        if fitted is None or i % refit_every == 0:
            fitted = ExponentialSmoothing(
                history, trend="add", seasonal="add", seasonal_periods=24, initialization_method="estimated",
            ).fit()
        models_by_origin[origin] = fitted
        forecast = fitted.forecast(HORIZON).values
        for h in range(HORIZON):
            preds[origin + h] = forecast[h]
    idx = evaluated_index(test_df)
    series = pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]
    return series, models_by_origin


# --------------------------------------------------------------------------- state-space (Kalman)
def state_space_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    n_train = len(train_df)
    y_train = train_df[TARGET].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)
    y_test.index = y_test.index + n_train

    model = UnobservedComponents(y_train, level="local level", seasonal=24)
    res = model.fit(disp=False, maxiter=50)

    preds = {}
    observed_upto = 0
    for origin in window_origins(len(test_df)):
        new_segment = y_test.iloc[observed_upto:origin]
        if len(new_segment):
            res = res.append(new_segment, refit=False)
        forecast = res.forecast(HORIZON).values
        for h in range(HORIZON):
            preds[origin + h] = forecast[h]
        observed_upto = origin
    idx = evaluated_index(test_df)
    return pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]


# --------------------------------------------------------------------------- multivariate statistical
def var_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, maxlags: int = 6) -> pd.Series:
    from statsmodels.tsa.api import VAR

    cols = [TARGET, "wind_speed_ms"]
    n_train = len(train_df)
    combined = _combined(train_df, test_df, cols)

    var_res = VAR(train_df[cols].reset_index(drop=True)).fit(maxlags=maxlags)
    k_ar = var_res.k_ar

    preds = {}
    for origin in window_origins(len(test_df)):
        g = n_train + origin
        history = combined.iloc[g - k_ar:g][cols].values
        forecast = var_res.forecast(history, steps=HORIZON)
        target_col_idx = cols.index(TARGET)
        for h in range(HORIZON):
            preds[origin + h] = forecast[h, target_col_idx]
    idx = evaluated_index(test_df)
    return pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]


# --------------------------------------------------------------------------- classical ML
_ML_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
_ML_WEATHER_COLS = ["wind_speed_ms", "wind_speed_cubed", "wind_direction_deg", "temperature_c",
                     "pressure_hpa", "air_density_kg_m3", "hour_of_day"]


def _ml_feature_row(extended: list[float], weather_row: pd.Series) -> list[float]:
    lags = [extended[-lag] for lag in _ML_LAGS]
    return lags + [weather_row[c] for c in _ML_WEATHER_COLS]


def ml_recursive_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingRegressor

    cols = [TARGET] + _ML_WEATHER_COLS
    n_train = len(train_df)
    combined = _combined(train_df, test_df, cols)

    # 1-step-ahead supervised training set, built from real lag values only (train period).
    max_lag = max(_ML_LAGS)
    rows, targets = [], []
    train_target = combined[TARGET].iloc[:n_train].tolist()
    for t in range(max_lag, n_train):
        # extended = history strictly before t, so _ml_feature_row's "lag_1" is train_target[t-1] -
        # matching the recursive predict loop below exactly (there, extended also ends one
        # position before the one being forecast). Using train_target[:t+1] here would leak the
        # target itself into its own lag_1 feature.
        rows.append(_ml_feature_row(train_target[:t], combined.iloc[t]))
        targets.append(train_target[t])
    X_train = pd.DataFrame(rows, columns=[f"lag_{lag}" for lag in _ML_LAGS] + _ML_WEATHER_COLS)
    y_train = pd.Series(targets)

    model = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    preds = {}
    for origin in window_origins(len(test_df)):
        g = n_train + origin
        extended = combined[TARGET].iloc[:g].tolist() # true history only, up to this window's origin
        for h in range(HORIZON):
            row = _ml_feature_row(extended, combined.iloc[g + h])
            X = pd.DataFrame([row], columns=X_train.columns)
            pred = float(model.predict(X)[0])
            preds[origin + h] = pred
            extended.append(pred) # recursive: this step's prediction feeds the next lag_1..lag_h
    idx = evaluated_index(test_df)
    return pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]


# --------------------------------------------------------------------------- deep learning / attention
_DL_FEATURE_COLS = [TARGET, "wind_speed_ms", "wind_speed_cubed", "temperature_c", "pressure_hpa", "hour_of_day"]


def _make_sequences(values: np.ndarray, lookback: int, horizon: int):
    """values: (N, F) with TARGET in column 0. Returns X (n, lookback, F), y (n, horizon)."""
    X, y = [], []
    for t in range(lookback, len(values) - horizon + 1):
        X.append(values[t - lookback:t])
        y.append(values[t:t + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _train_sequence_model(model, train_df: pd.DataFrame, mean, std, epochs: int = 25, lr: float = 1e-3):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    values = ((train_df[_DL_FEATURE_COLS].values - mean) / std).astype(np.float32)
    X, y = _make_sequences(values, LOOKBACK, HORIZON)
    y_raw = y * std[0] + mean[0] # train against real MW scale, features stay normalized
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y_raw)), batch_size=64, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def _sequence_predict(model, train_df: pd.DataFrame, test_df: pd.DataFrame, mean, std) -> pd.Series:
    import torch

    combined = _combined(train_df, test_df, _DL_FEATURE_COLS)
    n_train = len(train_df)
    norm = ((combined.values - mean) / std).astype(np.float32)

    preds = {}
    with torch.no_grad():
        for origin in window_origins(len(test_df)):
            g = n_train + origin
            window = norm[g - LOOKBACK:g]
            x = torch.from_numpy(window).unsqueeze(0)
            forecast = model(x).squeeze(0).numpy()
            for h in range(HORIZON):
                preds[origin + h] = float(forecast[h])
    idx = evaluated_index(test_df)
    return pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]


def lstm_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = 25) -> pd.Series:
    import torch
    import torch.nn as nn

    class LSTMForecaster(nn.Module):
        def __init__(self, n_features, hidden=64, horizon=HORIZON):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, horizon)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])

    mean = train_df[_DL_FEATURE_COLS].values.mean(axis=0)
    std = train_df[_DL_FEATURE_COLS].values.std(axis=0) + 1e-6
    torch.manual_seed(42)
    model = LSTMForecaster(n_features=len(_DL_FEATURE_COLS))
    model = _train_sequence_model(model, train_df, mean, std, epochs=epochs)
    return _sequence_predict(model, train_df, test_df, mean, std)


def transformer_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = 25) -> pd.Series:
    import math

    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=LOOKBACK):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerForecaster(nn.Module):
        def __init__(self, n_features, d_model=32, nhead=4, layers=2, horizon=HORIZON):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            self.pos = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Linear(d_model, horizon)

        def forward(self, x):
            z = self.pos(self.proj(x))
            z = self.encoder(z)
            return self.head(z.mean(dim=1)) # mean-pool over the lookback window

    mean = train_df[_DL_FEATURE_COLS].values.mean(axis=0)
    std = train_df[_DL_FEATURE_COLS].values.std(axis=0) + 1e-6
    torch.manual_seed(42)
    model = TransformerForecaster(n_features=len(_DL_FEATURE_COLS))
    model = _train_sequence_model(model, train_df, mean, std, epochs=epochs)
    return _sequence_predict(model, train_df, test_df, mean, std)


# --------------------------------------------------------------------------- foundation model (zero-shot)
def chronos_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, context_len: int = 512) -> pd.Series:
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu")
    n_train = len(train_df)
    combined = _combined(train_df, test_df, [TARGET])[TARGET]

    preds = {}
    for origin in window_origins(len(test_df)):
        g = n_train + origin
        context = combined.iloc[max(0, g - context_len):g].values.astype("float32")
        _, mean = pipeline.predict_quantiles(
            inputs=torch.tensor(context), prediction_length=HORIZON, quantile_levels=[0.5],
        )
        forecast = mean.squeeze(0).numpy()
        for h in range(HORIZON):
            preds[origin + h] = float(forecast[h])
    idx = evaluated_index(test_df)
    return pd.Series(preds).reindex(range(len(test_df))).set_axis(test_df.index).loc[idx]


# --------------------------------------------------------------------------- hybrid
def hybrid_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     ets_series: pd.Series, ets_models_by_origin: dict) -> pd.Series:
    """Holt-Winters owns trend/seasonality (reuses ets_forecast's already-fitted models);
    HistGradientBoosting is trained on the *train-period residuals* against weather features and
    corrects for the exogenous, weather-driven part Holt-Winters can't see at all."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    first_model = next(iter(ets_models_by_origin.values()))
    fitted_len = len(first_model.fittedvalues)
    residual_train = train_df[TARGET].iloc[:fitted_len].values - first_model.fittedvalues.values

    X_train = train_df[_ML_WEATHER_COLS].iloc[:fitted_len]
    gbm = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
    gbm.fit(X_train, residual_train)

    idx = evaluated_index(test_df)
    residual_correction = gbm.predict(test_df.loc[idx, _ML_WEATHER_COLS])
    return ets_series + pd.Series(residual_correction, index=idx)
