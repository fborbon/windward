"""The field's technique taxonomy (naive baselines through foundation models), and which single
representative technique per family this module actually implements and runs against real data.

Scope is deliberately one genuinely-executed technique per family rather than every named
algorithm (e.g. SARIMA stands in for AR/MA/ARMA/ARIMA/SARIMA) - the point of this module is to
demonstrate real, working judgment across the whole field, not to ship eleven half-finished
model files. Everything under "implemented" is real code that actually runs in evaluate.py, not
a description.
"""

FAMILIES = [
    {
        "family": "Naive / baseline",
        "techniques": "Last value, seasonal naive, moving average",
        "typical_use": "Baselines",
        "implemented": "Last value (persistence), seasonal naive (t-24h), 24h moving average",
        "note": "All three are pure arithmetic on true observed history - no fitting, and any "
                "candidate model has to clear this bar to justify its own complexity.",
    },
    {
        "family": "Classical statistical",
        "techniques": "AR, MA, ARMA, ARIMA, SARIMA",
        "typical_use": "Stable univariate series",
        "implemented": "SARIMA(1,1,1)(1,0,0)[24]",
        "note": "SARIMA is the superset of AR/MA/ARMA/ARIMA - a seasonal order isn't needed to "
                "demonstrate the family, but wind production has a real (if noisy) diurnal "
                "component, so it's a fair inclusion.",
    },
    {
        "family": "Exponential smoothing",
        "techniques": "SES, Holt, Holt-Winters, ETS",
        "typical_use": "Trend/seasonality",
        "implemented": "Holt-Winters (additive trend + additive daily seasonality)",
        "note": None,
    },
    {
        "family": "State-space",
        "techniques": "Kalman filter, structural time series",
        "typical_use": "Dynamic systems, noisy signals",
        "implemented": "Structural time series (local level + daily seasonal), Kalman-filtered",
        "note": "statsmodels' UnobservedComponents - the same state-space machinery SARIMA is "
                "estimated with, but here the components (level/seasonal) are the model itself, "
                "not a means to whiten residuals for an ARMA fit.",
    },
    {
        "family": "Multivariate statistical",
        "techniques": "VAR, VECM",
        "typical_use": "Several interacting time series",
        "implemented": "VAR(6) over [output_mw, wind_speed_ms]",
        "note": "The only family here that models production and wind speed as a jointly "
                "evolving system, instead of one driving the other one-way.",
    },
    {
        "family": "Classical ML",
        "techniques": "Linear/Ridge/Lasso, Random Forest, XGBoost, LightGBM",
        "typical_use": "Forecasting with engineered features",
        "implemented": "HistGradientBoostingRegressor on lag + weather features, recursive 24h",
        "note": "The same model family forecasting/train.py registers in production (Gradient "
                "Boosting) - here trained as a 1-step forecaster and applied recursively.",
    },
    {
        "family": "Deep learning",
        "techniques": "MLP, CNN/TCN, LSTM, GRU",
        "typical_use": "Complex nonlinear temporal patterns",
        "implemented": "LSTM sequence-to-sequence (72h in -> 24h out), PyTorch",
        "note": None,
    },
    {
        "family": "Attention / Transformer",
        "techniques": "TFT, Informer, Autoformer, FEDformer, PatchTST",
        "typical_use": "Long-range dependencies",
        "implemented": "Compact Transformer-encoder forecaster (72h in -> 24h out), PyTorch",
        "note": "A from-scratch minimal encoder, not a full TFT/PatchTST implementation - same "
                "attention mechanism, sized for one farm-year of data rather than a benchmark.",
    },
    {
        "family": "Modern specialized Transformers / time-series foundation models",
        "techniques": "TimesFM, Chronos, TimeGPT, Moirai, Lag-Llama",
        "typical_use": "General-purpose / zero-shot forecasting",
        "implemented": "Amazon Chronos-Bolt (tiny), genuine zero-shot inference",
        "note": "No training at all - the pretrained model forecasts directly from each window's "
                "true recent history. TimeGPT is a paid API (no key here); Moirai/Lag-Llama are "
                "heavier multivariate/probabilistic foundation models in the same category.",
    },
    {
        "family": "Hybrid",
        "techniques": "Statistical + ML/Deep Learning",
        "typical_use": "Production forecasting",
        "implemented": "Holt-Winters (trend+seasonal) + HistGradientBoosting on the residuals",
        "note": "Common real-world pattern: let a statistical model own trend/seasonality, let "
                "an ML model correct the part driven by exogenous weather the statistical model "
                "never sees.",
    },
]
