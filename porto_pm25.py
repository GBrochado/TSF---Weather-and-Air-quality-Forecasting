
# porto_pm25_forecasting_full_v3_lstm.py
# ------------------------------------------------------------
# End-to-end pipeline (clean + working):
# - Load & clean hourly dataset
# - Train: Seasonal Naive, SARIMA, SARIMAX (exog), LSTM (sequence model)
# - Evaluate horizons from test start
# - Cross-validation for ALL models via sampled rolling TEST windows (no refit)
#
# Notes:
# - SARIMAX exog is aggressively cleaned to avoid "exog contains inf or nans"
# - LSTM is trained ONCE on train split and then used recursively for multi-step forecasts
# - Requires TensorFlow (tensorflow / tensorflow-cpu). If missing, install:
#     pip install tensorflow
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    data_path: str
    season: int = 24

    horizons: Dict[str, int] = None
    test_hours: int = 24 * 184  # ~6 months

    # Statsmodels orders
    sarima_order: Tuple[int, int, int] = (1, 0, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)

    sarimax_order: Tuple[int, int, int] = (1, 0, 1)
    sarimax_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)

    # LSTM
    lstm_lookback: int = 168  # 7 days
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 30
    lstm_batch: int = 256
    lstm_patience: int = 5
    lstm_lr: float = 1e-3

    # Test-window CV
    n_windows: int = 10
    seed: int = 42
    verbose: bool = True


def default_config(data_path: str) -> Config:
    return Config(
        data_path=data_path,
        horizons={"24h": 24, "48h": 48, "7d": 168, "2weeks": 336},
    )


# -----------------------------
# Metrics / helpers
# -----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _detect_datetime_col(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in ["datetime", "timestamp", "date", "time", "dt"]:
        if cand in lower:
            return lower[cand]
    return df.columns[0]


def _detect_target_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_", "").replace(".", "")
        if "pm25" in cl or "pm2p5" in cl:
            return c
    raise ValueError("Couldn't detect PM2.5 target column (expected something containing 'pm25').")


def _ensure_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")
    df = df.sort_index()

    if df.index.has_duplicates:
        df = df.groupby(df.index).mean(numeric_only=True)

    # Force strict hourly grid (creates missing hours)
    df = df.asfreq("H")

    # If too many NaNs after asfreq, resample instead
    miss = df.isna().mean().mean() if len(df) else 0.0
    if miss > 0.05:
        df = df.resample("H").mean(numeric_only=True).asfreq("H")

    return df


def _final_impute_exog(exog: pd.DataFrame) -> pd.DataFrame:
    exog = exog.replace([np.inf, -np.inf], np.nan)

    if isinstance(exog.index, pd.DatetimeIndex):
        exog = exog.interpolate("time", limit_direction="both")
    else:
        exog = exog.interpolate(limit_direction="both")

    exog = exog.ffill().bfill()

    # Drop all-NaN cols
    all_nan_cols = exog.columns[exog.isna().all()].tolist()
    if all_nan_cols:
        exog = exog.drop(columns=all_nan_cols)

    exog = exog.fillna(0.0).astype("float64")

    # Final finite guard
    arr = exog.to_numpy()
    arr[~np.isfinite(arr)] = 0.0
    return pd.DataFrame(arr, index=exog.index, columns=exog.columns)


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=idx,
    )


# -----------------------------
# Baseline
# -----------------------------
def seasonal_naive_series(y_hist: pd.Series, idx_future: pd.DatetimeIndex, season: int = 24) -> pd.Series:
    if len(y_hist) < season:
        return pd.Series(np.repeat(float(y_hist.iloc[-1]), len(idx_future)), index=idx_future)
    last_season = y_hist.iloc[-season:]
    reps = int(np.ceil(len(idx_future) / season))
    vals = np.tile(last_season.values, reps)[:len(idx_future)]
    return pd.Series(vals, index=idx_future)


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Dataset:
    y: pd.Series
    X_exog: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame


def load_and_build_dataset(cfg: Config) -> Dataset:
    df_raw = pd.read_csv(cfg.data_path)

    dt_col = _detect_datetime_col(df_raw)
    df_raw[dt_col] = pd.to_datetime(df_raw[dt_col], errors="coerce")
    df_raw = df_raw.dropna(subset=[dt_col]).set_index(dt_col)

    df_raw = _coerce_numeric_df(df_raw)
    df = _ensure_hourly_index(df_raw)

    target_col = _detect_target_col(df)

    y = df[target_col].astype("float64").replace([np.inf, -np.inf], np.nan)
    y = y.interpolate("time", limit_direction="both").ffill().bfill()
    y = y.fillna(y.median()).astype("float64")

    exog = df.drop(columns=[target_col], errors="ignore")
    exog = _coerce_numeric_df(exog)
    exog = _final_impute_exog(exog)

    if cfg.test_hours >= len(y):
        raise ValueError(f"test_hours={cfg.test_hours} is >= dataset length n={len(y)}")

    split = len(y) - cfg.test_hours
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    X_train, X_test = exog.iloc[:split], exog.iloc[split:]

    X_train = _final_impute_exog(X_train)
    X_test = _final_impute_exog(X_test)

    return Dataset(y=y, X_exog=exog, y_train=y_train, y_test=y_test, X_train=X_train, X_test=X_test)


# -----------------------------
# LSTM model
# -----------------------------
@dataclass
class LSTMArtifacts:
    model: object
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    feature_cols: List[str]
    lookback: int


def _build_lstm_frame(y: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    # features at each timestamp include exog + calendar; y is not included as feature row itself
    cal = _calendar_features(y.index)
    X = pd.concat([exog, cal], axis=1)
    X = _final_impute_exog(_coerce_numeric_df(X))
    return X


def _make_sequences(X_scaled: np.ndarray, y_scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    # Predict y[t] from X[t-lookback:t] and y[t-lookback:t] implicitly via X? We include y as extra channel in X.
    # Here we already pack y into X as first column before scaling in fit_lstm below.
    n = len(y_scaled)
    if n <= lookback:
        raise ValueError(f"Not enough data for lookback={lookback}. Need > lookback, got n={n}")
    X_seq = np.zeros((n - lookback, lookback, X_scaled.shape[1]), dtype=np.float32)
    y_seq = np.zeros((n - lookback,), dtype=np.float32)
    for i in range(lookback, n):
        X_seq[i - lookback] = X_scaled[i - lookback:i]
        y_seq[i - lookback] = y_scaled[i]
    return X_seq, y_seq


def fit_lstm(cfg: Config, y_train: pd.Series, X_train_exog: pd.DataFrame) -> LSTMArtifacts:
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
    except Exception as e:
        raise ImportError(
            "TensorFlow is required for the LSTM model. Install with: pip install tensorflow"
        ) from e

    # Reproducibility
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build feature frame (exog + calendar)
    X_feat = _build_lstm_frame(y_train, X_train_exog)

    # Pack y as FIRST column into the LSTM input so it can learn autoregression
    frame = pd.concat([y_train.rename("y"), X_feat], axis=1)
    frame = frame.astype("float64").replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate("time", limit_direction="both").ffill().bfill().fillna(0.0)

    feature_cols = frame.columns.tolist()

    # Scale X and y separately
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(frame.values)
    y_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1)

    # Build sequences
    lookback = cfg.lstm_lookback
    X_seq, y_seq = _make_sequences(X_scaled, y_scaled, lookback=lookback)

    model = Sequential(
        [
            LSTM(cfg.lstm_units, input_shape=(lookback, X_seq.shape[2])),
            Dropout(cfg.lstm_dropout),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=cfg.lstm_lr), loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=cfg.lstm_patience, restore_best_weights=True)

    model.fit(
        X_seq,
        y_seq,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch,
        validation_split=0.1,
        callbacks=[es],
        verbose=1 if cfg.verbose else 0,
        shuffle=False,
    )

    return LSTMArtifacts(model=model, scaler_X=scaler_X, scaler_y=scaler_y,
                         feature_cols=feature_cols, lookback=lookback)


def lstm_recursive_forecast(
    art: LSTMArtifacts,
    y_hist: pd.Series,
    X_hist_exog: pd.DataFrame,
    X_future_exog: pd.DataFrame,
    idx_future: pd.DatetimeIndex,
) -> pd.Series:
    # Build combined feature frames for history and future
    X_hist_feat = _build_lstm_frame(y_hist, X_hist_exog)
    X_future_feat = _build_lstm_frame(pd.Series(index=idx_future, data=np.nan), X_future_exog)

    # We'll maintain a rolling window of the packed frame rows: [y, exog+calendar]
    # Use raw (unscaled) rows then scale via scaler_X.
    hist_frame = pd.concat([y_hist.rename("y"), X_hist_feat], axis=1).astype("float64")
    hist_frame = hist_frame.replace([np.inf, -np.inf], np.nan).interpolate("time", limit_direction="both").ffill().bfill().fillna(0.0)

    # Ensure column alignment
    hist_frame = hist_frame.reindex(columns=art.feature_cols)
    # If any missing cols appeared (shouldn't), fill zeros
    hist_frame = hist_frame.fillna(0.0)

    # Need at least lookback rows
    if len(hist_frame) < art.lookback:
        raise ValueError(f"Not enough history for LSTM lookback={art.lookback}. Have {len(hist_frame)} rows.")

    # Start rolling buffer (unscaled)
    buffer = hist_frame.iloc[-art.lookback:].copy()

    preds = []
    for t in idx_future:
        # Scale buffer
        X_scaled = art.scaler_X.transform(buffer.values)
        X_in = X_scaled.reshape(1, art.lookback, X_scaled.shape[1]).astype(np.float32)

        yhat_scaled = float(art.model.predict(X_in, verbose=0)[0, 0])
        yhat = float(art.scaler_y.inverse_transform([[yhat_scaled]])[0, 0])
        preds.append(yhat)

        # Build next raw row for time t: [yhat, exog+calendar at t]
        row = pd.concat(
            [
                pd.Series({"y": yhat}),
                X_future_feat.loc[[t]].iloc[0],
            ]
        )
        row = row.reindex(art.feature_cols).astype("float64").fillna(0.0)
        row.name = t

        # Append and roll
        buffer = pd.concat([buffer.iloc[1:], row.to_frame().T], axis=0)

    return pd.Series(preds, index=idx_future)


# -----------------------------
# Statsmodels
# -----------------------------
@dataclass
class Models:
    sarima: sm.tsa.statespace.SARIMAXResultsWrapper
    sarimax: sm.tsa.statespace.SARIMAXResultsWrapper
    lstm: LSTMArtifacts


def _align_exog_to_results(res, exog: pd.DataFrame) -> pd.DataFrame:
    names = [c for c in getattr(res.model, "exog_names", []) if c != "intercept"]
    if names:
        exog = exog.reindex(columns=names)
    return _final_impute_exog(exog)


def fit_models(cfg: Config, ds: Dataset) -> Models:
    sarima_mod = sm.tsa.statespace.SARIMAX(
        ds.y_train,
        order=cfg.sarima_order,
        seasonal_order=cfg.sarima_seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )
    sarima_res = sarima_mod.fit(disp=False)

    X_train = _final_impute_exog(ds.X_train)
    sarimax_mod = sm.tsa.statespace.SARIMAX(
        ds.y_train,
        exog=X_train,
        order=cfg.sarimax_order,
        seasonal_order=cfg.sarimax_seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )
    sarimax_res = sarimax_mod.fit(disp=False)

    lstm_art = fit_lstm(cfg, ds.y_train, ds.X_train)

    return Models(sarima=sarima_res, sarimax=sarimax_res, lstm=lstm_art)


# -----------------------------
# Evaluation: horizons from test start
# -----------------------------
def horizon_table(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[["Model", "Horizon", "MAE", "RMSE", "Baseline", "Baseline_MAE", "Baseline_RMSE"]]


def evaluate_from_test_start(cfg: Config, ds: Dataset, models: Models) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_sarima, rows_sarimax, rows_lstm = [], [], []

    X_test_aligned = _align_exog_to_results(models.sarimax, ds.X_test)

    for h_label, h in cfg.horizons.items():
        idx = ds.y_test.index[:h]
        y_true = ds.y_test.iloc[:h]
        y_hist = ds.y_train

        # baseline
        y_base = seasonal_naive_series(y_hist, idx, season=cfg.season)
        base_mae, base_rmse = mean_absolute_error(y_true, y_base), rmse(y_true, y_base)

        # SARIMA
        yhat_sarima = models.sarima.get_forecast(steps=h).predicted_mean
        yhat_sarima = pd.Series(np.asarray(yhat_sarima).ravel(), index=idx)
        rows_sarima.append({
            "Model": "SARIMA", "Horizon": h_label,
            "MAE": float(mean_absolute_error(y_true, yhat_sarima)),
            "RMSE": rmse(y_true, yhat_sarima),
            "Baseline": f"SeasonalNaive({cfg.season})",
            "Baseline_MAE": float(base_mae),
            "Baseline_RMSE": float(base_rmse),
        })

        # SARIMAX
        yhat_sarimax = models.sarimax.get_forecast(steps=h, exog=X_test_aligned.iloc[:h]).predicted_mean
        yhat_sarimax = pd.Series(np.asarray(yhat_sarimax).ravel(), index=idx)
        rows_sarimax.append({
            "Model": "SARIMAX", "Horizon": h_label,
            "MAE": float(mean_absolute_error(y_true, yhat_sarimax)),
            "RMSE": rmse(y_true, yhat_sarimax),
            "Baseline": f"SeasonalNaive({cfg.season})",
            "Baseline_MAE": float(base_mae),
            "Baseline_RMSE": float(base_rmse),
        })

        # LSTM recursive
        # History exog must cover y_hist index; use ds.X_train
        yhat_lstm = lstm_recursive_forecast(
            models.lstm,
            y_hist=y_hist,
            X_hist_exog=ds.X_train,
            X_future_exog=X_test_aligned.iloc[:h],
            idx_future=idx,
        )
        rows_lstm.append({
            "Model": f"LSTM (lookback={cfg.lstm_lookback})", "Horizon": h_label,
            "MAE": float(mean_absolute_error(y_true, yhat_lstm)),
            "RMSE": rmse(y_true, yhat_lstm),
            "Baseline": f"SeasonalNaive({cfg.season})",
            "Baseline_MAE": float(base_mae),
            "Baseline_RMSE": float(base_rmse),
        })

    return horizon_table(rows_sarima), horizon_table(rows_sarimax), horizon_table(rows_lstm)


# -----------------------------
# Test-window CV (sampled rolling windows; no refit)
# -----------------------------
def _sample_test_starts(n_test: int, h: int, n_windows: int = 10, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed + h)
    eligible = np.arange(0, n_test - h + 1)
    k = min(n_windows, len(eligible))
    return np.sort(rng.choice(eligible, size=k, replace=False))


def test_windows_cv_all_models(cfg: Config, ds: Dataset, models: Models) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    X_test_aligned = _align_exog_to_results(models.sarimax, ds.X_test)

    for h_label, h in cfg.horizons.items():
        starts = _sample_test_starts(len(ds.y_test), h, n_windows=cfg.n_windows, seed=cfg.seed)

        if cfg.verbose:
            print(f"\nHorizon {h_label} (h={h}) - windows={len(starts)}")

        for s in starts:
            idx = ds.y_test.index[s:s+h]
            y_true = ds.y_test.iloc[s:s+h]

            y_hist = pd.concat([ds.y_train, ds.y_test.iloc[:s]])

            # Baseline
            yhat = seasonal_naive_series(y_hist, idx, season=cfg.season)
            rows.append({"Model": f"Seasonal naive (lag={cfg.season})", "Horizon": h_label, "Start": int(s),
                         "MAE": float(mean_absolute_error(y_true, yhat)), "RMSE": rmse(y_true, yhat)})

            # SARIMA
            res_ext = models.sarima.append(ds.y_test.iloc[:s], refit=False) if s > 0 else models.sarima
            yhat = res_ext.get_forecast(steps=h).predicted_mean
            yhat = pd.Series(np.asarray(yhat).ravel(), index=idx)
            rows.append({"Model": "SARIMA", "Horizon": h_label, "Start": int(s),
                         "MAE": float(mean_absolute_error(y_true, yhat)), "RMSE": rmse(y_true, yhat)})

            # SARIMAX
            X_past = X_test_aligned.iloc[:s]
            res_ext = models.sarimax.append(ds.y_test.iloc[:s], exog=X_past, refit=False) if s > 0 else models.sarimax
            X_future = X_test_aligned.iloc[s:s+h]
            yhat = res_ext.get_forecast(steps=h, exog=X_future).predicted_mean
            yhat = pd.Series(np.asarray(yhat).ravel(), index=idx)
            rows.append({"Model": "SARIMAX", "Horizon": h_label, "Start": int(s),
                         "MAE": float(mean_absolute_error(y_true, yhat)), "RMSE": rmse(y_true, yhat)})

            # LSTM
            # Need exog history for y_hist index
            # Build X_hist_exog = train + test up to s
            X_hist_exog = pd.concat([ds.X_train, X_test_aligned.iloc[:s]], axis=0).reindex(y_hist.index)
            X_hist_exog = _final_impute_exog(X_hist_exog)

            yhat = lstm_recursive_forecast(
                models.lstm,
                y_hist=y_hist,
                X_hist_exog=X_hist_exog,
                X_future_exog=X_future,
                idx_future=idx,
            )
            rows.append({"Model": f"LSTM (lookback={cfg.lstm_lookback})", "Horizon": h_label, "Start": int(s),
                         "MAE": float(mean_absolute_error(y_true, yhat)), "RMSE": rmse(y_true, yhat)})

    cv = pd.DataFrame(rows)
    h_order = list(cfg.horizons.keys())

    cv_mean = cv.groupby(["Model", "Horizon"])[["MAE", "RMSE"]].mean().reset_index()
    comparison = (
        cv_mean.pivot_table(index="Model", columns="Horizon", values=["MAE", "RMSE"], aggfunc="mean")
              .reindex(columns=pd.MultiIndex.from_product([["MAE", "RMSE"], h_order]))
              .round(3)
    )
    return cv, comparison



def _fmt_sarima(order: Tuple[int,int,int], seasonal: Tuple[int,int,int,int]) -> str:
    p,d,q = order
    P,D,Q,s = seasonal
    return f"({p},{d},{q})({P},{D},{Q},{s})"


# -----------------------------
# Main
# -----------------------------
def main(cfg: Config):
    if cfg.verbose:
        print("Loading data...")
    ds = load_and_build_dataset(cfg)

    if cfg.verbose:
        print("Building dataset...")
        print(f"Train: {ds.y_train.index.min()} -> {ds.y_train.index.max()} (n={len(ds.y_train)})")
        print(f"Test : {ds.y_test.index.min()} -> {ds.y_test.index.max()} (n={len(ds.y_test)})")

    if cfg.verbose:
        print("Fitting models...")
    models = fit_models(cfg, ds)

    if cfg.verbose:
        print("\n=== Horizon evaluation from test start ===\n")

    tbl_sarima, tbl_sarimax, tbl_lstm = evaluate_from_test_start(cfg, ds, models)
    print(f"SARIMA: SARIMA{_fmt_sarima(cfg.sarima_order, cfg.sarima_seasonal_order)}")
    print(tbl_sarima.to_string(index=False))

    print(f"SARIMAX: SARIMAX{_fmt_sarima(cfg.sarimax_order, cfg.sarimax_seasonal_order)}")
    print(tbl_sarimax.to_string(index=False))

    print(f"LSTM: lookback={cfg.lstm_lookback}, units={cfg.lstm_units}, dropout={cfg.lstm_dropout}, "
          f"epochs={cfg.lstm_epochs}, batch={cfg.lstm_batch}")
    print(tbl_lstm.to_string(index=False))

    if cfg.verbose:
        print("\n=== Test-window backtesting (no refit) ===")
    cv, comparison = test_windows_cv_all_models(cfg, ds, models)

    print("\nMean scores across sampled test windows:")
    print(comparison)

    return ds, models, cv, comparison


# -----------------------------
# CLI
# -----------------------------
def _parse_tuple(s: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated ints, e.g. 1,0,1")
    return tuple(parts)  # type: ignore


def _parse_seasonal_tuple(s: str) -> Tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected 4 comma-separated ints, e.g. 1,1,1,24")
    return tuple(parts)  # type: ignore


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", dest="data_path", required=True, help="Path to CSV")
    ap.add_argument("--test_hours", type=int, default=24 * 184, help="Test length in hours")
    ap.add_argument("--season", type=int, default=24, help="Season length in hours")
    ap.add_argument("--n_windows", type=int, default=10, help="Sampled CV windows per horizon")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    ap.add_argument("--sarima_order", type=_parse_tuple, default=(1, 0, 1))
    ap.add_argument("--sarima_seasonal", type=_parse_seasonal_tuple, default=(1, 1, 1, 24))
    ap.add_argument("--sarimax_order", type=_parse_tuple, default=(1, 0, 1))
    ap.add_argument("--sarimax_seasonal", type=_parse_seasonal_tuple, default=(1, 1, 1, 24))

    ap.add_argument("--lstm_lookback", type=int, default=168)
    ap.add_argument("--lstm_units", type=int, default=64)
    ap.add_argument("--lstm_dropout", type=float, default=0.2)
    ap.add_argument("--lstm_epochs", type=int, default=30)
    ap.add_argument("--lstm_batch", type=int, default=256)
    ap.add_argument("--lstm_patience", type=int, default=5)
    ap.add_argument("--lstm_lr", type=float, default=1e-3)

    args = ap.parse_args()

    cfg = default_config(args.data_path)
    cfg.test_hours = args.test_hours
    cfg.season = args.season
    cfg.n_windows = args.n_windows
    cfg.seed = args.seed

    cfg.sarima_order = args.sarima_order
    cfg.sarima_seasonal_order = args.sarima_seasonal
    cfg.sarimax_order = args.sarimax_order
    cfg.sarimax_seasonal_order = args.sarimax_seasonal

    cfg.lstm_lookback = args.lstm_lookback
    cfg.lstm_units = args.lstm_units
    cfg.lstm_dropout = args.lstm_dropout
    cfg.lstm_epochs = args.lstm_epochs
    cfg.lstm_batch = args.lstm_batch
    cfg.lstm_patience = args.lstm_patience
    cfg.lstm_lr = args.lstm_lr

    main(cfg)
