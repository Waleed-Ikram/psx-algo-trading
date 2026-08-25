import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

ARIMA_SIGNALS_DIR = "results/arima_signals"
XGBOOST_SIGNALS_DIR = "results/xgboost_signals"
XGBOOST_FEATURE_COLS = [
    "RSI", "MACD", "SMA20", "SMA50", "BB_width",
    "Return_1d", "Momentum_5d", "Momentum_20d", "Volume_ratio",
]


def arima_signals(df, min_train_days=504, dead_band=0.001, ticker=None):
    """
    Walk-forward ARIMA(1,0,1) signals: 1 if the forecast return beats
    +dead_band, -1 if below -dead_band, else 0. Refit at every step on
    an expanding window ending the day before, so nothing from day T
    onward ever informs day T's signal.
    """
    df = df.copy()
    ticker_label = ticker if ticker is not None else "UNKNOWN"

    log_returns = np.log(df["close"]).diff().dropna()

    n = len(df)
    signal_values = np.zeros(n, dtype=int)

    for T in range(min_train_days, n):
        train_data = log_returns.iloc[0:T]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train_data, order=(1, 0, 1)).fit()
                forecast = model.forecast(steps=1).iloc[0]
        except Exception:
            signal_values[T] = 0
        else:
            if forecast > dead_band:
                signal_values[T] = 1
            elif forecast < -dead_band:
                signal_values[T] = -1
            else:
                signal_values[T] = 0

        if (T - min_train_days) % 200 == 0:
            print(f"ARIMA fitting: row {T} / {n} ({ticker_label})")

    df["Signal"] = signal_values

    if ticker is not None:
        out_dir = Path(ARIMA_SIGNALS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_arima.csv"
        df[["Signal"]].to_csv(out_path, index_label="date")
    else:
        print("No ticker given — skipping CSV cache save.")

    return df


def load_arima_signals(ticker, results_dir=ARIMA_SIGNALS_DIR):
    path = Path(results_dir) / f"{ticker}_arima.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached ARIMA signals found for '{ticker}' at {path}"
        )

    signals = pd.read_csv(path, index_col="date", parse_dates=True)
    signals.index = pd.DatetimeIndex(signals.index).tz_localize(None)
    return signals["Signal"]


def run_arima_all(stock_dict, min_train_days=504, dead_band=0.001,
                   results_dir=ARIMA_SIGNALS_DIR):

    #Populate the ARIMA cache for every ticker

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now()
    start_time = time.time()
    print(f"Start time: {start_dt:%Y-%m-%d %H:%M:%S}")

    summary_rows = []

    for ticker, df in stock_dict.items():
        out_path = out_dir / f"{ticker}_arima.csv"

        if out_path.exists():
            print(f"Skipping {ticker} — cache already exists at {out_path}")
            summary_rows.append({
                "Ticker": ticker, "Status": "skipped", "Time (s)": np.nan,
                "Buy (1)": np.nan, "Sell (-1)": np.nan, "Zero (0)": np.nan,
                "Error": "",
            })
            continue

        print(f"Running {ticker}...")
        ticker_start = time.time()
        try:
            result_df = arima_signals(
                df, min_train_days=min_train_days, dead_band=dead_band,
                ticker=ticker,
            )
        except Exception as exc:
            ticker_elapsed = time.time() - ticker_start
            print(f"FAILED {ticker} after {ticker_elapsed:.1f}s: {exc}")
            summary_rows.append({
                "Ticker": ticker, "Status": "failed",
                "Time (s)": round(ticker_elapsed, 1), "Buy (1)": np.nan,
                "Sell (-1)": np.nan, "Zero (0)": np.nan, "Error": str(exc),
            })
            continue

        ticker_elapsed = time.time() - ticker_start
        counts = result_df["Signal"].value_counts()
        print(f"Done {ticker} in {ticker_elapsed:.1f}s")
        summary_rows.append({
            "Ticker": ticker, "Status": "computed",
            "Time (s)": round(ticker_elapsed, 1),
            "Buy (1)": int(counts.get(1, 0)),
            "Sell (-1)": int(counts.get(-1, 0)),
            "Zero (0)": int(counts.get(0, 0)),
            "Error": "",
        })

    end_dt = datetime.now()
    elapsed = time.time() - start_time
    elapsed_minutes, elapsed_seconds = divmod(elapsed, 60)

    summary = pd.DataFrame(summary_rows).set_index("Ticker")
    n_computed = int((summary["Status"] == "computed").sum())
    n_skipped = int((summary["Status"] == "skipped").sum())
    n_failed = int((summary["Status"] == "failed").sum())

    print(f"\nEnd time: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total elapsed time: {int(elapsed_minutes)}m {elapsed_seconds:.1f}s")
    print(
        f"run_arima_all summary: {n_computed} computed, "
        f"{n_skipped} skipped, {n_failed} failed"
    )

    print("\nSummary table:")
    print(summary.to_string())

    failed = summary[summary["Status"] == "failed"]
    skipped = summary[summary["Status"] == "skipped"]
    print(
        "\nFailed tickers:",
        ", ".join(failed.index.tolist()) if not failed.empty else "none",
    )
    print(
        "Skipped tickers:",
        ", ".join(skipped.index.tolist()) if not skipped.empty else "none",
    )

    return summary


def xgboost_signals(df, min_train_days=504, refit_every=21, ticker=None):
    """
    Predicts next-day price direction and turns it into a trading signal.

    Signal is 1 (buy) when the model is at least 55% confident the
    price will go up, -1 (sell) when it is at least 55% confident the
    price will go down, and 0 when it isn't confident enough either way.

    The model is retrained every `refit_every` days, and it only ever
    trains on data from before the day it is predicting, so it can
    never see the future. If a prediction fails for any reason, the
    signal falls back to 0.

    """
    df = df.copy()
    ticker_label = ticker if ticker is not None else "UNKNOWN"

    features_full = df[XGBOOST_FEATURE_COLS]
    next_day_return = df["close"].shift(-1) / df["close"] - 1
    y_full = (next_day_return > 0).astype(int)

    n = len(df)
    signal_values = np.zeros(n, dtype=int)

    total_refits = len(range(min_train_days, n, refit_every))
    refit_count = 0
    model = None

    for T in range(min_train_days, n):
        try:
            if (T - min_train_days) % refit_every == 0:
                # rows 0..T-1 only — never trains on day T
                train_slice = pd.concat(
                    [features_full.iloc[0:T], y_full.iloc[0:T].rename("y")],
                    axis=1,
                ).dropna()

                model = XGBClassifier(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=1.0,
                    # deprecated in this xgboost version; harmless, just
                    # silenced below so it doesn't warn on every refit.
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        train_slice[XGBOOST_FEATURE_COLS], train_slice["y"]
                    )

                refit_count += 1
                if refit_count % 5 == 0:
                    print(
                        f"XGBoost refit {refit_count}/{total_refits} "
                        f"({ticker_label})"
                    )

            proba_up = model.predict_proba(features_full.iloc[[T]])[0, 1]
        except Exception:
            signal_values[T] = 0
            model = None
            continue

        if proba_up > 0.55:
            signal_values[T] = 1
        elif proba_up < 0.45:
            signal_values[T] = -1
        else:
            signal_values[T] = 0

    df["Signal"] = signal_values

    if ticker is not None:
        out_dir = Path(XGBOOST_SIGNALS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_xgboost.csv"
        df[["Signal"]].to_csv(out_path, index_label="date")
    else:
        print("No ticker given — skipping CSV cache save.")

    return df


def load_xgboost_signals(ticker, results_dir=XGBOOST_SIGNALS_DIR):
    """Load a ticker's cached XGBoost signal (Series of 1/-1/0 by date)."""
    path = Path(results_dir) / f"{ticker}_xgboost.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached XGBoost signals found for '{ticker}' at {path}"
        )

    signals = pd.read_csv(path, index_col="date", parse_dates=True)
    signals.index = pd.DatetimeIndex(signals.index).tz_localize(None)
    return signals["Signal"]


def run_xgboost_all(stock_dict, min_train_days=504, refit_every=21,
                     results_dir=XGBOOST_SIGNALS_DIR):
    """Same pattern as run_arima_all(), for XGBoost."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now()
    start_time = time.time()
    print(f"Start time: {start_dt:%Y-%m-%d %H:%M:%S}")

    summary_rows = []

    for ticker, df in stock_dict.items():
        out_path = out_dir / f"{ticker}_xgboost.csv"

        if out_path.exists():
            print(f"Skipping {ticker} — cache already exists at {out_path}")
            summary_rows.append({
                "Ticker": ticker, "Status": "skipped", "Time (s)": np.nan,
                "Buy (1)": np.nan, "Sell (-1)": np.nan, "Zero (0)": np.nan,
                "Error": "",
            })
            continue

        print(f"Running {ticker}...")
        ticker_start = time.time()
        try:
            result_df = xgboost_signals(
                df, min_train_days=min_train_days, refit_every=refit_every,
                ticker=ticker,
            )
        except Exception as exc:
            ticker_elapsed = time.time() - ticker_start
            print(f"FAILED {ticker} after {ticker_elapsed:.1f}s: {exc}")
            summary_rows.append({
                "Ticker": ticker, "Status": "failed",
                "Time (s)": round(ticker_elapsed, 1), "Buy (1)": np.nan,
                "Sell (-1)": np.nan, "Zero (0)": np.nan, "Error": str(exc),
            })
            continue

        ticker_elapsed = time.time() - ticker_start
        counts = result_df["Signal"].value_counts()
        print(f"Done {ticker} in {ticker_elapsed:.1f}s")
        summary_rows.append({
            "Ticker": ticker, "Status": "computed",
            "Time (s)": round(ticker_elapsed, 1),
            "Buy (1)": int(counts.get(1, 0)),
            "Sell (-1)": int(counts.get(-1, 0)),
            "Zero (0)": int(counts.get(0, 0)),
            "Error": "",
        })

    end_dt = datetime.now()
    elapsed = time.time() - start_time
    elapsed_minutes, elapsed_seconds = divmod(elapsed, 60)

    summary = pd.DataFrame(summary_rows).set_index("Ticker")
    n_computed = int((summary["Status"] == "computed").sum())
    n_skipped = int((summary["Status"] == "skipped").sum())
    n_failed = int((summary["Status"] == "failed").sum())

    print(f"\nEnd time: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total elapsed time: {int(elapsed_minutes)}m {elapsed_seconds:.1f}s")
    print(
        f"run_xgboost_all summary: {n_computed} computed, "
        f"{n_skipped} skipped, {n_failed} failed"
    )

    print("\nSummary table:")
    print(summary[["Time (s)", "Buy (1)", "Sell (-1)", "Zero (0)"]].to_string())

    failed = summary[summary["Status"] == "failed"]
    skipped = summary[summary["Status"] == "skipped"]
    print(
        "\nFailed tickers:",
        ", ".join(failed.index.tolist()) if not failed.empty else "none",
    )
    print(
        "Skipped tickers:",
        ", ".join(skipped.index.tolist()) if not skipped.empty else "none",
    )

    return summary


if __name__ == "__main__":
    from src.backtest import run_backtest
    from src.data import load_data
    from src.evaluation import compute_metrics
    from src.features import generate_features

    ticker = "KEL"
    cache_path = Path(ARIMA_SIGNALS_DIR) / f"{ticker}_arima.csv"

    start_dt = datetime.now()
    start_time = time.time()
    print(f"Start time: {start_dt:%Y-%m-%d %H:%M:%S}")

    kel = generate_features(load_data(ticker))

    if cache_path.exists():
        print(f"Cache already exists at {cache_path}, skipping ARIMA fit.")
    else:
        print(f"No cache found at {cache_path}, running ARIMA walk-forward fit.")
        arima_signals(kel, ticker=ticker)

    kel_arima_signal = load_arima_signals(ticker)

    print("\nARIMA signal value counts (KEL):")
    print(kel_arima_signal.value_counts())

    kel_with_signal = kel.copy()
    kel_with_signal["Signal"] = kel_arima_signal
    kel_results = run_backtest(kel_with_signal, kel_with_signal)

    metrics = compute_metrics(kel_results)

    print("\nKEL — ARIMA strategy metrics:\n")
    for name, value in metrics.items():
        print(f"{name:<25}: {value}")

    end_dt = datetime.now()
    elapsed = time.time() - start_time
    elapsed_minutes, elapsed_seconds = divmod(elapsed, 60)

    print(f"\nEnd time: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total elapsed time: {int(elapsed_minutes)}m {elapsed_seconds:.1f}s")
    print(f"Total time taken (KEL): {elapsed:.1f}s")

    # --- XGBoost, KEL only ---
    xgb_cache_path = Path(XGBOOST_SIGNALS_DIR) / f"{ticker}_xgboost.csv"

    xgb_start_dt = datetime.now()
    xgb_start_time = time.time()
    print(f"\n\nStart time: {xgb_start_dt:%Y-%m-%d %H:%M:%S}")

    if xgb_cache_path.exists():
        print(f"Cache already exists at {xgb_cache_path}, skipping XGBoost fit.")
    else:
        print(
            f"No cache found at {xgb_cache_path}, "
            f"running XGBoost walk-forward fit."
        )
        xgboost_signals(kel, ticker=ticker)

    kel_xgboost_signal = load_xgboost_signals(ticker)

    print("\nXGBoost signal value counts (KEL):")
    print(kel_xgboost_signal.value_counts())

    kel_with_xgb_signal = kel.copy()
    kel_with_xgb_signal["Signal"] = kel_xgboost_signal
    kel_xgb_results = run_backtest(kel_with_xgb_signal, kel_with_xgb_signal)

    xgb_metrics = compute_metrics(kel_xgb_results)

    print("\nKEL — XGBoost strategy metrics:\n")
    for name, value in xgb_metrics.items():
        print(f"{name:<25}: {value}")

    xgb_end_dt = datetime.now()
    xgb_elapsed = time.time() - xgb_start_time
    xgb_elapsed_minutes, xgb_elapsed_seconds = divmod(xgb_elapsed, 60)

    print(f"\nEnd time: {xgb_end_dt:%Y-%m-%d %H:%M:%S}")
    print(
        f"Total elapsed time: {int(xgb_elapsed_minutes)}m "
        f"{xgb_elapsed_seconds:.1f}s"
    )
    print(f"Total time taken (KEL): {xgb_elapsed:.1f}s")
