"""
Predictive models: ARIMA/GARCH baselines, XGBoost/LightGBM tree
models, and PyTorch LSTM/GRU deep learning models.

ARIMA section
-------------
Order (1,0,1): daily log returns on PSX stocks show no unit root (they
are already a return series, not a price level), so the differencing
term is left at d=0. A single AR lag and a single MA lag is the
simplest specification capable of capturing the short-lived
autocorrelation that thin, less-efficient emerging-market order flow
can produce, without overfitting a walk-forward loop that already
refits thousands of times. This is deliberately a simple baseline
against which the ML strategies are compared, not a tuned forecaster
(no grid search over (p,d,q) is performed here).

Walk-forward expanding window: at each day T, ARIMA(1,0,1) is refit
using only log returns observed up to that day (an expanding window
starting at min_train_days), then forecasts one day ahead. The window
never includes any data beyond T, so there is no look-ahead bias —
this mirrors the walk-forward discipline used everywhere else in this
project. The refit-every-day approach is computationally expensive
but is the correct way to simulate how the model would actually have
been used in production.

Option A caching: refitting ARIMA thousands of times per stock is slow
(each fit is tens to hundreds of milliseconds; a full history is
1,000+ fits), and the resulting signal series never changes for a
given ticker/history/parameter combination. Rather than refit on every
backtest run, arima_signals() is run once per ticker and its output is
cached to results/arima_signals/TICKER_arima.csv. The backtester then
reads signals via load_arima_signals() instead of ever calling
arima_signals() again. run_arima_all() drives this cache: it skips any
ticker whose CSV already exists, so re-running it is idempotent.

Dead-band rule: a raw ARIMA forecast is almost never exactly zero, so
acting on its sign alone would generate a position change most days —
churning on noise the model has no real conviction in and paying
transaction costs for it. The dead_band (default 0.001, i.e. 0.1%)
requires the forecast log return to exceed a minimum magnitude before
a directional signal (1 or -1) is issued; anything smaller is treated
as "no edge" and mapped to 0.

XGBoost section
---------------
Monthly refitting (refit_every=21 trading days): a single XGBoost fit
is milliseconds, not the tens-to-hundreds of milliseconds an ARIMA fit
costs, so it could technically be refit daily too. Refitting monthly
instead is a deliberate design choice, not a performance workaround —
it more realistically simulates how a tree model would actually be
deployed (nobody retrains a production model every day on a handful of
extra rows), and it keeps the walk-forward loop to ~90 distinct models
per ticker instead of ~1,850, which is far easier to inspect when
interpreting which periods the model's behaviour changed in. Between
refits, the most recently trained model is simply reused to score new
rows as they arrive.

Probability threshold dead-band: predict_proba() returns a continuous
probability of an up move, which — like ARIMA's raw forecast — is
almost never exactly 0.5. A signal only fires when that probability
clears 0.55 (long) or drops below 0.45 (short); everything inside that
five-point band on either side of "no idea" is treated as no edge and
mapped to 0, for the same reason as ARIMA's dead_band: avoid trading on
marginal, low-conviction predictions.

Fixed hyperparameters (no tuning): n_estimators=100 gives enough trees
to capture signal without needing early stopping; max_depth=3 keeps
trees shallow, since daily financial returns are noisy and deep trees
would just memorise that noise; subsample=0.8 and colsample_bytree=0.8
regularise via row/column sampling, standard practice for
gradient-boosted trees on tabular data; random_state=42 makes every
refit reproducible given the same training slice. No grid search or
cross-validation is run over these — walk-forward already refits ~90
times per ticker, and tuning inside that loop would multiply the
runtime for what this project treats as a baseline comparison point,
not a production model. This is a limitation, noted explicitly here
rather than hidden: reported XGBoost performance is a lower bound on
what a tuned tree model could achieve.
"""

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
    Generate next-day ARIMA(1,0,1) trading signals via walk-forward refit.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned DataFrame with features already generated, indexed by
        date, containing at least a 'close' column.
    min_train_days : int
        Minimum number of log-return observations required before the
        first forecast is made, default 504 (~2 years of trading days).
        Rows before this point are left at signal = 0.
    dead_band : float
        Minimum absolute forecast log return required to generate a
        directional signal, default 0.001 (0.1%). See module docstring.
    ticker : str, optional
        Ticker symbol, used only to make the progress prints
        informative (e.g. "ARIMA fitting: row 800 / 2354 (KEL)") and to
        name the cached output CSV. If omitted, progress prints show
        "UNKNOWN" and the CSV cache step is skipped.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with a new 'Signal' column added:
        1 (predicted return > dead_band), -1 (predicted return <
        -dead_band), 0 (inside the dead band, still warming up, or the
        ARIMA fit failed to converge that day). The Signal on row T is
        the prediction made using data up to T-1, valid for execution
        on T+1 (next-day execution) — see src.backtest.run_backtest.

    Notes
    -----
    ARIMA is refit from scratch at every step on an expanding window
    (log_returns.iloc[0:T]), so no information from day T or later is
    ever used to produce day T's signal — this is walk-forward, not
    k-fold, validation. If a fit fails to converge on a given day, that
    day's signal falls back to 0 and the loop continues rather than
    raising.
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
        print("No ticker given — skipping CSV cache save (Option A).")

    return df


def load_arima_signals(ticker, results_dir=ARIMA_SIGNALS_DIR):
    """
    Load cached ARIMA signals for a ticker (Option A caching read path).

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "KEL". Reads results_dir/TICKER_arima.csv,
        as written by arima_signals().
    results_dir : str
        Directory containing cached ARIMA signal CSVs.

    Returns
    -------
    pandas.Series
        'Signal' values (1, -1, 0) indexed by date. This is what the
        backtester should use instead of calling arima_signals() again.

    Raises
    ------
    FileNotFoundError
        If results_dir/TICKER_arima.csv does not exist.
    """
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
    """
    Populate the ARIMA signal cache for every ticker in stock_dict.

    Parameters
    ----------
    stock_dict : dict of {str: pandas.DataFrame}
        Cleaned OHLCV(+features) data per ticker, e.g. from
        src.data.load_selected().
    min_train_days : int
        Passed through to arima_signals() for each ticker not already
        cached.
    dead_band : float
        Passed through to arima_signals() for each ticker not already
        cached.
    results_dir : str
        Directory to check for existing caches and to write new ones.

    Returns
    -------
    pandas.DataFrame
        One row per ticker in stock_dict, indexed by ticker, with
        columns: 'Status' ('computed', 'skipped', or 'failed'),
        'Time (s)' (float, NaN if skipped), 'Buy (1)', 'Sell (-1)',
        'Zero (0)' (signal counts, NaN if skipped or failed), and
        'Error' (str, only populated if status == 'failed'). Also
        printed as a table (this is Step 4's cache-population summary,
        not a metrics run — compute_metrics() is never called here).

    Notes
    -----
    A ticker is skipped if results_dir/TICKER_arima.csv already exists
    — this makes the function idempotent to re-run and is the whole
    point of Option A caching: ARIMA is only ever fit once per ticker.
    A ticker whose fit raises (as opposed to an individual day's ARIMA
    fit failing, which arima_signals() already handles internally) is
    recorded as 'failed' rather than aborting the whole batch, so one
    bad ticker doesn't lose progress on the other 24.
    """
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
    Generate next-day XGBoost direction signals via walk-forward refit.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features already generated by generate_features()
        — must contain all 9 columns in XGBOOST_FEATURE_COLS, plus
        'close'.
    min_train_days : int
        Minimum number of rows required before the first prediction,
        default 504 (~2 years of trading days). Rows before this point
        are left at signal = 0.
    refit_every : int
        How often to retrain the model, in trading days, default 21
        (~monthly). Between refits, the most recently trained model is
        reused to score new rows. See module docstring for why monthly
        rather than daily.
    ticker : str, optional
        Ticker symbol, used only to make the progress prints
        informative and to name the cached output CSV. If omitted,
        progress prints show "UNKNOWN" and the CSV cache step is
        skipped.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with a new 'Signal' column added: 1 (P(up)
        > 0.55), -1 (P(up) < 0.45), 0 (inside the dead band, still
        warming up, or the model/refit failed that day). The Signal on
        row T is the prediction made using features available at T and
        a model trained only on labels known by T-1, valid for
        execution on T+1 — see src.backtest.run_backtest.

    Notes
    -----
    The label for row i (next_day_return[i] > 0) depends on close[i+1],
    so it is only known once day i+1 has happened. At walk-forward step
    T, the training slice therefore uses rows 0..T-1 (never row T or
    later) — the model is never trained on, nor does it ever see
    features for, the day it is about to predict. This mirrors
    arima_signals()'s log_returns.iloc[0:T] window exactly, just with a
    classification label instead of a continuous forecast. If a refit
    or a prediction fails, that day's signal falls back to 0 and the
    loop continues rather than raising.
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
                # Rows 0..T-1 only: label i needs close[i+1], so using
                # this slice never trains on information from day T.
                train_slice = pd.concat(
                    [features_full.iloc[0:T], y_full.iloc[0:T].rename("y")],
                    axis=1,
                ).dropna()

                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    # use_label_encoder was removed from xgboost's core
                    # in newer releases (installed here: 3.3.0) — kept
                    # per spec, silently ignored, suppressed below so it
                    # doesn't warn on every one of ~90 refits.
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
        print("No ticker given — skipping CSV cache save (Option A).")

    return df


def load_xgboost_signals(ticker, results_dir=XGBOOST_SIGNALS_DIR):
    """
    Load cached XGBoost signals for a ticker (Option A caching read path).

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "KEL". Reads results_dir/TICKER_xgboost.csv,
        as written by xgboost_signals().
    results_dir : str
        Directory containing cached XGBoost signal CSVs.

    Returns
    -------
    pandas.Series
        'Signal' values (1, -1, 0) indexed by date.

    Raises
    ------
    FileNotFoundError
        If results_dir/TICKER_xgboost.csv does not exist.
    """
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
    """
    Populate the XGBoost signal cache for every ticker in stock_dict.

    Parameters
    ----------
    stock_dict : dict of {str: pandas.DataFrame}
        Cleaned OHLCV+features data per ticker, e.g. from
        src.data.load_selected() run through generate_features().
    min_train_days : int
        Passed through to xgboost_signals() for each ticker not already
        cached.
    refit_every : int
        Passed through to xgboost_signals() for each ticker not already
        cached.
    results_dir : str
        Directory to check for existing caches and to write new ones.

    Returns
    -------
    pandas.DataFrame
        One row per ticker in stock_dict, indexed by ticker, with
        columns 'Status' ('computed', 'skipped', or 'failed'),
        'Time (s)', 'Buy (1)', 'Sell (-1)', 'Zero (0)', and 'Error'.
        Also printed as a table: Ticker | Time (s) | Buy (1) |
        Sell (-1) | Zero (0).

    Notes
    -----
    Same pattern as run_arima_all(): a ticker is skipped if
    results_dir/TICKER_xgboost.csv already exists, and a ticker whose
    fit raises is recorded as 'failed' rather than aborting the batch.
    """
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
