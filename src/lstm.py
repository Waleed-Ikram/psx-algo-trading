"""
PyTorch LSTM deep-learning model for next-day direction prediction.

Architecture rationale
-----------------------
A single-layer, 32-unit LSTM is used deliberately small: this project's
walk-forward windows are ~500-2000 rows of daily data per stock, and a
larger or deeper network (more hidden units, more layers) would have far
more parameters than there is signal in that much data, memorising noise
in training and doing so from scratch again every refit_every window (no
persistent training state across refits by walk-forward design). 32 units
is enough capacity to learn short sequences of technical-indicator
dynamics without overfitting on limited financial data.

Sequence length (seq_len=20)
------------------------------
20 trading days is one calendar month, matching refit_every=21 — the
model is asked to summarise "the last month's worth of feature dynamics"
into a single prediction, the same horizon at which it is retrained. This
also mirrors XGBoost_signals' feature set (the 9 same technical
indicators), so any performance difference between the two models comes
from the temporal (sequence) structure the LSTM can exploit and the tree
model cannot, not from different inputs.

Monthly retraining (refit_every=21)
-------------------------------------
Same cadence as xgboost_signals() in src.models, and for the same reason:
refitting a neural network daily would not reflect how such a model would
realistically be deployed and would multiply an already expensive
training loop (each refit trains a fresh LSTMClassifier from scratch —
walk-forward requires this, since the model must never see day T's label
before predicting day T) for little practical benefit.

StandardScaler fitted on training data only
----------------------------------------------
As with xgboost_signals(), the scaler is fit once per refit window on
features_full.iloc[0:T] (rows 0..T-1) only, then used to transform both
the training sequences and the prediction window. This prevents
look-ahead bias entering through normalisation statistics (mean/std)
computed with knowledge of future rows.

Dead-band (dead_band=0.02, i.e. thresholds 0.52 / 0.48)
-----------------------------------------------------------
The model's sigmoid output is a continuous probability of an up move,
and only predictions that clear a two-point band around "no idea" (0.5)
are turned into a directional signal; everything inside the band is
treated as no edge and mapped to 0. This is narrower than xgboost_signals'
five-point band (0.55/0.45): on KEL, the LSTM's raw probabilities
clustered close enough to 0.5 that the wider band collapsed every day to
signal -1 or 0, never 1 — a narrower band was needed for the model's
actual (less separated) probability distribution to produce both
directional signals at all.

Early stopping (patience=5, min_delta=1e-4)
-----------------------------------------------
Each refit trains for up to epochs=50 iterations, but stops as soon as
training loss fails to improve by more than 1e-4 over 5 consecutive
epochs — this avoids wasting computation once a refit has converged
(most refits converge well before 50 epochs on such a small dataset)
without changing the architecture or loss function.

Fixed random seed (torch.manual_seed(42))
---------------------------------------------
Set once per call to lstm_signals(), matching XGBoost's random_state=42:
weight initialisation and dropout are otherwise stochastic, and a fixed
seed makes a given ticker's walk-forward run reproducible.
"""

import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

LSTM_SIGNALS_DIR = "results/lstm_signals"
LSTM_FEATURE_COLS = [
    "RSI", "MACD", "SMA20", "SMA50", "BB_width",
    "Return_1d", "Momentum_5d", "Momentum_20d", "Volume_ratio",
]


class LSTMClassifier(nn.Module):
    """
    Single-layer LSTM binary classifier for next-day price direction.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep, default 9 (the same 9
        technical indicators used by xgboost_signals(), see
        LSTM_FEATURE_COLS).
    hidden_size : int
        Number of LSTM hidden units, default 32 — kept small to avoid
        overfitting on the limited amount of daily financial data
        available per walk-forward training window (see module
        docstring).
    num_layers : int
        Number of stacked LSTM layers, default 1.
    dropout : float
        Dropout probability, default 0.2. Applied after the LSTM's last
        hidden state, before the final linear layer; also passed to
        nn.LSTM's internal dropout, but only takes effect there when
        num_layers > 1 (PyTorch requirement — a single layer has no
        inter-layer connection to drop out).

    Notes
    -----
    Input shape is (batch, sequence, features) because batch_first=True.
    Only the last timestep's hidden state (lstm_out[:, -1, :]) is used —
    the model produces one prediction per sequence (next-day direction),
    not a prediction per timestep, so the final hidden state is the
    summary of the whole 20-day window that matters.
    """

    def __init__(self, input_size=9, hidden_size=32,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return self.sigmoid(out)


def build_sequences(X, y, seq_len):
    """
    Convert tabular (rows x features) arrays into overlapping sequences.

    Parameters
    ----------
    X : numpy.ndarray
        Shape (n_rows, n_features).
    y : numpy.ndarray
        Shape (n_rows,), binary direction labels aligned with X's rows.
    seq_len : int
        Number of past rows per sequence.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        X_sequences of shape (n_rows - seq_len, seq_len, n_features) and
        y_labels of shape (n_rows - seq_len,). For output index j
        (0-indexed), X_sequences[j] = X[j : j + seq_len] and
        y_labels[j] = y[j + seq_len] — i.e. the label for the day
        immediately after the window, never a day inside it.
    """
    n_rows = X.shape[0]
    X_sequences = np.array(
        [X[i - seq_len:i] for i in range(seq_len, n_rows)]
    )
    y_labels = np.array(
        [y[i] for i in range(seq_len, n_rows)]
    )
    return X_sequences, y_labels


def train_lstm(X_train_seq, y_train, hidden_size=32, epochs=50,
               lr=0.001, patience=5, min_delta=1e-4):
    """
    Fit a fresh LSTMClassifier on one walk-forward training window.

    Parameters
    ----------
    X_train_seq : numpy.ndarray
        Shape (n_sequences, seq_len, n_features).
    y_train : numpy.ndarray
        Shape (n_sequences,), binary labels.
    hidden_size : int
        Passed through to LSTMClassifier.
    epochs : int
        Maximum number of full passes over X_train_seq. No
        mini-batching is used — the training set for a single refit
        window is small enough to fit in one forward/backward pass per
        epoch. Training may stop earlier than `epochs`; see `patience`.
    lr : float
        Adam optimiser learning rate.
    patience : int
        Stop training early once training loss has failed to improve
        by more than `min_delta` for this many consecutive epochs —
        avoids wasting computation once a refit has converged.
    min_delta : float
        Minimum decrease in training loss, versus the best loss seen
        so far, that counts as "improvement" for early stopping.

    Returns
    -------
    torch.nn.Module
        The trained LSTMClassifier, in eval() mode.
    """
    model = LSTMClassifier(
        input_size=X_train_seq.shape[2], hidden_size=hidden_size,
    )

    X_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    epochs_no_improve = 0

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.eval()
    return model


def lstm_signals(df, ticker=None, min_train_days=504, seq_len=20,
                  refit_every=21, hidden_size=32, epochs=50,
                  learning_rate=0.001, dead_band=0.02):
    """
    Generate next-day LSTM direction signals via walk-forward refit.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features already generated by generate_features()
        — must contain all 9 columns in LSTM_FEATURE_COLS, plus 'close'.
    ticker : str, optional
        Ticker symbol, used only to make the progress prints
        informative and to name the cached output CSV. If omitted,
        progress prints show "UNKNOWN" and the CSV cache step is
        skipped.
    min_train_days : int
        Minimum number of rows required before the first prediction,
        default 504 (~2 years of trading days). Rows before this point
        are left at signal = 0.
    seq_len : int
        Sequence length fed to the LSTM, default 20 (~one trading
        month, see module docstring).
    refit_every : int
        How often to retrain the model, in trading days, default 21
        (~monthly, matching xgboost_signals()). Between refits, the
        most recently trained model is reused to score new rows.
    hidden_size : int
        LSTM hidden units, passed through to LSTMClassifier.
    epochs : int
        Maximum training epochs per refit, passed through to
        train_lstm() (default 50; early stopping — see train_lstm's
        `patience`/`min_delta` — usually halts a refit before this).
    learning_rate : float
        Adam optimiser learning rate, passed through to train_lstm().
    dead_band : float
        Probability distance from 0.5 required to generate a
        directional signal, default 0.02 (thresholds 0.52 / 0.48).
        See module docstring for why this is narrower than
        xgboost_signals' band.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with a new 'Signal' column added: 1 (P(up)
        > 0.5 + dead_band), -1 (P(up) < 0.5 - dead_band), 0 (inside the
        dead band, still warming up, or the model/refit failed that
        day). The Signal on row T is the prediction made using features
        available at T and a model trained only on labels known by
        T-1, valid for execution on T+1 — see src.backtest.run_backtest.

    Notes
    -----
    Rows are first dropped of any NaNs across LSTM_FEATURE_COLS (early
    rows haven't satisfied every indicator's lookback window yet); the
    walk-forward loop and min_train_days/refit_every counters operate on
    this NaN-free frame, and signals are mapped back onto the original
    index at the end (rows dropped as NaN, and rows before min_train_days,
    keep signal = 0). At walk-forward step T, both the model's training
    slice and the StandardScaler are fit on rows 0..T-1 only (never row T
    or later) — mirroring xgboost_signals()'s look-ahead discipline
    exactly. If a refit or a prediction fails, that day's signal falls
    back to 0 and the loop continues rather than raising.
    """
    torch.manual_seed(42)

    df = df.copy()
    ticker_label = ticker if ticker is not None else "UNKNOWN"

    clean_df = df.dropna(subset=LSTM_FEATURE_COLS)
    original_index = clean_df.index

    features_full = clean_df[LSTM_FEATURE_COLS].to_numpy()
    next_day_return = clean_df["close"].shift(-1) / clean_df["close"] - 1
    y_full = (next_day_return > 0).astype(int).to_numpy()

    n = len(clean_df)
    signal_values = np.zeros(n, dtype=int)

    total_refits = len(range(min_train_days, n, refit_every))
    refit_count = 0
    model = None
    scaler = None
    last_refit_probs = []

    for T in range(min_train_days, n):
        try:
            if (T - min_train_days) % refit_every == 0:
                X_train_raw = features_full[0:T]
                y_train_raw = y_full[0:T]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_raw)

                X_train_seq, y_train_seq = build_sequences(
                    X_train_scaled, y_train_raw, seq_len,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = train_lstm(
                        X_train_seq, y_train_seq, hidden_size=hidden_size,
                        epochs=epochs, lr=learning_rate,
                    )

                last_refit_probs = []
                refit_count += 1
                if refit_count % 5 == 0:
                    print(
                        f"LSTM refit {refit_count}/{total_refits} "
                        f"({ticker_label})"
                    )

            window_raw = features_full[T - seq_len:T]
            window_scaled = scaler.transform(window_raw)
            window_tensor = torch.tensor(
                window_scaled, dtype=torch.float32,
            ).view(1, seq_len, -1)

            with torch.no_grad():
                prob_up = model(window_tensor).item()
            last_refit_probs.append(prob_up)
        except Exception:
            signal_values[T] = 0
            continue

        if prob_up > 0.5 + dead_band:
            signal_values[T] = 1
        elif prob_up < 0.5 - dead_band:
            signal_values[T] = -1
        else:
            signal_values[T] = 0

    clean_signals = pd.Series(signal_values, index=original_index)
    df["Signal"] = clean_signals.reindex(df.index, fill_value=0)

    if not (signal_values == 1).any():
        if last_refit_probs:
            probs_arr = np.array(last_refit_probs)
            print(
                f"No Signal=1 produced ({ticker_label}) — raw probability "
                f"diagnostics from the last refit: min={probs_arr.min():.4f}, "
                f"max={probs_arr.max():.4f}, mean={probs_arr.mean():.4f}, "
                f"std={probs_arr.std():.4f}"
            )
        else:
            print(
                f"No Signal=1 produced ({ticker_label}) — no probabilities "
                f"were recorded from the last refit to diagnose."
            )

    if ticker is not None:
        out_dir = Path(LSTM_SIGNALS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_lstm.csv"
        df[["Signal"]].to_csv(out_path, index_label="date")
    else:
        print("No ticker given — skipping CSV cache save (Option A).")

    return df


def load_lstm_signals(ticker, results_dir=LSTM_SIGNALS_DIR):
    """
    Load cached LSTM signals for a ticker (Option A caching read path).

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "KEL". Reads results_dir/TICKER_lstm.csv,
        as written by lstm_signals().
    results_dir : str
        Directory containing cached LSTM signal CSVs.

    Returns
    -------
    pandas.Series
        'Signal' values (1, -1, 0) indexed by date.

    Raises
    ------
    FileNotFoundError
        If results_dir/TICKER_lstm.csv does not exist.
    """
    path = Path(results_dir) / f"{ticker}_lstm.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached LSTM signals found for '{ticker}' at {path}"
        )

    signals = pd.read_csv(path, index_col="date", parse_dates=True)
    signals.index = pd.DatetimeIndex(signals.index).tz_localize(None)
    return signals["Signal"]


def run_lstm_all(stock_dict, min_train_days=504, seq_len=20,
                  refit_every=21, hidden_size=32, epochs=20,
                  learning_rate=0.001, dead_band=0.05,
                  results_dir=LSTM_SIGNALS_DIR):
    """
    Populate the LSTM signal cache for every ticker in stock_dict.

    Parameters
    ----------
    stock_dict : dict of {str: pandas.DataFrame}
        Cleaned OHLCV+features data per ticker, e.g. from
        src.data.load_selected() run through generate_features().
    min_train_days, seq_len, refit_every, hidden_size, epochs,
    learning_rate, dead_band : see lstm_signals().
        Passed through to lstm_signals() for each ticker not already
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
    Same pattern as run_arima_all() / run_xgboost_all(): a ticker is
    skipped if results_dir/TICKER_lstm.csv already exists, and a ticker
    whose fit raises is recorded as 'failed' rather than aborting the
    batch.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now()
    start_time = time.time()
    print(f"Start time: {start_dt:%Y-%m-%d %H:%M:%S}")

    summary_rows = []

    for ticker, df in stock_dict.items():
        out_path = out_dir / f"{ticker}_lstm.csv"

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
            result_df = lstm_signals(
                df, ticker=ticker, min_train_days=min_train_days,
                seq_len=seq_len, refit_every=refit_every,
                hidden_size=hidden_size, epochs=epochs,
                learning_rate=learning_rate, dead_band=dead_band,
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
        f"run_lstm_all summary: {n_computed} computed, "
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
    cache_path = Path(LSTM_SIGNALS_DIR) / f"{ticker}_lstm.csv"

    start_dt = datetime.now()
    start_time = time.time()
    print(f"Start time: {start_dt:%Y-%m-%d %H:%M:%S}")

    kel = generate_features(load_data(ticker))

    if cache_path.exists():
        print(f"Cache already exists at {cache_path}, skipping LSTM fit.")
    else:
        print(f"No cache found at {cache_path}, running LSTM walk-forward fit.")
        lstm_signals(kel, ticker=ticker)

    kel_lstm_signal = load_lstm_signals(ticker)

    print("\nLSTM signal value counts (KEL):")
    print(kel_lstm_signal.value_counts())

    kel_with_signal = kel.copy()
    kel_with_signal["Signal"] = kel_lstm_signal
    kel_results = run_backtest(kel_with_signal, kel_with_signal)

    metrics = compute_metrics(kel_results)

    print("\nKEL — LSTM strategy metrics:\n")
    for name, value in metrics.items():
        print(f"{name:<25}: {value}")

    end_dt = datetime.now()
    elapsed = time.time() - start_time
    elapsed_minutes, elapsed_seconds = divmod(elapsed, 60)

    print(f"\nEnd time: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total elapsed time: {int(elapsed_minutes)}m {elapsed_seconds:.1f}s")
    print(f"Total time taken (KEL): {elapsed:.1f}s")
