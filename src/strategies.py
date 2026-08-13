"""
Trading strategies: abstract base class, classical strategies
(momentum, mean reversion), and ML-based strategy wrappers.
"""

import numpy as np


def momentum_signals(df):
    """
    Trend-following signal using RSI, MACD and 20-day momentum together.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features added by generate_features(), containing
        at least 'RSI', 'MACD' and 'Momentum_20d'.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with a new 'Signal' column:
        1 = buy/hold long, -1 = exit long position and move to cash
        (long-only system — no short selling), 0 = no position.

    Notes
    -----
    All three indicators (RSI, MACD, Momentum_20d) must agree before a
    position is taken, which filters out noisy single-indicator signals.
    Each day's signal uses only that day's closing values (no shift) —
    the backtester is responsible for executing it at the next day's
    open.
    """
    df = df.copy()

    buy = (df["RSI"] > 55) & (df["MACD"] > 0) & (df["Momentum_20d"] > 0)
    sell = (df["RSI"] < 45) & (df["MACD"] < 0) & (df["Momentum_20d"] < 0)

    df["Signal"] = np.select([buy, sell], [1, -1], default=0)
    return df


def mean_reversion_signals(df):
    """
    Counter-trend signal using RSI and 5-day momentum (pullback size).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features added by generate_features(), containing
        at least 'RSI' and 'Momentum_5d'.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with a new 'Signal' column:
        1 = buy/hold long, -1 = exit long position and move to cash
        (long-only system — no short selling), 0 = no position.

    Notes
    -----
    Buys stocks that look oversold and have pulled back sharply, sells
    stocks that look overbought and have run up sharply. Each day's
    signal uses only that day's closing values (no shift) — the
    backtester is responsible for executing it at the next day's open.
    """
    df = df.copy()

    buy = (df["RSI"] < 35) & (df["Momentum_5d"] < -0.03)
    sell = (df["RSI"] > 65) & (df["Momentum_5d"] > 0.03)

    df["Signal"] = np.select([buy, sell], [1, -1], default=0)
    return df


if __name__ == "__main__":
    import pandas as pd

    from src.data import load_data
    from src.features import generate_features

    pd.set_option("display.width", 200)

    kel = generate_features(load_data("KEL"))

    momentum = momentum_signals(kel)
    mean_rev = mean_reversion_signals(kel)

    print("Momentum signal counts:")
    print(momentum["Signal"].value_counts())

    print("\nMean reversion signal counts:")
    print(mean_rev["Signal"].value_counts())

    momentum_cols = ["close", "RSI", "MACD", "Momentum_20d", "Signal"]
    print("\nMomentum — last 5 rows:")
    print(momentum[momentum_cols].tail(5))

    mean_rev_cols = ["close", "RSI", "Momentum_5d", "Signal"]
    print("\nMean reversion — last 5 rows:")
    print(mean_rev[mean_rev_cols].tail(5))
