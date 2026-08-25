"""Classical trading strategies: momentum and mean reversion are written in this file"""

import numpy as np


def momentum_signals(df):
    """
    Signal: 1 = buy/hold, -1 = exit to cash, 0 = no position. Uses today's close only — the
    backtester executes it at tomorrow's open.
    """
    df = df.copy()

    buy = (df["RSI"] > 55) & (df["MACD"] > 0) & (df["Momentum_20d"] > 0)
    sell = (df["RSI"] < 45) & (df["MACD"] < 0) & (df["Momentum_20d"] < 0)

    df["Signal"] = np.select([buy, sell], [1, -1], default=0)
    return df


def mean_reversion_signals(df):
    """
    buy stocks that look oversold and have pulled back
    sharply, sell stocks that look overbought and have run up sharply
    (RSI + 5-day momentum).

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
