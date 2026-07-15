"""
Feature engineering: technical indicators (RSI, MACD, momentum,
Bollinger Bands) computed from cleaned OHLCV data.
"""

import pandas as pd
import pandas_ta as ta


def generate_features(df):
    """
    Add technical-indicator feature columns to a cleaned stock DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned OHLCV data (as returned by src.data.load_data), sorted
        ascending by date, with at least an 'open', 'high', 'low', 'close',
        'volume' and 'DailyReturn' column.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with 9 new feature columns appended:

        1. RSI          — 14-day Relative Strength Index (pandas_ta)
        2. MACD          — MACD line, (12, 26, 9) (pandas_ta)
        3. SMA20         — 20-day simple moving average of Close
        4. SMA50         — 50-day simple moving average of Close
        5. BB_width      — Bollinger Band width, (upper - lower) / middle,
                            20-day window, 2 standard deviations (pandas_ta)
        6. Return_1d     — daily return (renamed from DailyReturn)
        7. Momentum_5d   — (Close / Close.shift(5)) - 1
        8. Momentum_20d  — (Close / Close.shift(20)) - 1
        9. Volume_ratio  — Volume / Volume.rolling(20).mean()

    Notes
    -----
    Every feature is computed from data available up to and including
    the current row only (rolling/shift, never centered=True), so there
    is no look-ahead bias. Early rows will contain NaNs until each
    feature's lookback window is satisfied (e.g. SMA50 needs 50 rows);
    these are left in place — dropping NaNs is left to the caller.
    """
    df = df.copy()

    df["RSI"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]

    df["SMA20"] = df["close"].rolling(20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()

    bbands = ta.bbands(df["close"], length=20, std=2)
    bb_lower = bbands.filter(like="BBL").iloc[:, 0]
    bb_middle = bbands.filter(like="BBM").iloc[:, 0]
    bb_upper = bbands.filter(like="BBU").iloc[:, 0]
    df["BB_width"] = (bb_upper - bb_lower) / bb_middle

    df["Return_1d"] = df["DailyReturn"]
    df["Momentum_5d"] = (df["close"] / df["close"].shift(5)) - 1
    df["Momentum_20d"] = (df["close"] / df["close"].shift(20)) - 1
    df["Volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    return df


if __name__ == "__main__":
    from src.data import load_data

    kel = load_data("KEL")
    kel_features = generate_features(kel)

    feature_cols = [
        "RSI", "MACD", "SMA20", "SMA50", "BB_width",
        "Return_1d", "Momentum_5d", "Momentum_20d", "Volume_ratio",
    ]

    print("Shape:", kel_features.shape)
    print("\nLast 3 rows (feature columns only):")
    print(kel_features[feature_cols].tail(3))
