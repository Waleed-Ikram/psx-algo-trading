"""
All 9 features implemented in this file

"""

import pandas as pd
import pandas_ta as ta


def generate_features(df):
    """
    Adds 9 technical-indicator columns to a cleaned OHLCV DataFrame:
    RSI(14), MACD(12,26,9), SMA20, SMA50, BB_width, Return_1d,
    Momentum_5d, Momentum_20d, Volume_ratio.

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
