"""
Tests for feature engineering (src/features.py: generate_features).

Covers:
- All 9 documented feature columns are present in the output.
- No feature column uses future data: truncating the input DataFrame
  at row N and recomputing features must produce identical values for
  rows 0..N-1 as computing features on the full DataFrame and slicing
  afterwards. If any feature depended on rows >= N (e.g. a centred
  rolling window, or a scaler/statistic fit on the whole series), this
  equality would break.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import generate_features

EXPECTED_FEATURE_COLS = [
    "RSI", "MACD", "SMA20", "SMA50", "BB_width",
    "Return_1d", "Momentum_5d", "Momentum_20d", "Volume_ratio",
]


def _make_synthetic_ohlcv(n=150, seed=42):
    """
    Synthetic daily OHLCV data, long enough to clear every feature's
    warm-up window (SMA50 needs 50 rows; MACD(12,26,9) needs ~35).
    Only 'close', 'volume' and 'DailyReturn' are actually read by
    generate_features(), but open/high/low are included for realism.
    """
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + daily_returns)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(1_000, 100_000, n).astype(float)

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    df["DailyReturn"] = df["close"].pct_change().round(6)
    return df


def test_generate_features_returns_all_expected_columns():
    df = _make_synthetic_ohlcv()
    result = generate_features(df)

    for col in EXPECTED_FEATURE_COLS:
        assert col in result.columns, f"missing expected feature column: {col}"


def test_features_do_not_use_future_data():
    """
    Truncating the input at row N must not change any feature value for
    rows 0..N-1 relative to computing features on the full series. Every
    feature is a function of the current row and rows before it only.
    """
    df = _make_synthetic_ohlcv(n=150)
    N = 100

    full_result = generate_features(df)
    truncated_result = generate_features(df.iloc[:N])

    full_head = full_result.iloc[:N][EXPECTED_FEATURE_COLS]
    truncated_head = truncated_result[EXPECTED_FEATURE_COLS]

    pd.testing.assert_frame_equal(
        full_head, truncated_head, check_exact=False, rtol=1e-8, atol=1e-10,
    )


def test_features_do_not_use_future_data_at_multiple_cutoffs():
    """
    Same look-ahead check as above, repeated at a few different
    truncation points, to reduce the chance a single lucky cutoff masks
    a leak that only shows up near specific window boundaries.
    """
    df = _make_synthetic_ohlcv(n=150)
    full_result = generate_features(df)

    for N in (60, 90, 120, 149):
        truncated_result = generate_features(df.iloc[:N])
        full_head = full_result.iloc[:N][EXPECTED_FEATURE_COLS]
        truncated_head = truncated_result[EXPECTED_FEATURE_COLS]

        pd.testing.assert_frame_equal(
            full_head, truncated_head, check_exact=False, rtol=1e-8, atol=1e-10,
        )
