"""Tests for the core backtest engine: the 5-day example, and next-day execution timing."""

import pandas as pd
import pytest

from src.backtest import run_backtest


def _make_frames(prices_open, prices_close, signals, tradeable=None, start="2024-01-01"):
    n = len(prices_open)
    dates = pd.date_range(start, periods=n, freq="D")
    if tradeable is None:
        tradeable = [True] * n

    prices_df = pd.DataFrame(
        {"open": prices_open, "close": prices_close}, index=dates
    )
    signals_df = pd.DataFrame(
        {"Signal": signals, "Tradeable": tradeable}, index=dates
    )
    return signals_df, prices_df


def test_five_day_example():
    """Runs the 5-day example by hand and checks the engine gets the same numbers."""
    signals_df, prices_df = _make_frames(
        prices_open=[10, 11, 12, 11, 13],
        prices_close=[10, 11, 12, 11, 13],
        signals=[1, 1, -1, 0, 0],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000, commission=0, slippage=0
    )

    expected = [
        100000.0,
        100000.0,
        100000.0 / 11 * 12,
        100000.0,
        100000.0,
    ]

    assert result["portfolio_value"].tolist() == pytest.approx(expected, abs=1e-6)


def test_trade_happens_next_day():
    """
    Makes sure a trade uses tomorrow's opening price, not today's
    closing price. Day1 has a very different open (50) and close (200)
    on purpose, so the two cases give clearly different answers: using
    the open buys 2000 shares (value 400,000); using the close would
    buy 500 shares (value stays 100,000). If this test used the close
    by mistake, we'd see it right away.
    """
    signals_df, prices_df = _make_frames(
        prices_open=[100, 50],
        prices_close=[100, 200],
        signals=[1, 0],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000, commission=0, slippage=0
    )

    day0 = result.iloc[0]
    day1 = result.iloc[1]

    assert bool(day0["trade_executed"]) is False
    assert day0["portfolio_value"] == pytest.approx(100000.0)

    assert bool(day1["trade_executed"]) is True
    assert day1["shares_held"] == pytest.approx(100000.0 / 50.0)
    assert day1["portfolio_value"] == pytest.approx(400000.0)
