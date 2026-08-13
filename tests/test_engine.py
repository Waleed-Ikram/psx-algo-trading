"""
Tests for the core backtest engine (src/backtest.py: run_backtest).

Covers:
- The hand-verified 5-day example from run_backtest's __main__ block.
- That a signal recorded on day T is only ever executed at day T+1's
  open, never on day T itself.
- That no trade occurs on a day where Tradeable is False, even if the
  prior day's signal calls for one (checked on both the entry leg and
  the exit leg).
"""

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


def test_five_day_hand_verified_example():
    """
    Prices [10, 11, 12, 11, 13], signals [1, 1, -1, 0, 0], zero costs.

    Day0: no prior signal -> stay in cash.                     = 100,000.00
    Day1: prior signal (day0) = 1 -> buy at open=11 with all
          cash; priced at close=11 (same as open) -> unchanged. = 100,000.00
    Day2: prior signal (day1) = 1, already long -> hold;
          priced at close=12: (100000 / 11) * 12                = 109,090.909091
    Day3: prior signal (day2) = -1, long -> sell at open=11
          (same price as bought at) -> back to cash, flat.      = 100,000.00
    Day4: prior signal (day3) = 0 -> hold cash.                 = 100,000.00
    """
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


def test_signal_executes_next_day_open_not_same_day():
    """
    A buy signal recorded on day T must be executed at day T+1's open,
    priced using day T+1's close for that day's portfolio value — never
    using day T's own close (which would be look-ahead) and never
    executed on day T itself.

    Day0 signal=1 (nothing to act on yet, no prior signal).
    Day1 open=50, close=200 (open and close deliberately far apart).

    If the engine correctly executes at day1's OPEN, all 100,000 cash
    buys 100000/50 = 2000 shares, priced at day1's close of 200 ->
    portfolio value = 400,000.

    If the engine incorrectly executed at day1's CLOSE instead (a
    look-ahead/execution-price bug), it would buy 100000/200 = 500
    shares, priced at the same close of 200 -> portfolio value would be
    unchanged at 100,000. The two scenarios are unambiguous to tell
    apart, which is the point of using open != close here.
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


def test_todays_own_signal_is_not_used_for_todays_trade():
    """
    Signals [0, 1, -1]: day1's own signal (1) must NOT trigger a trade
    on day1 itself — only day0's signal (0, a no-op) is available to
    act on that day. Day1's signal (1) only takes effect on day2.
    """
    signals_df, prices_df = _make_frames(
        prices_open=[10, 10, 10],
        prices_close=[10, 10, 10],
        signals=[0, 1, -1],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000, commission=0, slippage=0
    )

    assert bool(result.iloc[1]["trade_executed"]) is False
    assert result.iloc[1]["shares_held"] == 0.0

    assert bool(result.iloc[2]["trade_executed"]) is True
    assert result.iloc[2]["shares_held"] > 0.0


def test_no_trade_when_not_tradeable_on_entry_leg():
    """
    A prior signal of 1 would normally trigger a buy, but if the
    current day is marked Tradeable=False (e.g. a trading halt), no
    trade should occur and cash/shares must be unchanged.
    """
    signals_df, prices_df = _make_frames(
        prices_open=[10, 10],
        prices_close=[10, 10],
        signals=[1, 0],
        tradeable=[True, False],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000, commission=0, slippage=0
    )

    day1 = result.iloc[1]
    assert bool(day1["trade_executed"]) is False
    assert day1["shares_held"] == 0.0
    assert day1["cash"] == pytest.approx(100000.0)
    assert day1["portfolio_value"] == pytest.approx(100000.0)


def test_no_trade_when_not_tradeable_on_exit_leg():
    """
    Already long from day0's signal, and day1's own signal is -1
    (sell), but day1 is marked Tradeable=False -> the exit must be
    blocked and the position must remain open.
    """
    signals_df, prices_df = _make_frames(
        prices_open=[10, 10, 10],
        prices_close=[10, 10, 10],
        signals=[1, -1, 0],
        tradeable=[True, True, False],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000, commission=0, slippage=0
    )

    # Day1: prior signal (day0) = 1, tradeable -> buys.
    assert bool(result.iloc[1]["trade_executed"]) is True
    shares_after_buy = result.iloc[1]["shares_held"]
    assert shares_after_buy > 0.0

    # Day2: prior signal (day1) = -1, but day2 is Tradeable=False ->
    # the exit must be blocked, position stays open unchanged.
    day2 = result.iloc[2]
    assert bool(day2["trade_executed"]) is False
    assert day2["shares_held"] == pytest.approx(shares_after_buy)
    assert day2["cash"] == pytest.approx(0.0)
