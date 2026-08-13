"""
Tests for transaction cost / slippage handling in the backtest engine
(src/backtest.py: run_backtest).

Covers:
- Commission and slippage are both applied on the buy leg and the sell
  leg (not just one side).
- A round trip at an unchanged price loses exactly the round-trip cost
  fraction implied by the engine's mechanics: cost is charged on trade
  value on each leg, so it compounds — final cash after a flat-price
  round trip is initial_capital * (1 - cost_fraction) ** 2, i.e. a loss
  fraction of 1 - (1 - cost_fraction) ** 2 (not simply 2 * cost_fraction).
"""

import pandas as pd
import pytest

from src.backtest import run_backtest


def _make_frames(prices_open, prices_close, signals, start="2024-01-01"):
    n = len(prices_open)
    dates = pd.date_range(start, periods=n, freq="D")
    prices_df = pd.DataFrame(
        {"open": prices_open, "close": prices_close}, index=dates
    )
    signals_df = pd.DataFrame(
        {"Signal": signals, "Tradeable": [True] * n}, index=dates
    )
    return signals_df, prices_df


def test_commission_and_slippage_applied_on_buy():
    """
    Buying with cost_fraction = commission + slippage should cost
    exactly cash * cost_fraction, leaving shares_held =
    (cash - trade_cost) / open_price.
    """
    commission, slippage = 0.01, 0.002
    cost_fraction = commission + slippage

    signals_df, prices_df = _make_frames(
        prices_open=[10, 10],
        prices_close=[10, 10],
        signals=[1, 0],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000,
        commission=commission, slippage=slippage,
    )

    day1 = result.iloc[1]
    expected_cost = 100000 * cost_fraction
    expected_shares = (100000 - expected_cost) / 10

    assert bool(day1["trade_executed"]) is True
    assert day1["trade_cost"] == pytest.approx(expected_cost)
    assert day1["shares_held"] == pytest.approx(expected_shares)
    assert day1["cash"] == pytest.approx(0.0)


def test_commission_and_slippage_applied_on_sell():
    """
    Selling with cost_fraction = commission + slippage should cost
    exactly gross_proceeds * cost_fraction, leaving cash = gross_proceeds
    - trade_cost.
    """
    commission, slippage = 0.01, 0.002
    cost_fraction = commission + slippage

    signals_df, prices_df = _make_frames(
        prices_open=[10, 10, 10],
        prices_close=[10, 10, 10],
        signals=[1, -1, 0],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000,
        commission=commission, slippage=slippage,
    )

    shares_bought = result.iloc[1]["shares_held"]

    day2 = result.iloc[2]
    gross_proceeds = shares_bought * 10
    expected_cost = gross_proceeds * cost_fraction
    expected_cash = gross_proceeds - expected_cost

    assert bool(day2["trade_executed"]) is True
    assert day2["trade_cost"] == pytest.approx(expected_cost)
    assert day2["cash"] == pytest.approx(expected_cash)
    assert day2["shares_held"] == pytest.approx(0.0)


def test_zero_cost_round_trip_is_lossless():
    """
    Sanity check: with commission=slippage=0, a round trip at an
    unchanged price must leave the portfolio exactly at its starting
    value (isolates the cost mechanics tested below from any other
    engine behaviour).
    """
    signals_df, prices_df = _make_frames(
        prices_open=[10, 10, 10],
        prices_close=[10, 10, 10],
        signals=[1, -1, 0],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000, commission=0, slippage=0,
    )

    assert result.iloc[2]["cash"] == pytest.approx(100000.0)


@pytest.mark.parametrize("commission,slippage", [(0.0015, 0.001), (0.01, 0.002)])
def test_round_trip_at_unchanged_price_loses_exactly_cost_fraction(commission, slippage):
    """
    Buy and sell at the same price (10 -> 10 -> 10, no market move).
    Cost is charged on trade *value* on each leg, so the two legs
    compound multiplicatively rather than adding linearly:

        cash_after_buy  = initial_capital * (1 - cost_fraction)
        cash_after_sell = cash_after_buy   * (1 - cost_fraction)
                        = initial_capital * (1 - cost_fraction) ** 2

    The realised loss fraction is therefore exactly
    1 - (1 - cost_fraction) ** 2, not 2 * cost_fraction.
    """
    cost_fraction = commission + slippage
    initial_capital = 100000

    signals_df, prices_df = _make_frames(
        prices_open=[10, 10, 10],
        prices_close=[10, 10, 10],
        signals=[1, -1, 0],
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=initial_capital,
        commission=commission, slippage=slippage,
    )

    final_cash = result.iloc[2]["cash"]
    expected_final_cash = initial_capital * (1 - cost_fraction) ** 2
    expected_loss_fraction = 1 - (1 - cost_fraction) ** 2

    assert final_cash == pytest.approx(expected_final_cash)

    realised_loss_fraction = (initial_capital - final_cash) / initial_capital
    assert realised_loss_fraction == pytest.approx(expected_loss_fraction)
