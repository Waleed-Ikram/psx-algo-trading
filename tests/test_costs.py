"""Test for commission/slippage handling in the backtest engine."""

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


def test_round_trip_cost():
    """
    Buys then sells at the same price, using our real fees (0.15%
    commission + 0.1% slippage), and checks the loss matches what
    those fees should actually cost. The fee is taken twice — once on
    the buy, once on the sell — so the two losses stack up instead of
    just adding together.
    """
    commission, slippage = 0.0015, 0.001
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
