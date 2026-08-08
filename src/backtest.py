"""
Backtesting engine: core daily-loop P&L engine, transaction cost
and slippage modelling, and walk-forward validation splitting.
"""

import numpy as np
import pandas as pd


def run_backtest(signals_df, prices_df, initial_capital=100000,
                  commission=0.003, slippage=0.001):
    """
    Run a single-asset, long-only backtest with strictly next-day execution.

    No-look-ahead rule
    -------------------
    A signal read at the close of day T is only ever executed at day T+1's
    OPEN price, and the resulting portfolio value is only ever priced with
    day T+1's CLOSE. The loop walks forward one day at a time and, for row
    i, only ever reads signals_df.iloc[i - 1] (yesterday's signal) and
    prices_df.iloc[i] (today's open/close) — it never looks at any row
    beyond i. The very first row therefore has no prior signal to act on
    (start fully in cash), and the last row's own signal is never executed
    within this dataset because there is no day after it (no T+1 to trade
    on).

    Parameters
    ----------
    signals_df : pandas.DataFrame
        Indexed by date, must contain:
        - 'Signal': 1 (buy/hold long), -1 (sell/go flat), 0 (no position)
        - 'Tradeable': bool, False on suspended/no-trade days
    prices_df : pandas.DataFrame
        Indexed by the same dates as signals_df, must contain 'open' and
        'close' columns.
    initial_capital : float
        Starting cash in PKR.
    commission : float
        Fraction of trade value charged as commission, e.g. 0.003 = 0.3%.
    slippage : float
        Fraction of trade value lost to slippage, e.g. 0.001 = 0.1%.

    Returns
    -------
    pandas.DataFrame
        One row per date in signals_df, with columns:
        date, portfolio_value, cash, shares_held, signal, trade_executed,
        trade_cost.

        'signal' is that row's own signal (generated at that day's close,
        to be executed the following day) — not necessarily the signal
        that caused that row's trade, which was generated the previous
        day. 'trade_executed' / 'trade_cost' describe any trade made
        AT that row's open, acting on the previous row's signal.

    Notes
    -----
    Long-only: signal=1 opens a long position (if flat), signal=-1 closes
    an existing long position back to cash (if long) — there is no short
    selling. signal=0, or a signal that doesn't change position (e.g. 1
    while already long), results in no trade that day.
    """
    dates = signals_df.index
    n = len(dates)
    cost_fraction = commission + slippage

    cash = float(initial_capital)
    shares_held = 0.0

    records = []
    for i in range(n):
        trade_executed = False
        trade_cost = 0.0

        if i > 0:
            prior_signal = signals_df["Signal"].iloc[i - 1]
            tradeable_today = signals_df["Tradeable"].iloc[i]

            if tradeable_today:
                open_price = prices_df["open"].iloc[i]

                if prior_signal == 1 and shares_held == 0:
                    trade_cost = cash * cost_fraction
                    shares_held = (cash - trade_cost) / open_price
                    cash = 0.0
                    trade_executed = True
                elif prior_signal == -1 and shares_held > 0:
                    gross_proceeds = shares_held * open_price
                    trade_cost = gross_proceeds * cost_fraction
                    cash = gross_proceeds - trade_cost
                    shares_held = 0.0
                    trade_executed = True

        close_price = prices_df["close"].iloc[i]
        portfolio_value = cash + shares_held * close_price

        records.append({
            "date": dates[i],
            "portfolio_value": portfolio_value,
            "cash": cash,
            "shares_held": shares_held,
            "signal": signals_df["Signal"].iloc[i],
            "trade_executed": trade_executed,
            "trade_cost": trade_cost,
        })

    return pd.DataFrame(records)


def equal_weight_benchmark(stock_dict, initial_capital=100000,
                            commission=0.003, slippage=0.001):
    """
    Buy-and-hold benchmark: equal-weight every stock in stock_dict on
    the first common trading day, then hold to the end with no further
    trading or rebalancing.

    This is the primary benchmark every strategy in this project is
    compared against. It requires no signal, no forecast, and no
    skill — any active strategy (classical or ML) needs to beat this,
    net of realistic costs, to be worth its added complexity and risk.

    Parameters
    ----------
    stock_dict : dict of {str: pandas.DataFrame}
        Cleaned OHLCV data per ticker, e.g. from src.data.load_selected().
        Each DataFrame must contain 'open' and 'close'.
    initial_capital : float
        Starting cash in PKR, split equally across all stocks
        (initial_capital / n_stocks per stock).
    commission : float
        Fraction of trade value charged as commission on each stock's
        single buy trade, e.g. 0.003 = 0.3%.
    slippage : float
        Fraction of trade value lost to slippage on each stock's single
        buy trade, e.g. 0.001 = 0.1%.

    Returns
    -------
    pandas.DataFrame
        Same column structure as run_backtest() — date, portfolio_value,
        cash, shares_held, signal, trade_executed, trade_cost — so
        compute_metrics() works on it directly with no modification.
        'shares_held' and 'signal' aren't meaningful for a multi-asset
        portfolio and are left as NaN; 'cash' is 0.0 throughout (fully
        invested after day 1); 'trade_executed' is True only on the
        start date (one buy per stock that day) and 'trade_cost' is the
        combined commission+slippage paid across all stocks that day.

    Notes
    -----
    start_date is the latest of every stock's own first available
    date, so every stock in stock_dict actually has data to buy on day
    1. A few tickers are missing a handful of rows mid-history (an
    individual trading halt with no row in the raw data at all, rather
    than a Tradeable=False row) instead of sharing one identical
    calendar — those gaps are forward-filled per stock (last known
    close carried forward) so a full portfolio value can still be
    computed on every date any of the stocks traded. The alternative,
    restricting to the intersection of every stock's calendar, would
    silently drop legitimate trading days for the other stocks just
    because one was halted. This is a standard simplification for a
    passive benchmark and is noted here rather than hidden.
    """
    tickers = list(stock_dict.keys())
    n_stocks = len(tickers)
    cost_fraction = commission + slippage
    per_stock_capital = initial_capital / n_stocks

    start_date = max(df.index.min() for df in stock_dict.values())

    all_dates = pd.DatetimeIndex(sorted(set().union(*(
        df.index[df.index >= start_date] for df in stock_dict.values()
    ))))

    shares_held = {}
    total_trade_cost = 0.0
    for ticker in tickers:
        open_price = stock_dict[ticker].loc[start_date, "open"]
        trade_cost = per_stock_capital * cost_fraction
        shares_held[ticker] = (per_stock_capital - trade_cost) / open_price
        total_trade_cost += trade_cost
    shares_series = pd.Series(shares_held)

    close_prices = pd.DataFrame({
        ticker: stock_dict[ticker]["close"].reindex(all_dates).ffill()
        for ticker in tickers
    })
    portfolio_value = close_prices.mul(shares_series, axis=1).sum(axis=1)

    trade_executed = pd.Series(False, index=all_dates)
    trade_executed.loc[start_date] = True

    trade_cost = pd.Series(0.0, index=all_dates)
    trade_cost.loc[start_date] = total_trade_cost

    return pd.DataFrame({
        "date": all_dates,
        "portfolio_value": portfolio_value.values,
        "cash": 0.0,
        "shares_held": np.nan,
        "signal": np.nan,
        "trade_executed": trade_executed.values,
        "trade_cost": trade_cost.values,
    })


def run_backtest_all(stock_dict, feature_func, signal_func,
                      initial_capital=100000):
    """
    Run run_backtest() independently on every stock in a dict.

    No-look-ahead rule
    -------------------
    Each stock is backtested completely independently and sequentially
    within run_backtest(); see that function's docstring for the exact
    day-T-signal / day-T+1-execution rule. Features and signals are
    generated per stock using only that stock's own historical data up
    to each point in time.

    Parameters
    ----------
    stock_dict : dict of {str: pandas.DataFrame}
        Cleaned OHLCV data per ticker, e.g. from src.data.load_selected().
    feature_func : callable
        Function taking a cleaned DataFrame and returning it with feature
        columns added, e.g. src.features.generate_features.
    signal_func : callable
        Function taking a DataFrame with features and returning it with a
        'Signal' column added, e.g. src.strategies.momentum_signals.
    initial_capital : float
        Starting cash in PKR, passed through to run_backtest() for each
        stock.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Maps ticker to its run_backtest() results DataFrame.
    """
    results = {}
    for ticker, df in stock_dict.items():
        featured = feature_func(df)
        signaled = signal_func(featured)
        results[ticker] = run_backtest(
            signaled, signaled, initial_capital=initial_capital
        )
    return results


if __name__ == "__main__":
    # Manual unit test: 5 days, prices [10, 11, 12, 11, 13],
    # signals [1, 1, -1, 0, 0], 0% commission, 0% slippage.
    #
    # Hand-calculated expected portfolio value at each day's close:
    #   Day0: no prior signal to act on -> stay in cash.       = 100,000.00
    #   Day1: prior signal (day0) = 1 -> buy at open=11 with
    #         all cash: shares = 100000 / 11. Priced at
    #         close=11 (same as open) -> value unchanged.      = 100,000.00
    #   Day2: prior signal (day1) = 1, already long -> hold.
    #         shares unchanged, priced at close=12:
    #         (100000 / 11) * 12                                = 109,090.909091
    #   Day3: prior signal (day2) = -1, long -> sell at
    #         open=11 (same price as bought at) -> back to
    #         all cash, no gain/loss.                            = 100,000.00
    #   Day4: prior signal (day3) = 0 -> hold cash.               = 100,000.00

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = [10, 11, 12, 11, 13]
    signals = [1, 1, -1, 0, 0]

    prices_df = pd.DataFrame({"open": prices, "close": prices}, index=dates)
    signals_df = pd.DataFrame(
        {"Signal": signals, "Tradeable": [True] * 5}, index=dates
    )

    result = run_backtest(
        signals_df, prices_df, initial_capital=100000,
        commission=0, slippage=0,
    )

    expected = [
        100000.0,
        100000.0,
        100000.0 / 11 * 12,
        100000.0,
        100000.0,
    ]
    actual = result["portfolio_value"].tolist()

    print(result.to_string(index=False))
    print("\nExpected portfolio values:", expected)
    print("Actual portfolio values:  ", actual)

    passed = all(abs(a - e) < 1e-6 for a, e in zip(actual, expected))
    print("\nPASS" if passed else "\nFAIL")
    assert passed, "Backtest engine does not match manual calculation"

    # --- Equal-weight benchmark, 25 selected stocks ---
    from src.data import load_selected
    from src.evaluation import compute_metrics

    stock_dict = load_selected()

    benchmark_results = equal_weight_benchmark(stock_dict)
    benchmark_metrics = compute_metrics(benchmark_results)

    print(f"\n\nEqual-weight benchmark — {len(stock_dict)} selected stocks:\n")
    for name, value in benchmark_metrics.items():
        print(f"{name:<25}: {value}")
