"""Core backtesting engine which includes the daily P&L loop with transaction costs."""

import numpy as np
import pandas as pd


def run_backtest(signals_df, prices_df, initial_capital=100000,
                  commission=0.0015, slippage=0.001):
    """
    Single-asset, long-only backtest with next-day execution.
    The code looks at previous day's signal to decide whether to buy/sell at the next day's open.
    The cost is modeled as comission (0.15%) and slippage (0.1%)
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


def equal_weight_benchmark(stock_dict, initial_capital=100000, commission=0.0015, slippage=0.001):

   # Equal-weight buy-and-hold benchmark. Invest equally in every stock on the
   # first common trading day and hold until the end with no rebalancing.
   # Active strategies need to beat this benchmark after costs to justify
   # their added complexity.

    tickers = list(stock_dict.keys())
    n_stocks = len(tickers)
    cost_fraction = commission + slippage
    per_stock_capital = initial_capital / n_stocks

    start_date = max(df.index.min() for df in stock_dict.values())

    # Collect every trading date that appears in any stock's history
    # from start_date onward, then sort them into one shared calendar.
    all_dates_set = set()
    for df in stock_dict.values():
        dates_after_start = df.index[df.index >= start_date]
        all_dates_set.update(dates_after_start)
    all_dates = pd.DatetimeIndex(sorted(all_dates_set))

    shares_held = {}
    total_trade_cost = 0.0
    for ticker in tickers:
        open_price = stock_dict[ticker].loc[start_date, "open"]
        trade_cost = per_stock_capital * cost_fraction
        shares_held[ticker] = (per_stock_capital - trade_cost) / open_price
        total_trade_cost += trade_cost
    shares_series = pd.Series(shares_held)

    # Build a close-price column per stock, forward-filled onto the
    # shared calendar, then combine into total portfolio value.
    close_price_columns = {}
    for ticker in tickers:
        close_price_columns[ticker] = stock_dict[ticker]["close"].reindex(all_dates).ffill()
    close_prices = pd.DataFrame(close_price_columns)

    weighted_prices = close_prices.mul(shares_series, axis=1)
    portfolio_value = weighted_prices.sum(axis=1)

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


if __name__ == "__main__":
    # Hand-verified example: prices [10,11,12,11,13]
    # signals : [1,1,-1,0,0], no costs -> buy day1 @11, hold, sell day3 @11 flat.
    # Expected close values: 100000, 100000, 109090.91, 100000, 100000.

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
