"""
Evaluation: performance metrics (Sharpe, Sortino, drawdown) and
tearsheet generation for strategy results.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(results_df, risk_free_rate=0.12):
    """
    Compute standard performance metrics from a run_backtest() result.

    All metrics are derived purely from the 'portfolio_value' column
    (plus 'trade_executed' for turnover) — no external price data is
    used, so this works identically for any single-asset backtest
    result.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Output of src.backtest.run_backtest(), with columns:
        date, portfolio_value, cash, shares_held, signal,
        trade_executed, trade_cost.
    risk_free_rate : float
        Annual risk-free rate used for Sharpe/Sortino/Calmar, default
        0.12 (12%) — proxied by the Pakistan 12-month T-bill yield
        (approx. 12.1% as of 2025). A single fixed rate is used across
        the full sample period as a simplification; this is noted as a
        limitation in the dissertation rather than modelled as a
        time-varying rate.

    Returns
    -------
    dict
        {
          'Annualised Return': float, as a percentage (e.g. 12.3),
          'Sharpe Ratio': float, rounded to 3 decimal places,
          'Sortino Ratio': float, rounded to 3 decimal places,
          'Max Drawdown': float, as a percentage (e.g. -23.4),
          'Calmar Ratio': float, rounded to 3 decimal places
                          (NaN if max drawdown is 0),
          'Turnover (trades/yr)': float, rounded to 1 decimal place,
        }
    """
    portfolio_value = results_df["portfolio_value"]
    daily_returns = portfolio_value.pct_change().dropna()

    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]
    n_years = len(results_df) / TRADING_DAYS_PER_YEAR

    total_return = (final_value / initial_value) - 1
    annualised_return = (1 + total_return) ** (1 / n_years) - 1

    excess_daily_returns = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    sharpe = (
        excess_daily_returns.mean() / excess_daily_returns.std()
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = (annualised_return - risk_free_rate) / downside_std

    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    calmar = (
        annualised_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    )

    trades = int(results_df["trade_executed"].sum())
    turnover = trades / n_years

    return {
        "Annualised Return": round(annualised_return * 100, 1),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown": round(max_drawdown * 100, 1),
        "Calmar Ratio": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "Turnover (trades/yr)": round(turnover, 1),
    }


def compare_strategies(results_dict):
    """
    Build the main results comparison table across multiple strategies.

    Parameters
    ----------
    results_dict : dict of {str: pandas.DataFrame}
        Maps a strategy (or strategy+ticker) name to its run_backtest()
        results DataFrame.

    Returns
    -------
    pandas.DataFrame
        One row per strategy (index = strategy name), one column per
        metric returned by compute_metrics(): Annualised Return, Sharpe
        Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, and
        Turnover (trades/yr). This is the main results table for the
        dissertation.
    """
    metrics = {
        name: compute_metrics(results_df)
        for name, results_df in results_dict.items()
    }
    return pd.DataFrame.from_dict(metrics, orient="index")


if __name__ == "__main__":
    from src.backtest import run_backtest
    from src.data import load_data
    from src.features import generate_features
    from src.strategies import momentum_signals

    kel = generate_features(load_data("KEL"))
    kel_signals = momentum_signals(kel)
    kel_results = run_backtest(kel_signals, kel_signals)

    metrics = compute_metrics(kel_results)

    print("KEL — momentum strategy metrics:\n")
    for name, value in metrics.items():
        print(f"{name:<25}: {value}")
