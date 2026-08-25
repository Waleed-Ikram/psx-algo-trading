"""Performance metrics are all measured here : Sharpe, Sortino, drawdown, Calmar, turnover."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

#data split dates for train, oos and final evaluation

#Train 
TRAIN_START = "2016-06-27"
TRAIN_END = "2022-12-31"

#Out-Of-Sample
OOS_START = "2023-01-01"
OOS_END = "2024-12-31"

#Final evaluation
FINAL_START = "2025-01-01"
FINAL_END = "2025-12-31"

# Kept in chronological order: three disjoint sub-periods, then full sample.
PERIOD_BOUNDS = {
    "Train (2016-2022)": (TRAIN_START, TRAIN_END),
    "OOS (2023-2024)": (OOS_START, OOS_END),
    "Final Eval (2025)": (FINAL_START, FINAL_END),
    "Full Sample (2016-2025)": (TRAIN_START, FINAL_END),
}


MIN_PERIOD_ROWS = 30

# Below this, treat a return series' std as zero rather than a real
# number (see compute_metrics()).
STD_FLOOR = 1e-12

METRIC_KEYS = [
    "Annualised Return", "Sharpe Ratio", "Sortino Ratio",
    "Max Drawdown", "Calmar Ratio", "Turnover (trades/yr)",
]


def compute_metrics(results_df, risk_free_rate=0.12):
    """
    Standard performance metrics from a run_backtest() result:
    Annualised Return, Sharpe, Sortino, Max Drawdown, Calmar, Turnover.

    risk_free_rate defaults to 12% (Pakistan's 12-month T-bill yield),

    A portfolio that never trades has zero return variance, and
    computing std() on returns after subtracting a constant risk-free
    rate is numerically unstable near zero (can land on ~1e-19 instead
    of exactly 0) — dividing by that gives an absurd Sharpe. So Sharpe
    and Sortino are floored: below STD_FLOOR, they're NaN rather than
    computed, same as Calmar already is when max drawdown is 0.
    """
    portfolio_value = results_df["portfolio_value"]
    daily_returns = portfolio_value.pct_change().dropna()

    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]
    n_years = len(results_df) / TRADING_DAYS_PER_YEAR

    total_return = (final_value / initial_value) - 1
    annualised_return = (1 + total_return) ** (1 / n_years) - 1

    excess_daily_returns = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    if daily_returns.std() < STD_FLOOR:
        sharpe = float("nan")
    else:
        sharpe = (
            excess_daily_returns.mean() / excess_daily_returns.std()
        ) * np.sqrt(TRADING_DAYS_PER_YEAR)

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std < STD_FLOOR:
        sortino = float("nan")
    else:
        sortino = (annualised_return - risk_free_rate) / downside_std

    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    if max_drawdown != 0:
        calmar = annualised_return / abs(max_drawdown)
    else:
        calmar = np.nan

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

    metrics = {
        name: compute_metrics(results_df)
        for name, results_df in results_dict.items()
    }
    return pd.DataFrame.from_dict(metrics, orient="index")


def compute_metrics_by_period(results_df, risk_free_rate=0.12):
    """
    This function calculates metrics separately for Train, OOS, Final Eval,
    and Full Sample.

    Returns a dictionary in the form {period_label: metrics_dict_or_None}.
    A period returns None when it has fewer than MIN_PERIOD_ROWS rows. The
    portfolio value for each period is rebased to 100,000 at the start. This
    is only done to make portfolio paths comparable if they are plotted.
    """
    dates = pd.to_datetime(results_df["date"])

    period_results = {}
    for label, (start, end) in PERIOD_BOUNDS.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        period_df = results_df.loc[mask]

        if len(period_df) < MIN_PERIOD_ROWS:
            period_results[label] = None
            continue

        rebased_df = period_df.copy()
        rebased_df["portfolio_value"] = (
            period_df["portfolio_value"]
            / period_df["portfolio_value"].iloc[0]
            * 100000
        )

        period_results[label] = compute_metrics(
            rebased_df, risk_free_rate=risk_free_rate
        )

    return period_results


def compare_strategies_by_period(results_dict, period):
   """
   This function creates a table that compares all strategies across the
   metrics for one period only. The period must be one of the four labels
   used in compute_metrics_by_period().

   A strategy with too little data for that period is included with NaN
   values instead of being removed. Raises a ValueError if an unknown
   period label is provided.
   """
   
    if period not in PERIOD_BOUNDS:
        raise ValueError(
            f"Unknown period {period!r}; expected one of {list(PERIOD_BOUNDS)}"
        )

    rows = {}
    for name, results_df in results_dict.items():
        period_metrics = compute_metrics_by_period(
            results_df, risk_free_rate=0.12
        ).get(period)
        if period_metrics is not None:
            rows[name] = period_metrics
        else:
            rows[name] = {key: np.nan for key in METRIC_KEYS}

    return pd.DataFrame.from_dict(rows, orient="index")


if __name__ == "__main__":
    from src.backtest import run_backtest
    from src.data import load_data
    from src.features import generate_features
    from src.strategies import momentum_signals

    kel = generate_features(load_data("KEL"))
    kel_signals = momentum_signals(kel)
    kel_results = run_backtest(kel_signals, kel_signals)

    metrics = compute_metrics(kel_results)

    print("KEL — momentum strategy metrics (full blended sample):\n")
    for name, value in metrics.items():
        print(f"{name:<25}: {value}")

    print("\n\nKEL — momentum strategy metrics BY PERIOD:\n")
    period_metrics = compute_metrics_by_period(kel_results)
    for period_label, period_result in period_metrics.items():
        print(f"--- {period_label} ---")
        if period_result is None:
            print("  (fewer than MIN_PERIOD_ROWS rows in this period — skipped)")
        else:
            for name, value in period_result.items():
                print(f"  {name:<25}: {value}")
        print()
