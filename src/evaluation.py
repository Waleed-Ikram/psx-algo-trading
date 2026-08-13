"""
Evaluation: performance metrics (Sharpe, Sortino, drawdown) and
tearsheet generation for strategy results.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Pre-registered train / out-of-sample / final-eval split (see CLAUDE.md).
# Training: models are only ever fit on data up to TRAIN_END. OOS
# (2023-2024) is the primary evaluation window used to judge and compare
# strategies. Final Eval (2025) is the final held-out year, reported once
# at the end rather than used to pick between strategies.
TRAIN_START = "2016-06-27"
TRAIN_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2024-12-31"
FINAL_START = "2025-01-01"
FINAL_END = "2025-12-31"

# Ordered so dict output (Python 3.7+ preserves insertion order) reads
# chronologically: the three disjoint sub-periods, then the full sample.
PERIOD_BOUNDS = {
    "Train (2016-2022)": (TRAIN_START, TRAIN_END),
    "OOS (2023-2024)": (OOS_START, OOS_END),
    "Final Eval (2025)": (FINAL_START, FINAL_END),
    "Full Sample (2016-2025)": (TRAIN_START, FINAL_END),
}

# A period with fewer rows than this produces an unreliable Sharpe/
# Sortino/drawdown estimate (e.g. std of a handful of daily returns is
# noisy) and is reported as None rather than a misleading number.
MIN_PERIOD_ROWS = 30

METRIC_KEYS = [
    "Annualised Return", "Sharpe Ratio", "Sortino Ratio",
    "Max Drawdown", "Calmar Ratio", "Turnover (trades/yr)",
]


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


def compute_metrics_by_period(results_df, risk_free_rate=0.12):
    """
    Run compute_metrics() independently over each of the dissertation's
    pre-registered periods, plus the full sample, from a single
    run_backtest() result.

    Why this exists
    ----------------
    The project's methodology commits to a walk-forward split (see
    CLAUDE.md): models are only ever trained on data up to TRAIN_END,
    OOS_START..OOS_END (2023-2024) is the **primary out-of-sample
    evaluation window** used to judge and compare strategies, and
    FINAL_START..FINAL_END (2025) is the **final held-out year** —
    touched once, at the end, to report headline results, not used to
    pick between strategies. Before this function existed,
    compute_metrics() was only ever run over the entire 2016-2025
    signal history, blending all three periods (plus the pre-2018
    walk-forward warm-up) into one blended number. This is what
    actually enforces reporting them separately.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Output of src.backtest.run_backtest() (or
        equal_weight_benchmark()): a 'date' column plus 'portfolio_value'
        and the other columns compute_metrics() reads, spanning some or
        all of 2016-2025.
    risk_free_rate : float
        Passed through to compute_metrics() for every period.

    Returns
    -------
    dict
        {
          'Train (2016-2022)': {...6 metrics...} or None,
          'OOS (2023-2024)': {...6 metrics...} or None,
          'Final Eval (2025)': {...6 metrics...} or None,
          'Full Sample (2016-2025)': {...6 metrics...} or None,
        }
        A period is None if its slice of results_df has fewer than
        MIN_PERIOD_ROWS (30) rows — too little data for a stable Sharpe/
        Sortino/drawdown estimate.

    Notes
    -----
    Each period's portfolio_value series is rebased to start at 100,000
    (period_value / period_value.iloc[0] * 100000) before
    compute_metrics() runs on it, so the level is comparable across
    periods regardless of how the portfolio had already grown or
    shrunk by the time that period began. Every metric compute_metrics()
    returns depends only on the *shape* of the return series (pct
    changes, drawdown ratios) or on the total return ratio over the
    slice, never on its absolute level, so this rebasing changes no
    metric's value — it exists so the portfolio_value path itself is
    directly comparable if plotted per-period.
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
    Build a results comparison table (rows = strategies, columns = the
    6 compute_metrics() metrics) for one specific period only.

    OOS (2023-2024) is the primary evaluation window this project uses
    to compare strategies against each other and against the
    benchmark — models are only ever trained on data up to TRAIN_END,
    so OOS is the first period every strategy sees as genuinely unseen
    data. Final Eval (2025) is the final held-out year: report it once
    at the end as a confirmatory check, not as the basis for picking a
    winner between strategies.

    Parameters
    ----------
    results_dict : dict of {str: pandas.DataFrame}
        Maps a strategy (or strategy+ticker) name to its
        run_backtest() results DataFrame, exactly as compare_strategies()
        expects.
    period : str
        One of the four period labels compute_metrics_by_period()
        produces: 'Train (2016-2022)', 'OOS (2023-2024)',
        'Final Eval (2025)', or 'Full Sample (2016-2025)'.

    Returns
    -------
    pandas.DataFrame
        One row per strategy (index = strategy name), one column per
        metric. A strategy whose results_df doesn't cover `period` with
        at least MIN_PERIOD_ROWS rows gets a row of NaN rather than
        being dropped, so every strategy passed in still appears in
        the output.

    Raises
    ------
    ValueError
        If `period` isn't one of the four labels compute_metrics_by_period()
        produces.
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
        rows[name] = (
            period_metrics if period_metrics is not None
            else {key: np.nan for key in METRIC_KEYS}
        )

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
