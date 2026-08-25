"""
Final results script: runs Momentum, Mean Reversion, ARIMA and XGBoost
on all 25 selected tickers plus the equal-weight benchmark, computes
period-split metrics (Train/OOS/Final Eval/Full Sample), and saves the
tables and figures the report draws on.

OOS (2023-2024) is the primary result — the first period every
strategy sees as genuinely unseen data. Final Eval (2025) is reported
for completeness, not used to pick a winner.

"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import equal_weight_benchmark, run_backtest
from src.data import load_selected
from src.evaluation import METRIC_KEYS, PERIOD_BOUNDS, compute_metrics_by_period
from src.features import generate_features
from src.models import load_arima_signals, load_xgboost_signals
from src.strategies import mean_reversion_signals, momentum_signals

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

# Same as run_backtest()'s own defaults, passed explicitly so this
# script keeps reproducing the same numbers if those defaults change.
COMMISSION = 0.0015
SLIPPAGE = 0.001

STRATEGY_KEYS = ["momentum", "mean_reversion", "arima", "xgboost"]
STRATEGY_DISPLAY = {
    "momentum": "Momentum",
    "mean_reversion": "Mean Reversion",
    "arima": "ARIMA",
    "xgboost": "XGBoost",
}
BENCHMARK_LABEL = "Equal-Weight Benchmark"
ALL_STRATEGY_LABELS = [STRATEGY_DISPLAY[k] for k in STRATEGY_KEYS] + [BENCHMARK_LABEL]

PERIOD_CSV_NAMES = {
    "Full Sample (2016-2025)": "metrics_full_sample.csv",
    "Train (2016-2022)": "metrics_train.csv",
    "OOS (2023-2024)": "metrics_oos.csv",
    "Final Eval (2025)": "metrics_final_eval.csv",
}
OOS_LABEL = "OOS (2023-2024)"

STRATEGY_COLORS = {
    "Momentum": "#1f77b4",
    "Mean Reversion": "#ff7f0e",
    "ARIMA": "#2ca02c",
    "XGBoost": "#d62728",
    BENCHMARK_LABEL: "#7f7f7f",
}


# ---------------------------------------------------------------------
# STEP 1 — per-ticker, per-strategy backtests
# ---------------------------------------------------------------------

def _aligned_signal_frame(featured_df, signal_series):
    """Attach a cached Signal series onto a featured DataFrame; dates missing from the cache default to Signal=0."""
    out = featured_df.copy()
    out["Signal"] = signal_series.reindex(out.index, fill_value=0)
    return out


def build_ticker_results(ticker, raw_df):
    """Run all four strategies for one ticker. Returns {strategy_key: results_df}."""
    featured = generate_features(raw_df)

    momentum_df = momentum_signals(featured)
    mean_reversion_df = mean_reversion_signals(featured)
    arima_df = _aligned_signal_frame(featured, load_arima_signals(ticker))
    xgboost_df = _aligned_signal_frame(featured, load_xgboost_signals(ticker))

    return {
        "momentum": run_backtest(
            momentum_df, momentum_df, commission=COMMISSION, slippage=SLIPPAGE
        ),
        "mean_reversion": run_backtest(
            mean_reversion_df, mean_reversion_df,
            commission=COMMISSION, slippage=SLIPPAGE,
        ),
        "arima": run_backtest(
            arima_df, arima_df, commission=COMMISSION, slippage=SLIPPAGE
        ),
        "xgboost": run_backtest(
            xgboost_df, xgboost_df, commission=COMMISSION, slippage=SLIPPAGE
        ),
    }


def run_all_tickers(stock_dict):
    """
    Run build_ticker_results() for every ticker; a per-ticker failure
    is recorded, not fatal. Returns (results, failures).
    """
    results = {}
    failures = []

    tickers = list(stock_dict.keys())
    for i, ticker in enumerate(tickers, start=1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            results[ticker] = build_ticker_results(ticker, stock_dict[ticker])
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")
            failures.append((ticker, str(exc)))

    return results, failures


# ---------------------------------------------------------------------
# STEP 3 — period metrics, per ticker and aggregated
# ---------------------------------------------------------------------

def compute_all_period_metrics(results):
    """Run compute_metrics_by_period() for every ticker/strategy. Returns [ticker][strategy_key] -> {period_label: metrics_or_None}."""
    per_ticker_period_metrics = {}
    for ticker, strat_results in results.items():
        strategy_metrics = {}
        for strat, df in strat_results.items():
            strategy_metrics[strat] = compute_metrics_by_period(df)
        per_ticker_period_metrics[ticker] = strategy_metrics
    return per_ticker_period_metrics


def aggregate_period_metrics(per_ticker_period_metrics, strategy_key, period_label):
    """Mean of each of the 6 metrics across tickers with data for this period (NaN-filled if none do)."""
    rows = []
    for ticker_metrics in per_ticker_period_metrics.values():
        metrics_for_period = ticker_metrics[strategy_key][period_label]
        if metrics_for_period is not None:
            rows.append(metrics_for_period)
    if not rows:
        return {key: np.nan for key in METRIC_KEYS}

    means = pd.DataFrame(rows)[METRIC_KEYS].mean(numeric_only=True)
    return {
        "Annualised Return": round(means["Annualised Return"], 1),
        "Sharpe Ratio": round(means["Sharpe Ratio"], 3),
        "Sortino Ratio": round(means["Sortino Ratio"], 3),
        "Max Drawdown": round(means["Max Drawdown"], 1),
        "Calmar Ratio": round(means["Calmar Ratio"], 3),
        "Turnover (trades/yr)": round(means["Turnover (trades/yr)"], 1),
    }


# ---------------------------------------------------------------------
# STEP 4 — results tables
# ---------------------------------------------------------------------

def count_valid_sharpe_tickers(per_ticker_period_metrics, strategy_key, period_label):
    """
    How many tickers contributed a non-NaN Sharpe to this strategy's
    mean. A ticker with ~zero trades in a period gets NaN Sharpe and is
    silently skipped by pandas' mean() — this makes that count visible.
    """
    valid = 0
    for ticker_metrics in per_ticker_period_metrics.values():
        m = ticker_metrics[strategy_key][period_label]
        if m is not None and pd.notna(m["Sharpe Ratio"]):
            valid += 1
    return valid


def build_period_table(per_ticker_period_metrics, benchmark_period_metrics, period_label):
    """
    One period's comparison table: 5 strategy rows (4 averaged across
    tickers, benchmark taken directly), 6 metric columns plus
    'n_tickers_sharpe' (NaN for the benchmark — it's not a per-ticker
    average, so the count doesn't apply).
    """
    rows = {}
    for strat_key in STRATEGY_KEYS:
        display_name = STRATEGY_DISPLAY[strat_key]
        rows[display_name] = aggregate_period_metrics(
            per_ticker_period_metrics, strat_key, period_label
        )

    bench_metrics = benchmark_period_metrics.get(period_label)
    if bench_metrics is not None:
        rows[BENCHMARK_LABEL] = bench_metrics
    else:
        rows[BENCHMARK_LABEL] = {key: np.nan for key in METRIC_KEYS}

    table = pd.DataFrame.from_dict(rows, orient="index")[METRIC_KEYS]
    table = table.loc[ALL_STRATEGY_LABELS]

    n_tickers_sharpe = {}
    for strat_key in STRATEGY_KEYS:
        display_name = STRATEGY_DISPLAY[strat_key]
        n_tickers_sharpe[display_name] = count_valid_sharpe_tickers(
            per_ticker_period_metrics, strat_key, period_label
        )
    n_tickers_sharpe[BENCHMARK_LABEL] = np.nan
    table["n_tickers_sharpe"] = pd.Series(n_tickers_sharpe).loc[ALL_STRATEGY_LABELS]

    return table


def build_per_ticker_oos_table(per_ticker_period_metrics):
    """
    Long-format table, one row per (ticker, strategy), OOS period only.
    The benchmark is excluded — it isn't a per-ticker result.
    """
    rows = []
    for ticker in per_ticker_period_metrics:
        for strat_key in STRATEGY_KEYS:
            metrics = per_ticker_period_metrics[ticker][strat_key][OOS_LABEL]
            if metrics is None:
                metrics = {key: np.nan for key in METRIC_KEYS}
            row = {"Ticker": ticker, "Strategy": STRATEGY_DISPLAY[strat_key]}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows, columns=["Ticker", "Strategy"] + METRIC_KEYS)


# ---------------------------------------------------------------------
# STEP 5 — figures
# ---------------------------------------------------------------------

def _apply_clean_style():
    """Plain rcParams tweaks — no plt.style.use(), so this doesn't depend on seaborn being installed."""
    plt.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    })


def _oos_rebased_series(results_df):
    """Slice to the OOS period and rebase portfolio_value to start at 100,000. None if the slice is empty."""
    start, end = PERIOD_BOUNDS[OOS_LABEL]
    dates = pd.to_datetime(results_df["date"])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    period_df = results_df.loc[mask]
    if period_df.empty:
        return None

    rebased = (
        period_df["portfolio_value"] / period_df["portfolio_value"].iloc[0] * 100000
    )
    rebased.index = pd.DatetimeIndex(period_df["date"].values)
    return rebased.sort_index()


def mean_equity_curve_oos(results, strategy_key):
    """Mean OOS-rebased equity curve across all tickers for one strategy, aligned on date."""
    series_list = []
    for ticker, strat_results in results.items():
        series = _oos_rebased_series(strat_results[strategy_key])
        if series is not None:
            series_list.append(series.rename(ticker))

    combined = pd.concat(series_list, axis=1)
    return combined.mean(axis=1, skipna=True).sort_index()


def plot_equity_curves_oos(results, benchmark_results, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for strat_key in STRATEGY_KEYS:
        label = STRATEGY_DISPLAY[strat_key]
        curve = mean_equity_curve_oos(results, strat_key)
        ax.plot(curve.index, curve.values, label=label,
                 color=STRATEGY_COLORS[label], linewidth=1.6)

    bench_curve = _oos_rebased_series(benchmark_results)
    if bench_curve is not None:
        ax.plot(bench_curve.index, bench_curve.values, label=BENCHMARK_LABEL,
                 color=STRATEGY_COLORS[BENCHMARK_LABEL], linewidth=1.8,
                 linestyle="--")

    ax.set_title("Mean Strategy Performance — Out-of-Sample (2023-2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (PKR, rebased to 100,000)")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_sharpe_bar_oos(oos_table, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = oos_table.index.tolist()
    values = oos_table["Sharpe Ratio"].values
    colors = [STRATEGY_COLORS[label] for label in labels]

    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Sharpe Ratio by Strategy — OOS (2023-2024)")
    ax.set_xlabel("Sharpe Ratio")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_drawdown_bar_oos(oos_table, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = oos_table.index.tolist()
    values = oos_table["Max Drawdown"].values
    colors = [STRATEGY_COLORS[label] for label in labels]

    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Maximum Drawdown by Strategy — OOS (2023-2024)")
    ax.set_xlabel("Max Drawdown (%)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_turnover_vs_return_oos(oos_table, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, row in oos_table.iterrows():
        ax.scatter(row["Turnover (trades/yr)"], row["Annualised Return"],
                    color=STRATEGY_COLORS[label], s=90, zorder=3)
        ax.annotate(label, (row["Turnover (trades/yr)"], row["Annualised Return"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Turnover vs Net Return — OOS (2023-2024)")
    ax.set_xlabel("Turnover (trades/yr)")
    ax.set_ylabel("Annualised Return (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _apply_clean_style()

    print("Loading 25 selected stocks...")
    stock_dict = load_selected()
    print(f"Loaded {len(stock_dict)} tickers.\n")

    print("STEP 1: running Momentum / Mean Reversion / ARIMA / XGBoost per ticker...")
    results, failures = run_all_tickers(stock_dict)
    print(f"STEP 1 done: {len(results)} tickers succeeded, {len(failures)} failed.\n")

    print("STEP 2: equal-weight benchmark (25 stocks)...")
    benchmark_results = equal_weight_benchmark(
        stock_dict, commission=COMMISSION, slippage=SLIPPAGE
    )
    print("STEP 2 done.\n")

    print("STEP 3: computing period-split metrics (per ticker, then aggregating)...")
    per_ticker_period_metrics = compute_all_period_metrics(results)
    benchmark_period_metrics = compute_metrics_by_period(benchmark_results)
    print("STEP 3 done.\n")

    print("STEP 4: building and saving results tables...")
    saved_files = []

    period_tables = {}
    for period_label, csv_name in PERIOD_CSV_NAMES.items():
        table = build_period_table(
            per_ticker_period_metrics, benchmark_period_metrics, period_label
        )
        period_tables[period_label] = table
        out_path = TABLES_DIR / csv_name
        table.to_csv(out_path, index_label="Strategy")
        saved_files.append(out_path)

    per_ticker_oos_table = build_per_ticker_oos_table(per_ticker_period_metrics)
    per_ticker_oos_path = TABLES_DIR / "per_ticker_oos.csv"
    per_ticker_oos_table.to_csv(per_ticker_oos_path, index=False)
    saved_files.append(per_ticker_oos_path)
    print("STEP 4 done.\n")

    print("STEP 5: building and saving figures...")
    oos_table = period_tables[OOS_LABEL]

    equity_curves_path = FIGURES_DIR / "equity_curves_oos.png"
    plot_equity_curves_oos(results, benchmark_results, equity_curves_path)
    saved_files.append(equity_curves_path)

    sharpe_bar_path = FIGURES_DIR / "sharpe_by_strategy_oos.png"
    plot_sharpe_bar_oos(oos_table, sharpe_bar_path)
    saved_files.append(sharpe_bar_path)

    drawdown_bar_path = FIGURES_DIR / "drawdown_by_strategy_oos.png"
    plot_drawdown_bar_oos(oos_table, drawdown_bar_path)
    saved_files.append(drawdown_bar_path)

    turnover_scatter_path = FIGURES_DIR / "turnover_vs_return_oos.png"
    plot_turnover_vs_return_oos(oos_table, turnover_scatter_path)
    saved_files.append(turnover_scatter_path)
    print("STEP 5 done.\n")

    print("STEP 6: summary\n")
    print("=" * 72)
    print("PRIMARY RESULT — OOS (2023-2024), mean metrics across 25 tickers")
    print("(Equal-Weight Benchmark is a single 25-stock portfolio, not averaged)")
    print("=" * 72)
    print(oos_table.to_string())
    print()

    print("Files saved:")
    for path in saved_files:
        print(f"  {path}")
    print()

    if failures:
        print(f"Ticker failures ({len(failures)}):")
        for ticker, error in failures:
            print(f"  {ticker}: {error}")
    else:
        print("Ticker failures: none — all 25 tickers processed successfully.")

    return {
        "results": results,
        "benchmark_results": benchmark_results,
        "period_tables": period_tables,
        "per_ticker_oos_table": per_ticker_oos_table,
        "failures": failures,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    main()
