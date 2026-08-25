from pathlib import Path

import pandas as pd

PER_TICKER_OOS_PATH = Path("results/tables/per_ticker_oos.csv")
OUTPUT_PATH = Path("results/tables/consistency_summary.csv")

STRATEGIES = ["Momentum", "Mean Reversion", "ARIMA", "XGBoost"]
N_TICKERS_TOTAL = 25


def compute_consistency_summary(per_ticker_df):

    rows = []
    for strategy in STRATEGIES:
        strat_df = per_ticker_df[per_ticker_df["Strategy"] == strategy]

        n_tickers = int(strat_df["Sharpe Ratio"].notna().sum())
        n_positive_return = int((strat_df["Annualised Return"] > 0).sum())
        n_positive_sharpe = int((strat_df["Sharpe Ratio"] > 0).sum())

        rows.append({
            "Strategy": strategy,
            "n_tickers": n_tickers,
            "n_positive_return": n_positive_return,
            "n_positive_sharpe": n_positive_sharpe,
            "pct_positive_return": round(n_positive_return / N_TICKERS_TOTAL * 100, 1),
            "pct_positive_sharpe": round(n_positive_sharpe / n_tickers * 100, 1),
        })

    return pd.DataFrame(rows)


def main():
    per_ticker_df = pd.read_csv(PER_TICKER_OOS_PATH)

    summary = compute_consistency_summary(per_ticker_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Per-ticker OOS consistency summary (from {PER_TICKER_OOS_PATH}):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
