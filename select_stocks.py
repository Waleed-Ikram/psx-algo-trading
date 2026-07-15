"""
Select the top 25 most liquid KSE-100 stocks by average daily volume,
excluding short-history tickers, for use in the backtesting universe.
"""

from pathlib import Path

import pandas as pd

from src.data import load_all

MIN_ROWS = 1500
TOP_N = 25
OUTPUT_PATH = Path("results/selected_stocks.csv")


def select_stocks():
    data = load_all()

    records = []
    for ticker, df in data.items():
        volume = df["volume"][df["volume"] > 0]
        records.append(
            {
                "Ticker": ticker,
                "AvgDailyVolume": volume.mean(),
                "Rows": len(df),
            }
        )

    ranked = pd.DataFrame(records).sort_values(
        "AvgDailyVolume", ascending=False
    )

    eligible = ranked[ranked["Rows"] >= MIN_ROWS]
    selected = eligible.head(TOP_N).reset_index(drop=True)

    print(f"\nSelected {len(selected)} stocks (min {MIN_ROWS} rows, "
          f"ranked by average daily volume):\n")
    for _, row in selected.iterrows():
        print(
            f"{row['Ticker']:<10} "
            f"AvgDailyVolume={row['AvgDailyVolume']:>15,.0f} "
            f"Rows={row['Rows']:>6}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved selection to {OUTPUT_PATH}")

    return selected


if __name__ == "__main__":
    select_stocks()
