"""Load raw PSX CSVs and clean them into data/cleaned/."""

from pathlib import Path

import pandas as pd

# Top 25 stocks by average daily volume, picked by select_stocks.py
SELECTED_TICKERS = [
    "KEL", "BOP", "CNERGY", "TRG", "PIBTL", "PAEL", "FFL", "MLCF", "FCCL",
    "PTC", "HUMNL", "SSGC", "LOTCHEM", "POWER", "OGDC", "SNGP", "PPL",
    "DGKC", "HUBC", "NBP", "SEARL", "EFERT", "HBL", "BAFL", "PSO",
]

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

# TPLRF1 had only 398 rows, too short to be useful.
SKIP_FILES = {"TPLRF1.csv", "KSE100_index.csv", "KSE100_index_fixed.csv"}


def prepare_dataframe(df):

   # Sort by date and add Tradeable / DailyReturn / PriceFlag columns.

    df = df.copy()

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()

    df["Tradeable"] = df["volume"] != 0
    df["DailyReturn"] = df["close"].pct_change().round(6)
    df["PriceFlag"] = df["DailyReturn"].abs() > 0.20

    return df


def clean_ticker(raw_path, cleaned_dir=CLEANED_DIR):
    """Clean one raw CSV and save it to cleaned_dir. Never touches raw_path."""
    ticker = raw_path.stem

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df = prepare_dataframe(df)

    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_dir / raw_path.name)

    non_tradeable = int((~df["Tradeable"]).sum())
    price_flags = int(df["PriceFlag"].sum())

    print(
        f"{ticker}: {len(df)} rows, {non_tradeable} non-tradeable, "
        f"{price_flags} price flags"
    )

    return {
        "ticker": ticker,
        "rows": len(df),
        "non_tradeable": non_tradeable,
        "price_flags": price_flags,
    }


def clean_all(raw_dir=RAW_DIR, cleaned_dir=CLEANED_DIR):
    """Clean every raw CSV (skipping SKIP_FILES) and print a summary."""
    raw_dir = Path(raw_dir)
    csv_files = [
        f for f in sorted(raw_dir.glob("*.csv")) if f.name not in SKIP_FILES
    ]

    print(
        f"Found {len(csv_files)} CSV files to clean "
        f"(skipping: {sorted(SKIP_FILES)})"
    )

    all_stats = [clean_ticker(f, cleaned_dir=cleaned_dir) for f in csv_files]

    total_non_tradeable = sum(s["non_tradeable"] for s in all_stats)
    total_price_flags = sum(s["price_flags"] for s in all_stats)
    high_flag_stocks = sorted(
        (s for s in all_stats if s["price_flags"] > 5),
        key=lambda s: s["price_flags"], reverse=True,
    )

    print(
        f"\nCleaning summary: {len(all_stats)} files processed, "
        f"{total_non_tradeable} non-tradeable rows, "
        f"{total_price_flags} price-flagged rows"
    )

    if high_flag_stocks:
        print("\nStocks with PriceFlag > 5 (potential data quality issues):")
        for s in high_flag_stocks:
            print(f"  {s['ticker']:<10} {s['price_flags']:>3} flags")
    else:
        print("No stocks with PriceFlag > 5.")

    return all_stats


def main():
    """Clean all raw stock CSVs."""
    clean_all()


def load_data(ticker, data_dir="data/cleaned", end_date="2025-12-31"):
    
    #this function loads a cleaned CSV for a given ticker from the directory.
    
    path = Path(data_dir) / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cleaned data found for '{ticker}' at {path}"
        )

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.sort_index()
    df = df[df.index <= pd.Timestamp(end_date)]
    return df


def load_all(data_dir="data/cleaned", exclude=None, end_date="2025-12-31"):
    #Load every cleaned ticker CSV in data_dir into a dict.

    exclude = set(exclude) if exclude else set()
    data_dir = Path(data_dir)

    data = {}
    for path in sorted(data_dir.glob("*.csv")):
        ticker = path.stem
        if ticker in exclude:
            continue

        data[ticker] = load_data(ticker, data_dir=data_dir, end_date=end_date)

    print(f"Loaded {len(data)} files from {data_dir}")
    return data


def load_selected(data_dir="data/cleaned", end_date="2025-12-31"):
    """Load the 25 selected tickers only (see SELECTED_TICKERS)."""
    all_data = load_all(data_dir=data_dir, end_date=end_date)
    return {t: all_data[t] for t in SELECTED_TICKERS if t in all_data}


if __name__ == "__main__":
    main()
