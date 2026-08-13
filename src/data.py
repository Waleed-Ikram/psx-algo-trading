"""
Data layer: loading raw PSX CSVs, cleaning and validating them, and
reading/writing the SQLite database (data/database/psx.db).
"""

from pathlib import Path

import pandas as pd

BENCHMARK_FILE = "KSE100_index.csv"

# Top 25 stocks by average daily volume (min 1500 rows), selected via
# select_stocks.py. See results/selected_stocks.csv for the full ranking.
SELECTED_TICKERS = [
    "KEL", "BOP", "CNERGY", "TRG", "PIBTL", "PAEL", "FFL", "MLCF", "FCCL",
    "PTC", "HUMNL", "SSGC", "LOTCHEM", "POWER", "OGDC", "SNGP", "PPL",
    "DGKC", "HUBC", "NBP", "SEARL", "EFERT", "HBL", "BAFL", "PSO",
]

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

# TPLRF1 has only 398 rows (listed May 2024) — too short to be useful.
# The two KSE100_index files are handled separately by fix_index(), since
# the index needs its own download rather than a per-ticker raw CSV.
SKIP_FILES = {"TPLRF1.csv", "KSE100_index.csv", "KSE100_index_fixed.csv"}


def prepare_dataframe(df):
    """
    Add Tradeable, DailyReturn, and PriceFlag columns to a stock DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw OHLCV data with a date index (or a date column already set
        as the index), containing at least 'close' and 'volume'.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame, sorted ascending by date with a tz-naive
        DatetimeIndex named 'date', plus three new columns:
        - 'Tradeable': False on days where volume == 0 (halt/suspension)
        - 'DailyReturn': close.pct_change(), rounded to 6 dp (NaN on the
          first row)
        - 'PriceFlag': True where abs(DailyReturn) > 0.20 — a move big
          enough to be a genuine market event or a data error, worth a
          manual look either way.
    """
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
    """
    Clean a single raw stock CSV and save it to cleaned_dir.

    Parameters
    ----------
    raw_path : pathlib.Path
        Path to a raw CSV in data/raw/, e.g. data/raw/ABL.csv. Never
        modified — only read.
    cleaned_dir : str
        Directory to write the cleaned CSV to, under the same filename.

    Returns
    -------
    dict
        {'ticker': str, 'rows': int, 'non_tradeable': int,
        'price_flags': int} — summary stats for this ticker.
    """
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
    """
    Clean every raw stock CSV in raw_dir (skipping SKIP_FILES) and save
    the results to cleaned_dir.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw CSV files, e.g. data/raw.
    cleaned_dir : str
        Directory to write cleaned CSV files to, e.g. data/cleaned.

    Returns
    -------
    list of dict
        One clean_ticker() result dict per file processed.

    Notes
    -----
    Never modifies files in raw_dir — only reads them. Prints a summary
    at the end: files processed, total non-tradeable rows, total
    price-flagged rows, and any ticker with more than 5 price flags (a
    possible data-quality issue worth a manual look).
    """
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


def fix_index(raw_dir=RAW_DIR, cleaned_dir=CLEANED_DIR):
    """
    Download the KSE-100 index via psxdata, clean it, and save both the
    raw and cleaned versions.

    Parameters
    ----------
    raw_dir : str
        Directory to save the raw download to, as KSE100_index_fixed.csv.
    cleaned_dir : str
        Directory to save the cleaned version to, as KSE100_index.csv.

    Raises
    ------
    RuntimeError
        If psxdata returns no data for the KSE100 symbol.

    Notes
    -----
    The raw download is saved under a different name
    (KSE100_index_fixed.csv), not into an existing raw ticker file —
    raw files already on disk are never modified in place.
    """
    from psxdata import PSXClient

    print("Downloading KSE-100 index from PSX via psxdata...")

    client = PSXClient()
    df = client.stocks("KSE100", start="2016-06-01", end="2025-12-31")

    if df.empty:
        raise RuntimeError(
            "psxdata returned no data for symbol 'KSE100'. "
            "The PSX historical endpoint may not serve index price history. "
            "Check https://dps.psx.com.pk/historical directly."
        )

    df = df.set_index("date")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "KSE100_index_fixed.csv"
    df.to_csv(raw_path)
    print(f"Saved raw index to {raw_path} ({len(df)} rows)")

    df_clean = prepare_dataframe(df)

    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    clean_path = cleaned_dir / "KSE100_index.csv"
    df_clean.to_csv(clean_path)

    non_tradeable = int((~df_clean["Tradeable"]).sum())
    price_flags = int(df_clean["PriceFlag"].sum())
    print(
        f"Saved cleaned index to {clean_path} ({len(df_clean)} rows, "
        f"{non_tradeable} non-tradeable, {price_flags} price flags)"
    )


def main():
    """
    Run the full cleaning pipeline: all raw stock CSVs, then the
    KSE-100 index.

    If fix_index() fails (e.g. psxdata can't serve index history), the
    error is printed and the pipeline continues rather than crashing —
    the per-stock cleaning has already succeeded by that point.
    """
    clean_all()
    try:
        fix_index()
    except RuntimeError as error:
        print(f"fix_index() failed: {error}")
        print("The KSE-100 index file will need to be sourced separately.")


def load_data(ticker, data_dir="data/cleaned", end_date="2025-12-31"):
    """
    Load cleaned OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "ABL". The file data/cleaned/ABL.csv is loaded.
    data_dir : str
        Directory containing cleaned CSV files.
    end_date : str
        Rows after this date are truncated, default "2025-12-31" — the
        project's stated data range (see CLAUDE.md). Some cleaned files
        (e.g. KSE100_index.csv) carry rows past this date; this keeps
        every caller on the same declared range without needing to
        remember to slice it themselves.

    Returns
    -------
    pandas.DataFrame
        Indexed by a tz-naive DatetimeIndex, sorted ascending, truncated
        to end_date.

    Raises
    ------
    FileNotFoundError
        If data_dir/ticker.csv does not exist.
    """
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
    """
    Load cleaned OHLCV data for every ticker in data_dir.

    Parameters
    ----------
    data_dir : str
        Directory containing cleaned CSV files.
    exclude : list of str, optional
        Ticker symbols to skip.
    end_date : str
        Passed through to load_data() for every ticker.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Maps ticker symbol to its DataFrame, as returned by load_data.
    """
    exclude = set(exclude) if exclude else set()
    data_dir = Path(data_dir)

    data = {}
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == BENCHMARK_FILE:
            continue

        ticker = path.stem
        if ticker in exclude:
            continue

        data[ticker] = load_data(ticker, data_dir=data_dir, end_date=end_date)

    print(f"Loaded {len(data)} files from {data_dir}")
    return data


def load_selected(data_dir="data/cleaned", end_date="2025-12-31"):
    """
    Load cleaned OHLCV data for the 25 selected tickers (SELECTED_TICKERS).

    Parameters
    ----------
    data_dir : str
        Directory containing cleaned CSV files.
    end_date : str
        Passed through to load_all() (and from there, to load_data()
        for every ticker).

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Maps ticker symbol to its DataFrame, restricted to SELECTED_TICKERS.
    """
    all_data = load_all(data_dir=data_dir, end_date=end_date)
    return {t: all_data[t] for t in SELECTED_TICKERS if t in all_data}


if __name__ == "__main__":
    main()
