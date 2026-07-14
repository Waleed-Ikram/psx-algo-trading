"""
src/data/cleaner.py

Reads every raw stock CSV from data/raw/, adds three new columns, and
saves the result to data/cleaned/.

New columns added to each file:
  Tradeable   - False on days where volume == 0 (trading halt or suspension)
  DailyReturn - percentage change in the closing price (NaN on the first row)
  PriceFlag   - True when the daily move is more than +/- 20%

Run from the project root:
    python -m src.data.cleaner
"""

import logging
from pathlib import Path

import pandas as pd
from psxdata import PSXClient


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Walk up two levels from this file (src/data/cleaner.py) to reach the
# project root, then point at the data folders.
REPO_ROOT   = Path(__file__).resolve().parents[2]
RAW_DIR     = REPO_ROOT / "data" / "raw"
CLEANED_DIR = REPO_ROOT / "data" / "cleaned"

# These two files are skipped by clean_all():
#   TPLRF1.csv      - only 398 rows (listed May 2024), too short to be useful
#   KSE100_index.csv - the old yfinance download; rebuild it with fix_index()
SKIP_FILES = ["TPLRF1.csv", "KSE100_index.csv", "KSE100_index_fixed.csv"]

# Set up a logger so progress is written to the console with a timestamp.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core cleaning logic
# ---------------------------------------------------------------------------

def prepare_dataframe(df):
    """Add Tradeable, DailyReturn, and PriceFlag columns to a stock DataFrame.

    Also makes sure the date index is sorted oldest-to-newest and has no
    timezone information attached.
    """

    # Make sure the index is a proper DatetimeIndex (not just strings).
    # parse_dates=True in read_csv usually handles this, but we do it here
    # as a safety net.
    df.index = pd.to_datetime(df.index)

    # Remove timezone info if present (the KSE100 index download has it).
    # Stock CSVs from psxdata are already timezone-naive, so this is a no-op
    # for them, but it doesn't hurt to always run it.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Give the index a clear name so downstream code can reference it easily.
    df.index.name = "date"

    # Sort rows so the earliest date is at the top.
    df = df.sort_index()

    # --- Tradeable ---
    # A day with zero volume means the stock was not traded (halted or
    # suspended). We flag it False so strategies can skip these days.
    df["Tradeable"] = df["volume"] != 0

    # --- DailyReturn ---
    # pct_change() calculates (today - yesterday) / yesterday for each row.
    # The very first row has no previous row, so it stays NaN.
    df["DailyReturn"] = df["close"].pct_change()
    df["DailyReturn"] = df["DailyReturn"].round(6)

    # --- PriceFlag ---
    # A move bigger than 20% in a single day is suspicious. It could be a
    # genuine market event (circuit breaker, rights issue) or a data error.
    # We flag it True so researchers can inspect it manually.
    df["PriceFlag"] = df["DailyReturn"].abs() > 0.20

    return df


# ---------------------------------------------------------------------------
# Clean a single ticker
# ---------------------------------------------------------------------------

def clean_ticker(raw_path, cleaned_dir):
    """Load one raw CSV, clean it, save it, and return summary statistics.

    Returns a plain dictionary with four keys:
        ticker        - the stock symbol (e.g. "ENGRO")
        rows          - total number of rows in the cleaned file
        non_tradeable - number of days where Volume was 0
        price_flags   - number of days with a move larger than 20%
    """

    ticker = raw_path.stem  # e.g. "data/raw/ABL.csv" -> "ABL"

    # Load the CSV. index_col=0 makes the first column (date) the row index.
    # parse_dates=True tells pandas to convert that column to datetime objects.
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # Apply our cleaning and new columns.
    df = prepare_dataframe(df)

    # Save to data/cleaned/ with the same filename.
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    output_path = cleaned_dir / raw_path.name
    df.to_csv(output_path)

    # Count non-tradeable rows: where Tradeable is False.
    non_tradeable_count = int((df["Tradeable"] == False).sum())

    # Count price-flagged rows: where PriceFlag is True.
    price_flag_count = int(df["PriceFlag"].sum())

    logger.info(
        "%s: %d rows, %d non-tradeable, %d price flags",
        ticker, len(df), non_tradeable_count, price_flag_count
    )

    return {
        "ticker":        ticker,
        "rows":          len(df),
        "non_tradeable": non_tradeable_count,
        "price_flags":   price_flag_count,
    }


# ---------------------------------------------------------------------------
# Clean all stock CSVs
# ---------------------------------------------------------------------------

def clean_all(raw_dir=RAW_DIR, cleaned_dir=CLEANED_DIR):
    """Clean every stock CSV in raw_dir (skipping files in SKIP_FILES).

    Saves one cleaned file per stock to cleaned_dir, then prints a summary.
    Returns a list of stat dictionaries, one per file processed.
    """

    # Build the list of CSV files to process.
    all_csv_files = sorted(raw_dir.glob("*.csv"))

    csv_files = []
    for f in all_csv_files:
        if f.name not in SKIP_FILES:
            csv_files.append(f)

    logger.info("Found %d CSV files to clean (skipping: %s)", len(csv_files), SKIP_FILES)

    # Process each file and collect the stats.
    all_stats = []
    for csv_file in csv_files:
        stats = clean_ticker(csv_file, cleaned_dir)
        all_stats.append(stats)

    # ------------------------------------------------------------------
    # Print a summary after all files are processed.
    # ------------------------------------------------------------------

    total_non_tradeable = 0
    for s in all_stats:
        total_non_tradeable = total_non_tradeable + s["non_tradeable"]

    total_price_flags = 0
    for s in all_stats:
        total_price_flags = total_price_flags + s["price_flags"]

    # Find stocks where the number of price flags is greater than 5.
    # These might have genuine data quality problems worth investigating.
    high_flag_stocks = []
    for s in all_stats:
        if s["price_flags"] > 5:
            high_flag_stocks.append(s)

    # Sort high_flag_stocks from most flags to fewest.
    high_flag_stocks.sort(key=lambda s: s["price_flags"], reverse=True)

    print()
    print("-" * 58)
    print("Cleaning Summary")
    print("-" * 58)
    print(f"  Files processed               : {len(all_stats)}")
    print(f"  Volume=0 rows (non-tradeable) : {total_non_tradeable}")
    print(f"  PriceFlag rows (|ret| > 20%)  : {total_price_flags}")

    if len(high_flag_stocks) > 0:
        print()
        print("  Stocks with PriceFlag > 5 (potential data quality issues):")
        for s in high_flag_stocks:
            print(f"    {s['ticker']:<14} {s['price_flags']:>3} flags")
    else:
        print()
        print("  No stocks with PriceFlag > 5.")

    print("-" * 58)
    print()

    return all_stats


# ---------------------------------------------------------------------------
# Download and clean the KSE-100 index
# ---------------------------------------------------------------------------

def fix_index(raw_dir=RAW_DIR, cleaned_dir=CLEANED_DIR):
    """Download the KSE-100 index via psxdata and save cleaned CSV files.

    Saves two files:
      data/raw/KSE100_index_fixed.csv  - the raw download (never modify)
      data/cleaned/KSE100_index.csv    - the cleaned version with new columns

    Raises RuntimeError if psxdata returns no data for the KSE100 symbol.
    """

    logger.info("Downloading KSE-100 index from PSX via psxdata ...")

    client = PSXClient()
    df = client.stocks("KSE100", start="2016-06-01")

    if df.empty:
        raise RuntimeError(
            "psxdata returned no data for symbol 'KSE100'. "
            "The PSX historical endpoint may not serve index price history. "
            "Check https://dps.psx.com.pk/historical directly."
        )

    # psxdata returns 'date' as a regular column; move it to the index.
    df = df.set_index("date")

    # Save the raw download - we never modify files in data/raw/.
    raw_output = raw_dir / "KSE100_index_fixed.csv"
    df.to_csv(raw_output)
    logger.info("Saved raw index to %s (%d rows)", raw_output.name, len(df))

    # Apply the same cleaning as every other stock.
    df_clean = prepare_dataframe(df.copy())

    # Save the cleaned version.
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    clean_output = cleaned_dir / "KSE100_index.csv"
    df_clean.to_csv(clean_output)

    non_tradeable_count = int((df_clean["Tradeable"] == False).sum())
    price_flag_count    = int(df_clean["PriceFlag"].sum())

    logger.info(
        "Saved cleaned index to %s (%d rows, %d non-tradeable, %d price flags)",
        clean_output.name, len(df_clean), non_tradeable_count, price_flag_count
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full cleaning pipeline: all stocks first, then the index."""

    # Step 1: clean all 97 stock CSVs.
    clean_all()

    # Step 2: download and clean the KSE-100 index.
    # If psxdata can't serve the index data, we log a warning and continue
    # rather than crashing the whole pipeline.
    try:
        fix_index()
    except RuntimeError as error:
        logger.warning("fix_index() failed: %s", error)
        logger.warning(
            "The KSE-100 index file will need to be sourced separately."
        )


if __name__ == "__main__":
    main()
