"""Download daily OHLCV data for all KSE-100 constituents and the KSE-100 index."""

import time
from pathlib import Path

import psxdata
import yfinance as yf
from psxdata.exceptions import PSXDataError

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
REQUEST_DELAY = 0.5
RAW_DIR = Path(__file__).parent / "data" / "raw"


def download_stocks() -> list[str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = psxdata.tickers(index="KSE100")
    print(f"Found {len(tickers)} KSE-100 constituents.")

    failed = []
    for ticker in tickers:
        out_path = RAW_DIR / f"{ticker}.csv"
        if out_path.exists():
            print(f"Skipping {ticker} (already downloaded).")
            continue

        try:
            df = psxdata.stocks(ticker, start=START_DATE, end=END_DATE)
            if df.empty:
                print(f"No data returned for {ticker}.")
                failed.append(ticker)
                continue
            df.to_csv(out_path, index=False)
            print(f"Saved {ticker} ({len(df)} rows).")
        except PSXDataError as e:
            print(f"Failed to download {ticker}: {e}")
            failed.append(ticker)

        time.sleep(REQUEST_DELAY)

    return failed


def download_index():
    out_path = RAW_DIR / "KSE100_index.csv"
    df = yf.Ticker("^KSE").history(start=START_DATE, end=END_DATE)
    df.to_csv(out_path)
    print(f"Saved KSE-100 index ({len(df)} rows).")


def main():
    failed = download_stocks()
    download_index()

    print("\n--- Download summary ---")
    if failed:
        print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")
    else:
        print("All tickers downloaded successfully.")


if __name__ == "__main__":
    main()
