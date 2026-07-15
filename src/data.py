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


def load_data(ticker, data_dir="data/cleaned"):
    """
    Load cleaned OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "ABL". The file data/cleaned/ABL.csv is loaded.
    data_dir : str
        Directory containing cleaned CSV files.

    Returns
    -------
    pandas.DataFrame
        Indexed by a tz-naive DatetimeIndex, sorted ascending.

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
    return df


def load_all(data_dir="data/cleaned", exclude=None):
    """
    Load cleaned OHLCV data for every ticker in data_dir.

    Parameters
    ----------
    data_dir : str
        Directory containing cleaned CSV files.
    exclude : list of str, optional
        Ticker symbols to skip.

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

        data[ticker] = load_data(ticker, data_dir=data_dir)

    print(f"Loaded {len(data)} files from {data_dir}")
    return data


def load_selected(data_dir="data/cleaned"):
    """
    Load cleaned OHLCV data for the 25 selected tickers (SELECTED_TICKERS).

    Parameters
    ----------
    data_dir : str
        Directory containing cleaned CSV files.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Maps ticker symbol to its DataFrame, restricted to SELECTED_TICKERS.
    """
    all_data = load_all(data_dir=data_dir)
    return {t: all_data[t] for t in SELECTED_TICKERS if t in all_data}
