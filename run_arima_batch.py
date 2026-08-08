"""
Populate the ARIMA signal cache (results/arima_signals/) for all 25
selected tickers. Signals only — compute_metrics() is not run here;
see src/models.py for the Option A caching rationale.
"""

from src.data import SELECTED_TICKERS, load_selected
from src.models import run_arima_all


def run_arima_batch():
    print(f"Loading {len(SELECTED_TICKERS)} selected tickers...")
    stock_dict = load_selected()
    print(f"Loaded {len(stock_dict)} tickers: {', '.join(stock_dict.keys())}\n")

    return run_arima_all(stock_dict)


if __name__ == "__main__":
    run_arima_batch()
