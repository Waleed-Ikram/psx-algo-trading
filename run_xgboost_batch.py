"""
Populate the XGBoost signal cache (results/xgboost_signals/) for all 25
selected tickers. Signals only — compute_metrics() is not run here;
see src/models.py for the Option A caching rationale.
"""

from src.data import SELECTED_TICKERS, load_selected
from src.features import generate_features
from src.models import run_xgboost_all


def run_xgboost_batch():
    print(f"Loading {len(SELECTED_TICKERS)} selected tickers...")
    stock_dict = load_selected()
    print(f"Loaded {len(stock_dict)} tickers: {', '.join(stock_dict.keys())}\n")

    print("Generating features for all tickers...")
    featured_dict = {
        ticker: generate_features(df) for ticker, df in stock_dict.items()
    }
    print("Done.\n")

    return run_xgboost_all(featured_dict)


if __name__ == "__main__":
    run_xgboost_batch()
