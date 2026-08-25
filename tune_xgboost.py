"""
XGBoost hyperparameter search — training period only (2016-06-27 to
2022-12-31). Never touches OOS or Final Eval.

Stage 1: grid search (max_depth, n_estimators, learning_rate,
scale_pos_weight) on pooled ROC-AUC across all 25 tickers.

Stage 2: takes the Stage 1 winner and sweeps the signal threshold,
comparing gross vs. net backtest performance.

"""

import argparse
import itertools
import time
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.backtest import run_backtest
from src.data import SELECTED_TICKERS, load_data
from src.evaluation import TRAIN_END, TRAIN_START, compute_metrics
from src.features import generate_features
from src.models import XGBOOST_FEATURE_COLS

MIN_TRAIN_DAYS = 504
REFIT_EVERY = 21
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
RANDOM_STATE = 42

COMMISSION = 0.0015
SLIPPAGE = 0.001

GRID = {
    "max_depth": [2, 3, 5],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [1.0, "balanced"],
}
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

TABLES_DIR = Path("results/tables")
HYPERPARAM_SEARCH_PATH = TABLES_DIR / "hyperparameter_search.csv"
THRESHOLD_SWEEP_PATH = TABLES_DIR / "threshold_sweep.csv"

XGBParams = namedtuple(
    "XGBParams", ["max_depth", "n_estimators", "learning_rate", "scale_pos_weight"]
)

# Stage 1's winner (roc_auc=0.5246), so --stage2-only has one source of
# truth instead of a magic literal. Update if Stage 1 is re-run.
STAGE1_WINNER = XGBParams(
    max_depth=2, n_estimators=50, learning_rate=0.05, scale_pos_weight=1.0
)


def _resolve_scale_pos_weight(mode, y_train):
    """'balanced' -> n_negative/n_positive from this refit's labels; a float is used as-is."""
    if mode == "balanced":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0:
            return n_neg / n_pos
        else:
            return 1.0
    return float(mode)


def _walk_forward_probs(ticker, params, train_end, min_train_days, refit_every):
    """
    Same walk-forward loop as xgboost_signals(), reimplemented standalone
    so it never touches the results/xgboost_signals/ cache. Returns
    (dates, probs, labels), aligned arrays; labels is NaN for the last
    row (its true outcome falls outside the truncated training period).
    """
    df = generate_features(load_data(ticker, end_date=train_end))

    features_full = df[XGBOOST_FEATURE_COLS]
    next_day_return = df["close"].shift(-1) / df["close"] - 1
    y_full = (next_day_return > 0).astype(int)
    y_full_raw = next_day_return  # to detect the unknown-label tail row

    n = len(df)
    model = None

    dates, probs, labels = [], [], []

    for T in range(min_train_days, n):
        try:
            if (T - min_train_days) % refit_every == 0:
                # Rows 0..T-1 only — label i needs close[i+1], so this
                # slice never trains on information from day T.
                train_slice = pd.concat(
                    [features_full.iloc[0:T], y_full.iloc[0:T].rename("y")],
                    axis=1,
                ).dropna()

                spw = _resolve_scale_pos_weight(
                    params.scale_pos_weight, train_slice["y"]
                )

                model = XGBClassifier(
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                    learning_rate=params.learning_rate,
                    subsample=SUBSAMPLE,
                    colsample_bytree=COLSAMPLE_BYTREE,
                    scale_pos_weight=spw,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        train_slice[XGBOOST_FEATURE_COLS], train_slice["y"]
                    )

            proba_up = model.predict_proba(features_full.iloc[[T]])[0, 1]
        except Exception:
            model = None
            continue

        if pd.isna(y_full_raw.iloc[T]):
            label = np.nan
        else:
            label = int(y_full.iloc[T])

        dates.append(df.index[T])
        probs.append(proba_up)
        labels.append(label)

    return np.array(dates), np.array(probs), np.array(labels, dtype=float)


def _stage1_worker(ticker, params, train_end, min_train_days, refit_every):
    """One ticker's (prob, label) pairs for days with a known label — feeds the pooled ROC-AUC."""
    _, probs, labels = _walk_forward_probs(
        ticker, params, train_end, min_train_days, refit_every
    )
    known = ~np.isnan(labels)
    return probs[known], labels[known].astype(int)


def _stage2_worker(ticker, params, train_end, min_train_days, refit_every):
    """One ticker's full probability series — thresholding happens later, so the model is trained once here, not once per threshold."""
    dates, probs, _ = _walk_forward_probs(
        ticker, params, train_end, min_train_days, refit_every
    )
    return pd.Series(probs, index=pd.DatetimeIndex(dates))


def run_stage1(tickers):
    print("=== STAGE 1: hyperparameter grid search (prediction quality) ===")
    print(f"Training period only: {TRAIN_START} to {TRAIN_END}")
    print(f"Grid size: 36 combinations x {len(tickers)} tickers\n")

    combos = list(itertools.product(
        GRID["max_depth"], GRID["n_estimators"],
        GRID["learning_rate"], GRID["scale_pos_weight"],
    ))

    rows = []
    t_start = time.time()

    for i, (max_depth, n_estimators, learning_rate, spw_mode) in enumerate(combos, start=1):
        params = XGBParams(max_depth, n_estimators, learning_rate, spw_mode)
        t0 = time.time()

        all_probs = []
        all_labels = []
        for ticker in tickers:
            probs, labels = _stage1_worker(
                ticker, params, TRAIN_END, MIN_TRAIN_DAYS, REFIT_EVERY
            )
            all_probs.append(probs)
            all_labels.append(labels)

        pooled_probs = np.concatenate(all_probs)
        pooled_labels = np.concatenate(all_labels)

        roc_auc = roc_auc_score(pooled_labels, pooled_probs)
        accuracy = float(np.mean(pooled_labels == (pooled_probs > 0.5)))
        base_rate = float(np.mean(pooled_labels))

        elapsed = time.time() - t0
        print(
            f"[{i:2d}/36] max_depth={max_depth} n_estimators={n_estimators} "
            f"learning_rate={learning_rate} scale_pos_weight={spw_mode} "
            f"-> roc_auc={roc_auc:.4f} accuracy={accuracy:.4f} "
            f"({elapsed:.1f}s, n={len(pooled_labels)})"
        )

        rows.append({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "scale_pos_weight": spw_mode,
            "roc_auc": round(roc_auc, 4),
            "accuracy": round(accuracy, 4),
            "base_rate": round(base_rate, 4),
        })

    total_elapsed = time.time() - t_start
    print(f"\nStage 1 total elapsed: {total_elapsed / 60:.1f} min")

    return pd.DataFrame(rows)


def run_stage2(tickers, winning_params):
    print("\n=== STAGE 2: threshold sweep (winning hyperparameters only) ===")
    print(f"Parameters: {winning_params._asdict()}")
    print(f"Thresholds: {THRESHOLDS}\n")

    t_start = time.time()

    print("Generating walk-forward probabilities for all 25 tickers "
          "(once, reused across every threshold)...")
    prob_by_ticker = {}
    for ticker in tickers:
        prob_by_ticker[ticker] = _stage2_worker(
            ticker, winning_params, TRAIN_END, MIN_TRAIN_DAYS, REFIT_EVERY
        )
    print(f"Probabilities ready for {len(prob_by_ticker)} tickers "
          f"({time.time() - t_start:.1f}s so far).\n")

    print("Reloading truncated+featured frames for backtesting...")
    df_by_ticker = {}
    for ticker in tickers:
        df_by_ticker[ticker] = generate_features(load_data(ticker, end_date=TRAIN_END))

    rows = []
    for t in THRESHOLDS:
        t0 = time.time()
        net_metrics_list, gross_metrics_list = [], []

        for ticker in tickers:
            df = df_by_ticker[ticker]
            aligned_probs = prob_by_ticker[ticker].reindex(df.index)

            # NaN compares False either way -> Signal defaults to 0 (hold)
            buy = aligned_probs > t
            sell = aligned_probs < (1 - t)

            sig_df = df.copy()
            sig_df["Signal"] = np.select([buy, sell], [1, -1], default=0)

            net_results = run_backtest(
                sig_df, sig_df, commission=COMMISSION, slippage=SLIPPAGE
            )
            gross_results = run_backtest(
                sig_df, sig_df, commission=0, slippage=0
            )

            net_metrics_list.append(compute_metrics(net_results))
            gross_metrics_list.append(compute_metrics(gross_results))

        net_df = pd.DataFrame(net_metrics_list)
        gross_df = pd.DataFrame(gross_metrics_list)

        # A ticker with no trades at this threshold gets NaN Sharpe
        # (see compute_metrics); .mean(skipna=True) excludes it.
        n_nan_sharpe = int(net_df["Sharpe Ratio"].isna().sum())
        n_valid_sharpe = len(net_df) - n_nan_sharpe

        row = {
            "threshold": t,
            "mean_turnover": round(net_df["Turnover (trades/yr)"].mean(), 1),
            "mean_gross_return": round(gross_df["Annualised Return"].mean(), 2),
            "mean_net_return": round(net_df["Annualised Return"].mean(), 2),
            "mean_sharpe": round(net_df["Sharpe Ratio"].mean(skipna=True), 3),
            "n_nan_sharpe": n_nan_sharpe,
        }
        rows.append(row)

        elapsed = time.time() - t0
        print(
            f"  threshold={t:.2f}: mean_turnover={row['mean_turnover']} "
            f"mean_gross_return={row['mean_gross_return']} "
            f"mean_net_return={row['mean_net_return']} "
            f"mean_sharpe={row['mean_sharpe']} "
            f"(from {n_valid_sharpe}/{len(net_df)} tickers, "
            f"{n_nan_sharpe} excluded as NaN — zero trades) ({elapsed:.1f}s)"
        )

    total_elapsed = time.time() - t_start
    print(f"\nStage 2 total elapsed: {total_elapsed / 60:.1f} min")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-only", action="store_true",
        help=(
            "Skip Stage 1 entirely and re-run only the Stage 2 threshold "
            "sweep, reusing the previously selected winning hyperparameters "
            f"({STAGE1_WINNER._asdict()}). Does not read or write "
            "hyperparameter_search.csv."
        ),
    )
    args = parser.parse_args()

    script_start = time.time()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tickers = list(SELECTED_TICKERS)
    print(f"Tickers: {len(tickers)}")
    print(f"Training period cutoff: {TRAIN_END} "
          f"(OOS 2023-2024 and Final Eval 2025 are never loaded)\n")

    if args.stage2_only:
        winning_params = STAGE1_WINNER
        print(
            f"--stage2-only: skipping Stage 1, reusing previously selected "
            f"winning parameters: {winning_params._asdict()}\n"
        )
    else:
        stage1_df = run_stage1(tickers)
        stage1_df.to_csv(HYPERPARAM_SEARCH_PATH, index=False)
        print(f"\nSaved {HYPERPARAM_SEARCH_PATH}")

        ranked = stage1_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        print("\nTop 5 by roc_auc:")
        print(ranked.head(5).to_string(index=False))
        print("\nBottom 5 by roc_auc:")
        print(ranked.tail(5).to_string(index=False))
        print(f"\nBase rate (pooled positive-class fraction, should be ~constant "
              f"across combinations): {ranked['base_rate'].mean():.4f}")

        winner_row = ranked.iloc[0]
        if winner_row["scale_pos_weight"] == "balanced":
            winner_scale_pos_weight = "balanced"
        else:
            winner_scale_pos_weight = float(winner_row["scale_pos_weight"])

        winning_params = XGBParams(
            max_depth=int(winner_row["max_depth"]),
            n_estimators=int(winner_row["n_estimators"]),
            learning_rate=float(winner_row["learning_rate"]),
            scale_pos_weight=winner_scale_pos_weight,
        )
        print(f"\nSelected Stage 1 winner: {winning_params._asdict()} "
              f"(roc_auc={winner_row['roc_auc']:.4f})")

    stage2_df = run_stage2(tickers, winning_params)
    stage2_df.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    print(f"\nSaved {THRESHOLD_SWEEP_PATH}")

    print("\nFull Stage 2 table:")
    print(stage2_df.to_string(index=False))

    total_elapsed = time.time() - script_start
    minutes, seconds = divmod(total_elapsed, 60)
    print(f"\nTotal elapsed time: {int(minutes)}m {seconds:.1f}s")


if __name__ == "__main__":
    main()
