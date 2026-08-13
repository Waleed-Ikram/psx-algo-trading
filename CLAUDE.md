# CLAUDE.md — Project Instructions for Claude Code
> This file is read automatically by Claude Code at the start of every session.
> Do not delete it. Update "Current Task" each week.

---

## What this project is

MSc Dissertation — CF981, MSc Computational Finance, University of Essex.
Student: Panagiotis Kanellopoulos. Deadline: 25 August 2026.

**Title:** An Algorithmic Trading and Backtesting System for the Pakistan
Stock Exchange: Evaluating ML-Based Strategies in an Emerging Market

**Research question:** Do classical trading strategies (momentum, mean
reversion) and ML-based prediction retain an edge on the PSX after
realistic transaction costs, slippage, and market constraints are applied
— and does the market's lower efficiency make ML prediction more effective
here than in developed markets?

This is a research project, not a live trading system. The deliverable is:
a backtesting system that evaluates trading strategies on real PSX historical
data, with honest bias-prevention, and a Streamlit dashboard for the demo.

---

## Data

- **Source:** `psxdata` Python library (scrapes PSX directly)
- **Coverage:** ~100 KSE-100 constituent stocks, daily OHLCV
- **Date range:** 27 June 2016 – 31 December 2025
- **Benchmark:** KSE-100 index via yfinance (`^KSE`)
- **Raw files:** `data/raw/TICKER.csv` — never modify these
- **Cleaned files:** `data/cleaned/TICKER.csv`
- **Database:** `data/database/psx.db` (SQLite via sqlalchemy)

### Train / test split
| Period | Dates | Purpose |
|---|---|---|
| Training | 27 Jun 2016 – 31 Dec 2022 | Walk-forward training |
| Out-of-sample | 1 Jan 2023 – 31 Dec 2024 | Held-out test |
| Final eval | 1 Jan 2025 – 31 Dec 2025 | Final reported results |

### Transaction cost assumptions
- Commission: 0.15% per side
- Slippage: 0.1% per side
- Total round-trip cost: 0.5%

---

## Coding rules — follow these always

1. **Never modify files in `data/raw/`** — these are untouched originals
2. **No look-ahead bias** — at any point in time, code must only use data available up to that date
3. **Walk-forward validation only** — never random k-fold on time-series data
4. **Write tests for the backtesting engine** — bugs here silently corrupt all results
5. **Custom backtesting engine** — do not use vectorbt as the main engine (use it only as a cross-check)
6. **Log everything** — cleaning decisions, model parameters, results; everything must be reproducible
7. **Keep notebooks for exploration, src/ for production** — once something works in a notebook, clean it up and move it to src/

---

## Tech stack

- Language: Python 3.11+
- Data: pandas, numpy
- Storage: SQLite + sqlalchemy
- Features: pandas-ta
- Classical models: statsmodels (ARIMA), arch (GARCH)
- ML trees: xgboost, lightgbm, scikit-learn
- Deep learning: PyTorch (LSTM/GRU)
- Backtesting: custom Python class in src/backtest/engine.py
- Evaluation: quantstats, scipy.stats
- Charts: matplotlib, seaborn
- Dashboard: Streamlit + Plotly
- Tests: pytest

---

## Folder structure

```
psx-algo-trading/
├── data/raw/               # Original downloaded CSVs — DO NOT TOUCH
├── data/cleaned/           # Cleaned CSVs
├── data/database/          # SQLite .db file
├── notebooks/              # Exploration and experiments
├── src/
│   ├── data/
│   │   ├── loader.py       # Load raw CSVs
│   │   ├── cleaner.py      # Clean and validate data
│   │   └── database.py     # SQLite read/write
│   ├── features/
│   │   └── engineering.py  # RSI, MACD, momentum, Bollinger Bands
│   ├── strategies/
│   │   ├── base.py         # Abstract base class
│   │   ├── classical.py    # Momentum, mean reversion
│   │   └── ml.py           # ML strategy wrappers
│   ├── models/
│   │   ├── arima_garch.py  # ARIMA / GARCH baselines
│   │   ├── tree_models.py  # XGBoost / LightGBM
│   │   └── lstm.py         # PyTorch LSTM/GRU
│   ├── backtest/
│   │   ├── engine.py       # Core engine — daily loop, P&L
│   │   ├── costs.py        # Transaction costs, slippage
│   │   └── validation.py   # Walk-forward splitter
│   └── evaluation/
│       └── metrics.py      # Sharpe, Sortino, drawdown, tearsheet
├── dashboard/app.py        # Streamlit dashboard
├── tests/                  # pytest unit tests
├── results/figures/        # Charts for the report
├── results/tables/         # CSV results tables
└── dissertation/           # Report drafts
```

---

## Mark scheme (what matters most)

| Component | Weight |
|---|---|
| Quality of Investigation | 30% |
| Quality of Evaluation | 15% |
| Implementation | 20% |
| Report | 25% |
| PDO Presentation | 10% |

Investigation + Evaluation = 45%. This is where the dissertation is won or
lost. Every coding decision should serve the research question, not just
make the code more impressive.

---

## Current task
> Update this section at the start of each new week.

**Week 2 — Data cleaning and storage**

Tasks:
1. Write `src/data/cleaner.py`:
   - Load each CSV from `data/raw/`
   - Drop rows where Volume = 0 or Close = 0
   - Forward-fill missing trading days (Pakistan trading calendar)
   - Flag single-day price moves beyond ±20% as potential outliers
   - Save cleaned files to `data/cleaned/TICKER.csv`
   - Log a summary of changes per stock

2. Write `src/data/database.py`:
   - Load all cleaned CSVs into SQLite at `data/database/psx.db`
   - One table per ticker
   - One combined master table with a ticker column
   - Read functions for use by downstream modules

3. Run a data quality report: rows per stock, date range, missing values,
   any stocks with fewer than 500 rows after cleaning.
