# NASDAQ Order-Book Microstructure Research

Event-level market-microstructure research using public LOBSTER sample data, with a later extension to raw NASDAQ TotalView-ITCH.

**Current status:** reproducible baseline study completed for 10-event and
50-event horizons. Results show statistically predictive order-book signals,
but quoted-spread costs overwhelm the estimated gross edge. Additional
horizons, non-linear models and robustness analysis are in progress.

## Research question

Does order-flow imbalance contain incremental information about short-horizon mid-price movement after accounting for spread, depth, event intensity, liquidity regime, and simple execution costs?

## Current findings

The initial experiment uses the complete public AAPL 10-level LOBSTER sample for June 21, 2012. Models are evaluated using purged chronological train, validation and exploratory holdout blocks with no random row shuffling.

|   Horizon | Majority accuracy | Logistic accuracy | Balanced accuracy | Macro F1 | Ridge rank IC | Ridge directional accuracy | Gross simulation | Net simulation |
| --------: | ----------------: | ----------------: | ----------------: | -------: | ------------: | -------------------------: | ---------------: | -------------: |
| 10 events |            38.38% |            38.86% |            33.90% |   20.01% |         0.276 |                     40.21% |         +213 bps |     −3,191 bps |
| 50 events |            49.53% |            57.86% |            42.90% |   42.43% |         0.288 |                     57.20% |         +296 bps |     −1,527 bps |

The 50-event horizon produces a substantially stronger classification result than the 10-event horizon. At 10 events, the classifier largely defaults to the unchanged-price class and provides little improvement over the majority baseline.

Both horizons exhibit positive gross simulated returns, indicating that the order-book features contain some short-horizon ranking information. However, the forecast edge is considerably smaller than the cost of aggressively crossing the quoted spread. Consequently, both simulations produce negative returns after estimated transaction costs.

These results support the distinction between statistical price predictability and executable trading alpha. They should not be interpreted as evidence of a deployable trading strategy.

The experiment currently uses one stock and one trading day. The final holdout is therefore an intraday exploratory holdout rather than evidence of out-of-day or cross-asset generalisation.


## Initial data path

The first reproducible experiment uses the public AAPL LOBSTER sample:

- date: 2012-06-21;
- session: 09:30:00–16:00:00;
- depth: 10 occupied price levels;
- two row-aligned, headerless CSV files:
  - message events;
  - order-book snapshots after each event.

The raw files are downloaded from a public Hugging Face mirror. They are not committed to Git.

## What the repository builds

1. Raw-file ingestion with explicit schemas and fixed-point price conversion.
2. Message/snapshot row-alignment and quality checks.
3. A limited-depth price-level reconstruction audit.
4. Event-level order-book and order-flow features.
5. Future mid-price targets at 10, 50, and 100 event horizons.
6. Purged chronological train/validation/test blocks.
7. Majority, logistic-regression, Ridge, and later LightGBM baselines.
8. Statistical and cost-aware economic evaluation.
9. Failure analysis by spread, depth, event intensity, and time of day.
10. A short research note and reproducible chart pack.

## Important terminology

LOBSTER already reconstructs the order-book snapshots from NASDAQ ITCH. In the LOBSTER phase, this repository:

- validates message-to-snapshot transitions;
- reconstructs a limited price-level state from the supplied event stream;
- engineers predictive features from aligned messages and snapshots.

A true full-depth reconstruction from raw binary ITCH is a separate later phase.

## Repository structure

```text
configs/                    Experiment settings
data/                       Local-only raw and processed data
docs/                       Research design and report outline
models/                     Generated model artefacts
notebooks/                  Auditable exploration
reports/figures/            Generated charts
reports/tables/             Metrics and diagnostics
scripts/                    Windows PowerShell entry points
src/orderbook_research/     Reusable pipeline code
tests/                      Synthetic unit tests
```

## Windows setup

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
.venv\Scripts\Activate.ps1
```

## Download the public AAPL sample

```powershell
.\scripts\download_lobster.ps1
```

Expected local files:

```text
data/raw/lobster/AAPL_10/
AAPL_2012-06-21_34200000_57600000_message_10.csv
AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
```

No Kaggle account, API key, or Hugging Face account is required for the public repository.

## Run Phase 0 checks

```powershell
python -m orderbook_research.audit --levels 10 --max-rows 100000
pytest
python -m orderbook_research.smoke_test
```

The audit writes:

```text
reports/tables/data_audit.json
```

## Run the first baseline experiment

Start with 150,000 rows to verify runtime and outputs:

```powershell
python -m orderbook_research.train_baseline `
    --levels 10 `
    --horizon 50 `
    --max-rows 150000
```

Then run the full day by omitting `--max-rows`.

## Leakage controls

- Rows are never randomly shuffled before splitting.
- Splits are contiguous in event time.
- A purge gap equal to the maximum target horizon separates blocks.
- All imputers and scalers are fit on training rows only.
- Future mid-price values are used only in target construction.
- The final test block is not used for model or threshold selection.
- Rolling features use current and past observable events only.

## One-day limitation

The public sample provides one date. Therefore, a chronological intraday split tests later-time performance but does not establish day-to-day stability. The final report must state this prominently. Stronger validation requires additional dates from licensed LOBSTER data or reconstructed ITCH samples.

## Data and licensing

This repository contains code only. Review the source dataset's terms before redistributing data. Do not commit the raw CSV files.
