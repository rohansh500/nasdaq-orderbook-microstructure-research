# NASDAQ Order-Book Microstructure Research

Event-level market-microstructure research using public LOBSTER sample data, with a later extension to raw NASDAQ TotalView-ITCH.

**Current status:** Version 0.2 baseline study completed for 10-, 50-, and 100-event horizons. The repository now includes explicit LOBSTER schemas, one-step message/snapshot reconstruction checks, 35 microstructure features, purged chronological splits, majority and class-balanced logistic classifiers, zero-return and Ridge regression baselines, and spread-aware signal simulation. Phase A—purged walk-forward validation and econometric diagnostics—is next.

## Research question

Does order-flow imbalance contain incremental information about short-horizon mid-price movement after accounting for spread, depth, event intensity, liquidity regime, and simple execution costs?

## Current findings

The initial experiment uses the complete public AAPL 10-level LOBSTER sample for June 21, 2012. Models are evaluated using contiguous purged chronological training, validation, and exploratory-holdout blocks. Rows are never randomly shuffled.

### Classification results on the exploratory holdout

| Horizon | Flat target share | Majority accuracy | Balanced-logistic accuracy | Balanced accuracy | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 10 events | 38.38% | 38.38% | 38.86% | 33.90% | 20.01% |
| 50 events | 8.39% | 49.53% | 51.52% | 44.04% | 43.07% |
| 100 events | 4.13% | 51.89% | 53.38% | 41.78% | 40.92% |

The 10-event classifier provides little improvement over the majority baseline and remains dominated by the unchanged-price class. The 50-event horizon has the strongest balanced accuracy and macro F1, while the 100-event horizon has the highest raw accuracy. Because class proportions change sharply with the forecast horizon, raw accuracy is not sufficient by itself.

### Regression and execution results on the exploratory holdout

| Horizon | Zero-return MAE (bps) | Ridge MAE (bps) | Ridge rank IC | Non-zero directional accuracy | Gross edge per active signal (bps) | Estimated cost per active signal (bps) | Net edge per active signal (bps) | Break-even quoted-cost fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 events | 0.189 | 0.198 | 0.276 | 65.25% | 0.114 | 1.818 | -1.704 | 6.27% |
| 50 events | 0.555 | 0.540 | 0.288 | 62.44% | 0.295 | 1.795 | -1.500 | 16.45% |
| 100 events | 0.878 | 0.867 | 0.247 | 57.87% | 0.389 | 1.825 | -1.438 | 21.36% |

The Ridge model has positive rank information at all three horizons. The 50-event horizon produces the strongest rank IC and improves MAE over the zero-return baseline. At 10 events, Ridge improves RMSE and ranking but does not improve MAE, showing why multiple metrics and naive baselines are necessary.

Gross edge per active signal rises with the prediction horizon, but estimated spread costs remain much larger than the forecast edge. Even at 100 events, the signal would break even only if realised execution costs were approximately 21% of the quoted aggressive-crossing estimate.

The current evidence therefore supports a limited conclusion:

> Order-book and order-flow features contain short-horizon predictive information, but the baseline signal is not economically executable under aggressive quoted-spread assumptions.

These results should not be interpreted as evidence of a deployable trading strategy. They are based on one stock and one trading day. The final block is an intraday exploratory holdout, not an untouched final test or evidence of out-of-day and cross-asset generalisation.

## Purged walk-forward validation

To test whether the baseline results were concentrated in one favourable
intraday period, the models were evaluated across five expanding-window
folds using only the first 80% of the trading day. Each fold used a
100-event purge between training and validation.

| Horizon | Mean balanced accuracy | Mean rank IC | Rank IC positive folds | Mean non-zero directional accuracy | Ridge MAE improvement positive folds | Mean break-even cost fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 10 events | 36.56% | 0.198 | 5/5 | 60.12% | 0/5 | 3.69% |
| 50 events | 37.90% | 0.211 | 5/5 | 57.06% | 4/5 | 8.70% |
| 100 events | 34.97% | 0.164 | 5/5 | 54.03% | 2/5 | 10.76% |

Rank IC remained positive in every fold and horizon, indicating that the
order-book features retained some ability to rank future mid-price
movements across different parts of the day. The 50-event horizon provided
the strongest average statistical performance, while the 10-event horizon
was the most stable and the 100-event horizon produced the largest gross
edge per active signal.

The first walk-forward fold was materially weaker at the 50- and 100-event
horizons. This suggests that a model trained mainly on opening-session
dynamics does not transfer cleanly into the quieter midday regime.
Performance improved as the expanding training window incorporated a
broader range of intraday conditions.

All horizons produced positive mean gross edge in every fold, but none
covered the estimated quoted-spread cost. The results therefore support
short-horizon statistical predictability, not a directly executable
aggressive trading strategy.

### Moving-block bootstrap uncertainty

Moving-block bootstrap inference was applied to each walk-forward fold using
1,000-event blocks, 1,000 bootstrap draws and 95% percentile confidence
intervals. The final exploratory holdout remained unused.

| Horizon | Rank IC, 95% CI | Ridge MAE improvement, 95% CI | Non-zero directional accuracy, 95% CI | Gross edge per active signal, 95% CI | Net edge per active signal, 95% CI |
|---:|---:|---:|---:|---:|---:|
| 10 events | 0.198 [0.182, 0.210] | -5.93% [-6.66%, -5.33%] | 60.12% [59.31%, 60.90%] | 0.088 [0.077, 0.096] bps | -2.351 [-2.434, -2.265] bps |
| 50 events | 0.211 [0.183, 0.236] | -0.06% [-0.99%, 0.86%] | 57.07% [55.62%, 58.27%] | 0.215 [0.173, 0.251] bps | -2.352 [-2.447, -2.251] bps |
| 100 events | 0.164 [0.126, 0.194] | -3.09% [-4.74%, -1.50%] | 54.04% [52.35%, 55.63%] | 0.245 [0.164, 0.314] bps | -2.287 [-2.405, -2.162] bps |

Rank IC and non-zero directional accuracy remained above their respective
zero and 50% benchmarks at the horizon-mean level. Gross edge was also
positive at all three horizons. These findings support the presence of
short-horizon predictive information in the order-book features.

Point-return calibration was weaker. Ridge regression was conclusively
worse than the zero-return MAE baseline at 10 and 100 events. At 50 events,
the confidence interval crossed zero, so the evidence was insufficient to
conclude that Ridge improved average absolute error.

The estimated break-even fractions of quoted-spread cost were 3.72%,
8.74% and 10.36% at the 10-, 50- and 100-event horizons respectively.
All net-return confidence intervals remained below zero. The signal is
therefore statistically informative but insufficient for repeated
aggressive spread-crossing execution.

These intervals are conditional on the fitted models, the selected
1,000-event block length and one AAPL trading day. They do not establish
out-of-day or cross-asset generalisation.

### Econometric and residual diagnostics

Econometric diagnostics were applied across the same five purged
walk-forward folds. Residual autocorrelation was calculated using
non-overlapping horizon-spaced observations.

| Horizon | Mean return rank IC | Non-zero directional accuracy | Mean calibration slope | Residual lag-1 ACF | Ljung-Box rejections at lag 20 | Time buckets with positive rank IC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 events | 0.198 | 60.12% | 0.745 | -0.097 | 5/5 | 9/9 |
| 50 events | 0.211 | 57.06% | 0.727 | -0.032 | 2/5 | 9/9 |
| 100 events | 0.164 | 54.03% | 0.683 | -0.019 | 2/5 | 9/9 |

One-event mid-price returns displayed negative lag-one autocorrelation of
approximately -0.112, consistent with short-lived microstructure reversal.
Squared returns displayed positive lag-one autocorrelation of approximately
0.156, indicating short-horizon volatility clustering.

Calibration slopes were below one at every horizon, showing that the Ridge
forecasts were more useful for ranking outcomes than for estimating return
magnitudes. The 10-event residuals also failed the Ljung-Box independence
test in all five folds, suggesting that the linear feature representation
left systematic very-short-horizon temporal structure unexplained.

Performance varied materially through the day. The 11:00-12:00 period was
particularly weak for longer horizons, while the 50-event model produced
its strongest MAE and directional results after 14:30. This supports the
presence of intraday regime dependence and motivates explicit spread,
liquidity and volatility-regime analysis.

The 50-event horizon remains the primary research horizon because it offers
the strongest overall balance of ranking performance, point-forecast
accuracy and residual stability.

## Methodological grounding

This is an independent research project that applies methods developed through MSc Financial Engineering training:

- **Financial data engineering:** explicit schemas, fixed-point price conversion, missing-level handling, transformation, feature construction, and reproducible data-quality checks.
- **Financial econometrics:** return targets, correlation-based signal evaluation, chronological dependence controls, volatility features, and planned residual and bootstrap diagnostics.
- **Machine learning in finance:** regularised linear models, class imbalance handling, training-only preprocessing, purged validation, and comparison against defensible naive baselines.
- **Risk management:** gross-versus-net attribution, quoted-spread cost stress, break-even execution analysis, drawdown, and planned VaR and expected-shortfall diagnostics.
- **Model-risk discipline:** failure analysis and explicit separation between statistical predictability and executable alpha.

Deep-learning sequence models are deliberately deferred until more dates or raw ITCH samples provide enough independent training and validation data.

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
3. A one-step limited-depth price-level reconstruction audit.
4. Event-level order-book and order-flow features.
5. Future mid-price classification and regression targets at 10, 50, and 100 event horizons.
6. Purged chronological training, validation, and exploratory-holdout blocks.
7. Majority and class-balanced logistic classification baselines.
8. Zero-return and Ridge regression baselines.
9. Statistical and spread-aware economic evaluation.
10. Reproducible JSON metrics, model artefacts, tests, and research documentation.

Planned additions include walk-forward validation, dependence-aware inference, spread/liquidity/time-of-day regime analysis, cost stress testing, LightGBM comparison, feature-family ablation, a chart pack, and a short research note.

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

## Reproduce all baseline experiments

Run all three event horizons with the same model and evaluation configuration:

```powershell
.\scripts\run_all_baselines.ps1
```

The script generates:

```text
reports/tables/baseline_h10_metrics.json
reports/tables/baseline_h50_metrics.json
reports/tables/baseline_h100_metrics.json
```

A single horizon can also be run directly:

```powershell
python -m orderbook_research.train_baseline `
    --ticker AAPL `
    --levels 10 `
    --horizon 50
```

## Leakage controls

- Rows are never randomly shuffled before splitting.
- Splits are contiguous in event time.
- A purge gap of at least the maximum target horizon separates adjacent blocks.
- Imputers and scalers are fitted using training observations only.
- Future mid-price values are used only for target construction and evaluation.
- Rolling features use current and past observable events only.
- The final block is described as an exploratory holdout because it has been inspected during model development.
- Stronger claims are deferred until purged walk-forward and multi-day evaluation are available.

## Phase A: next research milestone

The next phase will add:

1. Purged expanding-window validation folds.
2. Fold-level classification and regression metrics.
3. Mean, dispersion, and worst-fold performance.
4. Moving-block bootstrap confidence intervals.
5. Return and residual autocorrelation diagnostics.
6. A consolidated horizon-comparison table generated from code.

This phase is intended to test whether the current findings persist across different intraday periods rather than depending on one favourable split.

## One-day limitation

The public sample provides one date. Chronological intraday evaluation controls leakage but does not establish day-to-day stability. Stronger validation requires additional dates from licensed LOBSTER data or reconstructed official ITCH samples.

## Data and licensing

This repository contains code and lightweight derived metrics only. Review the source dataset's terms before redistributing data. Do not commit the raw CSV files.