# Implementation history

The repository was developed through a sequence of independently testable
milestones.

## 1. Data integrity and transition audit

- Added explicit LOBSTER schemas and fixed-point price conversion.
- Verified message and snapshot row alignment.
- Audited event frequencies, timestamps, dummy levels, and crossed/locked books.
- Implemented a one-step limited-depth reconstruction audit.

## 2. Features, targets, and linear baselines

- Built 35 order-book, event-flow, volatility, and time features.
- Added 10-, 50-, and 100-event classification and regression targets.
- Established majority, balanced-logistic, zero-return, and Ridge baselines.
- Added spread-aware non-overlapping signal simulation.

## 3. Robustness analysis

- Added five purged expanding walk-forward folds.
- Added moving-block bootstrap confidence intervals.
- Added residual autocorrelation, Ljung-Box, calibration, and time-bucket
  diagnostics.
- Added depth, volatility, confidence-threshold, and cost-regime analysis.

## 4. Non-linear modelling and ablation

- Compared fixed-parameter LightGBM with linear baselines.
- Added feature-family ablation.
- Selected the 50-event no-time candidate using development folds only.

## 5. Frozen holdout evaluation

- Committed the evaluation protocol before running the selected candidate.
- Recorded the Git commit, configuration hash, Python version, and split rules.
- Generated final predictive, economic, importance, and drawdown outputs.

## 6. Independent Nasdaq ITCH reconstruction

- Added streaming BinaryFILE parsing for the displayed-order lifecycle.
- Maintained individual order references and aggregated price levels.
- Added stable batched Parquet output and regular-session samples.
- Reconstructed the complete AAPL session from the official January 30, 2019
  sample with all tracked integrity checks passing.

## 7. Public release

- Consolidated results and limitations.
- Added reproducibility, licensing, citation, CI, figures, and release metadata.
- Removed raw, event-level, smoke, model-binary, and large simulation outputs
  from the public repository.
