# Changelog

All notable public-release changes are documented here.

## Unreleased

### Added

- Research-oriented Next.js research website with six interactive views.
- Deterministic synthetic order-book scenario and 31-feature browser engine.
- Offline compact LightGBM tree export with development-row parity validation.
- Browser-side LightGBM tree inference without a Python API.
- Vercel monorepo deployment, website CI, and public-safe aggregate JSON exports.

## 1.0.0 - 2026-07-27

### Added

- Explicit LOBSTER schemas, data audit, and transition validation.
- Thirty-five order-book and event-flow features.
- 10-, 50-, and 100-event classification and regression targets.
- Purged walk-forward validation and moving-block bootstrap inference.
- Econometric, residual, regime, threshold, and cost diagnostics.
- Balanced logistic, Ridge, and fixed-parameter LightGBM models.
- Feature-family ablation and frozen-candidate holdout protocol.
- Final predictive, economic, importance, and drawdown figures.
- Streaming Nasdaq ITCH 5.0 BinaryFILE parser.
- Order-ID-level state, full-depth price aggregation, and top-N snapshots.
- Stable batched Parquet writing and reconstruction integrity checks.
- CI, reproducibility, data-licensing, citation, and release documentation.

### Results

- Frozen 50-event LightGBM rank IC: 0.363.
- Frozen LightGBM balanced accuracy: 46.19%.
- Frozen LightGBM MAE improvement versus zero: 6.67%.
- Gross edge versus estimated full crossing cost: 0.289 versus 1.854 bps.
- Full ITCH records processed: 368,366,634.
- AAPL book transitions reconstructed: 1,656,597.
- Tracked reconstruction integrity failures: zero.

### Limitations

- Predictive evidence is based on one AAPL trading day.
- The final block is a configuration-level holdout.
- The execution simulation is not a production fill model.
- The 2019 reconstruction is not a cross-day test of the 2012 model.
