# Research design

## Primary hypothesis

Recent order-flow imbalance and event activity contain information about the
sign and magnitude of short-horizon mid-price movement.

## Secondary hypotheses

1. Dynamic event-flow features add information beyond static book state.
2. Predictability varies across spread, depth, volatility, and intraday regimes.
3. Non-linear models improve ranking and classification relative to regularised
   linear baselines.
4. Statistical predictability does not necessarily survive execution costs.
5. A raw ITCH parser can independently reproduce displayed-book state while
   satisfying order and price-level invariants.

## Prediction tasks

For horizons of 10, 50, and 100 future events:

- classification labels are down, unchanged, and up;
- regression targets are future mid-price returns in basis points.

## Feature groups

The 35-feature baseline schema contains:

- spread, microprice, queue imbalance, and multi-level depth imbalance;
- best-quote order-flow imbalance;
- submission, cancellation, deletion, and execution indicators;
- rolling add, cancel, trade, OFI, event-intensity, and volatility features;
- explicit time-of-day variables, later removed from the frozen candidate after
  development-only ablation.

## Leakage controls

- Contiguous event-time splits only.
- No random train/test split.
- A 100-event purge between adjacent blocks.
- Training-only imputation and scaling.
- Rolling features use only current and historical observations.
- Five expanding walk-forward folds are restricted to the first 80% of the day.
- The candidate model, horizon, feature family, confidence threshold, and cost
  assumption were frozen before configuration-level holdout evaluation.

The final 20% had previously been viewed with exploratory linear baselines. It
was not used to select the final LightGBM no-time candidate, so the result is a
configuration-level holdout rather than a completely untouched test.

## Model comparison

The project compares:

- majority classification;
- class-balanced logistic regression;
- zero-return regression;
- Ridge regression;
- fixed-parameter LightGBM classification and regression.

No validation-driven LightGBM hyperparameter search or early stopping was used.
Feature-family ablation was performed only on development folds.

## Dependence-aware inference

Moving-block bootstrap intervals use 1,000-event blocks and 1,000 draws.
Residual diagnostics include autocorrelation, Ljung-Box tests, calibration,
skewness, kurtosis, and intraday time buckets.

## Economic simulation

The simulation:

- samples non-overlapping observations at the target horizon;
- uses classifier probability imbalance as the signal score;
- applies a fixed confidence threshold;
- measures future mid-price return;
- subtracts estimated entry and exit spread costs;
- reports activity, gross edge, net edge, break-even cost fraction, and drawdown.

It does not model latency, queue position, partial fills, fees, market impact,
order types, adverse selection, or capacity.

## Raw ITCH reconstruction boundary

The independent ITCH path maintains order-ID-level state and full aggregated
visible depth for one symbol. It exports top-N snapshots and checks:

- duplicate and missing order references;
- quantity underflows;
- timestamp monotonicity;
- order-to-level conservation;
- sorted price indices;
- crossed and locked exported snapshots;
- final state integrity.

This engineering result is separate from the 2012 predictive experiment and is
not treated as cross-day model validation.
