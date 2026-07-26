# Phase D final evaluation protocol

This document freezes the candidate and the evaluation rules before the final
Phase D run.

## Candidate

- Instrument: AAPL
- LOBSTER depth: 10 levels
- Forecast horizon: 50 events
- Classifier: fixed-parameter LightGBM classifier from Phase C
- Regressor: fixed-parameter LightGBM regressor from Phase C
- Feature set: all default features except explicit clock-time features
- Feature count: 31
- Signal confidence threshold: 0.10
- Transaction-cost assumption: full estimated quoted-spread crossing cost

## Data split

- First 80% of chronological rows: development training block
- 100-event purge before the final block
- Final 20%: frozen-candidate holdout
- Imputation and scaling for linear comparators are fitted only on development
- LightGBM models are fitted only on development

The final 20% was not used in Phases A-C to select or evaluate the frozen
LightGBM no-time candidate. It had previously been viewed for exploratory
linear-baseline results, so the final result is a configuration-level holdout,
not a completely untouched data set.

## Permitted final comparisons

The frozen LightGBM candidate is compared only with:

- majority-class classifier;
- class-balanced logistic regression;
- zero-return predictor;
- Ridge regression.

All models use the same 31-feature no-time input set for fairness.

## Prohibited after the final run

The holdout result must not be used to:

- change the horizon;
- restore or remove another feature family;
- tune LightGBM parameters;
- change the confidence threshold;
- select a lower transaction-cost assumption;
- claim multi-day or cross-asset generalisation.

Any model change after the final run requires new independent data.

## Primary final metrics

Classification:

- balanced accuracy;
- macro F1;
- per-class recall.

Regression:

- MAE improvement versus zero;
- rank IC;
- non-zero directional accuracy.

Economics:

- active-signal fraction;
- gross edge per active signal;
- estimated cost per active signal;
- net edge per active signal;
- break-even cost fraction;
- maximum drawdown.

## Reproducibility control

The final runner writes a manifest containing:

- Git commit hash;
- UTC completion time;
- exact configuration;
- SHA-256 configuration hash;
- Python version.

The runner refuses a second final execution unless `--allow-rerun` is supplied.
