# Research design

## Primary hypothesis

Recent order-flow imbalance predicts the sign and magnitude of short-horizon mid-price movement.

## Secondary hypotheses

1. Queue and depth imbalance are more informative when the spread is one tick.
2. Cancellation pressure and aggressive trade pressure add information beyond static depth.
3. Predictability decays as the event horizon increases.
4. Apparent directional accuracy does not necessarily survive spread costs.
5. Performance varies materially across time-of-day and liquidity regimes.

## Prediction tasks

### Classification

For horizons of 10, 50, and 100 future events:

- `-1`: future mid-price lower than current mid-price
- `0`: unchanged
- `1`: future mid-price higher than current mid-price

### Regression

Future mid-price return in basis points at the same event horizons.

## Leakage controls

- Contiguous event-time splits only
- No random train/test split
- Purge gap of at least the largest label horizon
- Training-only imputation and scaling
- Targets constructed only with future mid prices
- Rolling features contain no future rows
- Test block remains untouched until model and thresholds are frozen

## Economic simulation

The initial simulation:

- samples non-overlapping observations at the target horizon;
- uses `P(up) - P(down)` as the signal score;
- applies a confidence threshold;
- measures future mid-price return;
- subtracts half the current spread and half the future spread;
- reports gross return, net return, active-signal rate, and drawdown.

This is deliberately conservative but still incomplete. It does not model latency, queue position, partial fills, market impact, order types, fees, or capacity.

## Validation limitation

The public sample contains a single trading day. Chronological intraday testing is valid for leakage control but insufficient for claims about out-of-day robustness. The final report must distinguish:

- intraday holdout performance;
- true multi-day generalisation, which is not tested initially.
