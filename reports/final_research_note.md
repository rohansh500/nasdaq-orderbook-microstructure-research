# Final frozen-candidate evaluation

## Research question

Can event-level order-book state and order-flow features predict AAPL mid-price
movement over the next 50 events, and is the
result large enough to survive an aggressive quoted-spread cost assumption?

## Frozen protocol

- Instrument: AAPL
- Book depth: 10 levels
- Horizon: 50 events
- Development fraction: 80%
- Development-to-holdout purge: 100 events
- Selected classifier/regressor: fixed-parameter LightGBM
- Selected feature set: all default features except explicit clock-time features
- Selected feature count: 31
- Confidence threshold: 0.10
- Cost assumption: full estimated quoted-spread crossing cost

The candidate and evaluation rules were frozen before this run. The final block
was not used in Phases A-C to select the LightGBM no-time candidate. However,
the same block had previously been inspected with exploratory linear baselines,
so this is a configuration-level holdout rather than a completely untouched
data set. Truly independent evidence requires additional trading days.

## Final classification result

The frozen LightGBM classifier achieved:

- Accuracy: 55.22%
- Balanced accuracy: 46.19%
- Macro F1: 0.456

## Final return-prediction result

The frozen LightGBM regressor achieved:

- MAE: 0.518 bps
- Zero-return MAE: 0.555 bps
- MAE improvement versus zero: +6.67%
- Rank IC: 0.363
- Non-zero directional accuracy: 65.29%

## Final execution result

At the frozen 0.10 confidence threshold:

- Active-signal fraction: 69.83%
- Gross edge per active signal: 0.289 bps
- Estimated cost per active signal: 1.854 bps
- Net edge per active signal: -1.564 bps
- Break-even cost fraction: 15.62%
- Maximum drawdown: -1747.2 bps

## Interpretation

The final result should be interpreted as evidence about short-horizon price
formation, not as a deployable strategy. A positive rank IC or directional
accuracy shows that the features contain information. Economic viability
requires the gross edge to cover execution costs, latency, queue uncertainty,
fees and model decay.

## Limitations

1. One stock and one trading day cannot establish out-of-day generalisation.
2. LOBSTER provides reconstructed displayed-book snapshots rather than an
   independently parsed raw ITCH feed.
3. The execution model assumes immediate aggressive fills and does not model
   queue position, partial fills, latency or adverse selection.
4. The final block was previously viewed for exploratory linear baselines,
   although not for the frozen LightGBM no-time configuration.
5. Feature importance is descriptive and is not a causal attribution.

## Next research step

Phase E should add raw NASDAQ ITCH parsing and, when accessible, multiple days
or instruments. That extension is required for a truly untouched temporal test
and for independent order-book reconstruction benchmarks.
