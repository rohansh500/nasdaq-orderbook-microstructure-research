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

- Accuracy: 50.62%
- Balanced accuracy: 37.39%
- Macro F1: 0.375

## Final return-prediction result

The frozen LightGBM regressor achieved:

- MAE: 0.848 bps
- Zero-return MAE: 0.864 bps
- MAE improvement versus zero: +1.91%
- Rank IC: 0.248
- Non-zero directional accuracy: 58.42%

## Final execution result

At the frozen 0.10 confidence threshold:

- Active-signal fraction: 75.88%
- Gross edge per active signal: 0.239 bps
- Estimated cost per active signal: 3.230 bps
- Net edge per active signal: -2.991 bps
- Break-even cost fraction: 7.40%
- Maximum drawdown: -445.6 bps

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
