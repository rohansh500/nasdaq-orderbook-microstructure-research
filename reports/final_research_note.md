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
was not used in Phases A-C to select the LightGBM no-time candidate. The same
block had previously been inspected with exploratory linear baselines, so this
is a configuration-level holdout rather than a completely untouched data set.

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

The result supports short-horizon price predictability within the studied day,
not a deployable strategy. Economic viability would require the gross edge to
cover execution costs, latency, queue uncertainty, fees, impact, and model decay.

## Raw ITCH engineering extension

A separate streaming parser processed 368,366,634 market-wide Nasdaq ITCH
messages and reconstructed 1,656,597 AAPL displayed-book transitions. All
tracked order-reference, quantity, timestamp, aggregation, and crossed-book
integrity checks passed.

This 2019 reconstruction was not used as an out-of-day evaluation of the 2012
predictive model.

## Limitations

1. One stock and one predictive trading day cannot establish out-of-day
   generalisation.
2. The final block is a configuration-level holdout, not an independent day.
3. The execution model assumes immediate aggressive fills and does not model
   queue position, partial fills, latency, fees, impact, or adverse selection.
4. Feature importance is descriptive and is not a causal attribution.
5. Independent ITCH reconstruction demonstrates engineering correctness, not
   cross-day predictive stability.

## Potential extensions

Additional dates and instruments, probability calibration, passive execution,
queue position, latency, partial fills, and market impact remain valid future
research directions. They are not required for the v1.0.0 engineering and
research release.
