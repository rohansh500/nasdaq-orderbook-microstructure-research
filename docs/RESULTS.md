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

## Regime and execution-cost sensitivity

Phase B evaluated the existing models across displayed-depth and volatility
regimes, confidence thresholds, and transaction-cost assumptions using the
same five purged walk-forward folds. Regime cutoffs for depth and volatility
were learned from each fold's training data and then applied to its later
validation block. The final 20% exploratory holdout remained unused.

At the primary 50-event horizon, low-volatility observations produced the
strongest average statistical results. Rank IC reached 0.269, Ridge MAE
improved by 0.50% relative to the zero-return baseline, non-zero directional
accuracy reached 58.43%, and gross edge averaged 0.297 bps per active signal.
The estimated break-even cost fraction was 12.30%.

Lower displayed depth was also associated with stronger predictability. The
low-depth regime produced rank IC of 0.266, Ridge MAE improvement of 1.39%,
non-zero directional accuracy of 58.71%, and gross edge of 0.261 bps per
active signal. Performance weakened as displayed depth increased.

| 50-event regime | Rank IC | Ridge MAE improvement | Non-zero directional accuracy | Gross edge per active signal | Break-even cost fraction |
|---|---:|---:|---:|---:|---:|
| Low volatility | 0.269 | +0.50% | 58.43% | 0.297 bps | 12.30% |
| Medium volatility | 0.195 | -0.42% | 56.27% | 0.155 bps | 6.28% |
| High volatility | 0.104 | -0.99% | 55.13% | 0.132 bps | 5.27% |
| Low displayed depth | 0.266 | +1.39% | 58.71% | 0.261 bps | 10.04% |
| Medium displayed depth | 0.193 | -0.85% | 57.02% | 0.209 bps | 8.75% |
| High displayed depth | 0.108 | -1.74% | 53.05% | 0.098 bps | 4.28% |

Confidence filtering increased average gross edge but sharply reduced signal
activity. At 50 events, the gross edge rose from 0.213 bps at a 0.10
confidence threshold to approximately 0.35 bps at thresholds between 0.30
and 0.50. Over the same range, the active-signal fraction fell from 47.88%
to 1.85%.

| 50-event confidence threshold | Active-signal fraction | Gross edge per active signal | Full-cost net edge per active signal | Break-even cost fraction | Positive-gross folds |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 47.88% | 0.213 bps | -2.354 bps | 8.67% | 5/5 |
| 0.20 | 19.75% | 0.313 bps | -2.125 bps | 13.39% | 5/5 |
| 0.30 | 8.78% | 0.346 bps | -1.967 bps | 15.55% | 5/5 |
| 0.40 | 4.03% | 0.353 bps | -1.785 bps | 17.00% | 5/5 |
| 0.50 | 1.85% | 0.350 bps | -1.674 bps | 17.43% | 5/5 |

Across the full validation sample, no 50-event confidence threshold produced
positive average net edge when even 25% of the full quoted-spread cost was
applied. The strongest threshold therefore still required execution costs
below approximately 17% of the aggressive crossing-cost estimate.

Exact tick-spread buckets were highly imbalanced. Approximately 99.0% of
validation observations had spreads of three or more ticks, while one-tick
and two-tick observations represented only about 0.34% and 0.65%. After
non-overlapping horizon sampling, the 50-event one-tick and two-tick buckets
contained averages of only 3.8 and 4.4 observations per fold. Some
spread-confidence intersections produced isolated positive net folds or
slightly positive means, but these outcomes are too sparse to support a
profitability claim and are retained only as exploratory diagnostics.

Overall, the signal is strongest during quieter, lower-depth conditions and
among higher-confidence forecasts. These filters improve gross signal
quality, but the robust configurations remain economically insufficient for
repeated aggressive spread-crossing execution. The 50-event horizon remains
the primary horizon for subsequent nonlinear-model and feature-ablation
experiments.

## Non-linear models and feature-family ablation

Phase C compared the regularised linear baselines with fixed-parameter
LightGBM models across the same five purged walk-forward folds. No
hyperparameter search or validation-based early stopping was performed.

| Horizon | Logistic balanced accuracy | LightGBM balanced accuracy | Ridge rank IC | LightGBM rank IC | LightGBM MAE improvement versus zero |
|---:|---:|---:|---:|---:|---:|
| 10 events | 36.56% | 43.78% | 0.198 | 0.229 | +0.53% |
| 50 events | 37.90% | 41.26% | 0.211 | 0.266 | +0.51% |
| 100 events | 34.97% | 37.01% | 0.164 | 0.211 | -2.52% |

LightGBM improved rank IC over Ridge in every fold at all three horizons.
Classification balanced accuracy and macro F1 also improved materially,
particularly at the 10- and 50-event horizons. However, higher statistical
performance did not translate into stronger execution economics at the
50- and 100-event horizons. All full-cost net results remained negative.

Feature-family ablation at the primary 50-event horizon showed that event
and order-flow features were essential:

| Feature set | Balanced accuracy | MAE improvement | Rank IC | Gross edge per active signal |
|---|---:|---:|---:|---:|
| All features | 41.26% | +0.51% | 0.266 | 0.182 bps |
| Without event flow | 35.68% | -6.44% | 0.048 | 0.010 bps |
| Without time features | 43.89% | +3.51% | 0.275 | 0.271 bps |
| Without book state | 41.23% | +0.33% | 0.257 | 0.190 bps |
| Without volatility | 40.88% | -0.21% | 0.254 | 0.170 bps |

Removing event-flow features almost eliminated the ranking and economic
signal, supporting the central role of rolling trade pressure, order-flow
imbalance and event activity in short-horizon price formation.

Removing explicit time-of-day features improved every major average metric
and produced positive MAE improvement in all five folds. Although these
features received non-trivial LightGBM gain importance, the ablation
results indicate that they encouraged unstable intraday shortcuts rather
than robust generalisation.

The development-selected candidate for final evaluation is therefore the
50-event LightGBM model using all features except explicit time-of-day
variables. This configuration has not yet been evaluated on the reserved
final 20% exploratory holdout.

## Frozen-candidate holdout evaluation

After completing model comparison and feature-family ablation, the final
candidate was frozen before configuration-level holdout evaluation:

- Instrument: AAPL
- Horizon: 50 events
- Model: fixed-parameter LightGBM classifier and regressor
- Features: 31 features, excluding explicit clock-time variables
- Development fraction: first 80%
- Holdout fraction: final 20%
- Development-to-holdout purge: 100 events
- Classification confidence threshold: 0.10
- Execution cost: full estimated quoted-spread crossing cost

The final block had not been used to select the LightGBM no-time candidate,
although it had previously been inspected using exploratory linear
baselines. It is therefore a configuration-level holdout rather than a
completely untouched independent test.

### Final predictive performance

| Model | Balanced accuracy | Macro F1 | MAE | MAE improvement vs zero | Rank IC | Non-zero directional accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Balanced logistic / Ridge | 43.20% | 0.392 | 0.541 bps | +2.35% | 0.288 | 63.03% |
| LightGBM | 46.19% | 0.456 | 0.518 bps | +6.67% | 0.363 | 65.29% |

The frozen LightGBM candidate improved classification, point-return error,
ranking correlation and directional accuracy relative to the regularised
linear models. Its strongest gain was in return ranking, where holdout rank
IC reached 0.363.

The final block covered approximately 15:16 to 16:00. The result should
therefore be interpreted as closing-period evidence within one trading day,
not as proof of out-of-day or cross-asset generalisation.

### Final execution result

| Model | Active fraction | Gross edge per active signal | Estimated cost per active signal | Net edge per active signal | Break-even cost fraction |
|---|---:|---:|---:|---:|---:|
| Balanced logistic | 54.22% | 0.303 bps | 1.776 bps | -1.472 bps | 17.08% |
| LightGBM | 69.83% | 0.289 bps | 1.854 bps | -1.564 bps | 15.62% |

Although LightGBM provided stronger statistical forecasts, it did not
improve per-signal economics at the frozen threshold. Its probabilities
generated more active signals, while the balanced logistic classifier
retained a slightly higher gross edge and break-even cost fraction per
signal.

Both classifiers remained materially negative after full quoted-spread
costs. The frozen LightGBM candidate generated approximately 0.289 bps of
gross edge per active signal against an estimated cost of 1.854 bps.
The result therefore supports short-horizon price predictability, not an
immediately executable aggressive trading strategy.

The most influential LightGBM features were spread, rolling trade pressure,
order-flow imbalance, event intensity, queue imbalance and short-horizon
volatility. These findings are consistent with the Phase C ablation result
that event-flow features are central to the signal.

## Raw Nasdaq ITCH reconstruction

The repository also contains an independent streaming reconstruction path for
the official January 30, 2019 Nasdaq TotalView-ITCH 5.0 sample.

| Metric | Result |
|---|---:|
| Market-wide ITCH records processed | 368,366,634 |
| Binary payload processed | 10.51 GB |
| AAPL displayed-book transitions | 1,656,597 |
| Maximum simultaneous AAPL orders | 42,774 |
| Missing order references | 0 |
| Duplicate order references | 0 |
| Share underflows | 0 |
| Crossed or locked snapshots | 0 |
| Processing throughput | 40,470 records/second |
| Python-tracked peak allocation | 303.6 MB |

The parser consumed the complete compressed file to clean gzip EOF without
truncated records or payload-length mismatches. The sample did not expose a
zero-length BinaryFILE terminator.

This reconstruction is an independent market-data engineering benchmark. It
was not used to claim that the 2012 LOBSTER prediction model generalises to
2019.