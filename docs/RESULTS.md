# Consolidated results

## 1. Purged walk-forward validation

Five expanding-window folds were evaluated inside the first 80% of the 2012
AAPL day, with a 100-event purge between training and validation.

| Horizon | Mean balanced accuracy | Mean rank IC | Positive rank-IC folds | Non-zero directional accuracy | Positive Ridge-MAE folds | Break-even cost fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 10 events | 36.56% | 0.198 | 5/5 | 60.12% | 0/5 | 3.69% |
| 50 events | 37.90% | **0.211** | 5/5 | 57.06% | **4/5** | 8.70% |
| 100 events | 34.97% | 0.164 | 5/5 | 54.03% | 2/5 | **10.76%** |

The 50-event horizon offered the strongest balance of ranking, point-forecast,
and stability metrics. Every horizon produced positive gross edge in all five
folds, but none covered the quoted aggressive execution cost.

## 2. Moving-block bootstrap

Intervals use 1,000-event blocks, 1,000 draws, and 95% percentile bounds.

| Horizon | Rank IC, 95% CI | MAE improvement, 95% CI | Non-zero direction, 95% CI | Gross edge, 95% CI | Net edge, 95% CI |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.198 [0.182, 0.210] | -5.93% [-6.66%, -5.33%] | 60.12% [59.31%, 60.90%] | 0.088 [0.077, 0.096] bps | -2.351 [-2.434, -2.265] bps |
| 50 | 0.211 [0.183, 0.236] | -0.06% [-0.99%, 0.86%] | 57.07% [55.62%, 58.27%] | 0.215 [0.173, 0.251] bps | -2.352 [-2.447, -2.251] bps |
| 100 | 0.164 [0.126, 0.194] | -3.09% [-4.74%, -1.50%] | 54.04% [52.35%, 55.63%] | 0.245 [0.164, 0.314] bps | -2.287 [-2.405, -2.162] bps |

Ranking and conditional direction remained statistically informative, while
full-cost net intervals stayed below zero.

## 3. Econometric diagnostics

| Horizon | Mean rank IC | Calibration slope | Residual lag-1 ACF | Ljung-Box lag-20 rejections | Positive-rank time buckets |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.198 | 0.745 | -0.097 | 5/5 | 9/9 |
| 50 | 0.211 | 0.727 | -0.032 | 2/5 | 9/9 |
| 100 | 0.164 | 0.683 | -0.019 | 2/5 | 9/9 |

One-event returns showed lag-1 autocorrelation of approximately -0.112, while
squared returns showed lag-1 autocorrelation of approximately 0.156. The model
was more useful for ranking than for return-magnitude calibration, and
performance varied materially through the day.

## 4. Regime and cost sensitivity

At 50 events, the strongest development regimes were low volatility and low
displayed depth.

| Regime | Rank IC | Ridge MAE improvement | Non-zero direction | Gross edge | Break-even cost fraction |
|---|---:|---:|---:|---:|---:|
| Low volatility | **0.269** | +0.50% | 58.43% | **0.297 bps** | **12.30%** |
| Medium volatility | 0.195 | -0.42% | 56.27% | 0.155 bps | 6.28% |
| High volatility | 0.104 | -0.99% | 55.13% | 0.132 bps | 5.27% |
| Low depth | **0.266** | **+1.39%** | **58.71%** | 0.261 bps | 10.04% |
| Medium depth | 0.193 | -0.85% | 57.02% | 0.209 bps | 8.75% |
| High depth | 0.108 | -1.74% | 53.05% | 0.098 bps | 4.28% |

Confidence filtering increased gross edge but reduced activity sharply. At the
50-event horizon, gross edge rose from 0.213 bps at a 0.10 threshold to roughly
0.35 bps at 0.40 to 0.50, while active observations fell from 47.88% to 1.85%.
No robust configuration survived even 25% of the full quoted-spread estimate.

Exact one- and two-tick regimes were too sparse for reliable conclusions. About
99% of validation observations had spreads of three or more ticks.

## 5. Linear versus LightGBM

| Horizon | Logistic balanced accuracy | LightGBM balanced accuracy | Ridge rank IC | LightGBM rank IC | LightGBM MAE improvement vs zero |
|---:|---:|---:|---:|---:|---:|
| 10 | 36.56% | **43.78%** | 0.198 | **0.229** | +0.53% |
| 50 | 37.90% | **41.26%** | 0.211 | **0.266** | +0.51% |
| 100 | 34.97% | **37.01%** | 0.164 | **0.211** | -2.52% |

LightGBM improved rank IC over Ridge in every fold at every horizon. Statistical
improvement did not consistently translate into better per-signal economics.

### Feature-family ablation at 50 events

| Feature set | Balanced accuracy | MAE improvement | Rank IC | Gross edge |
|---|---:|---:|---:|---:|
| All features | 41.26% | +0.51% | 0.266 | 0.182 bps |
| Without event flow | 35.68% | -6.44% | 0.048 | 0.010 bps |
| Without time | **43.89%** | **+3.51%** | **0.275** | **0.271 bps** |
| Without book state | 41.23% | +0.33% | 0.257 | 0.190 bps |
| Without volatility | 40.88% | -0.21% | 0.254 | 0.170 bps |

Removing event flow nearly eliminated the signal. Removing explicit time
variables improved development-fold performance and defined the final candidate.

## 6. Frozen configuration-level holdout

| Model | Balanced accuracy | Macro F1 | MAE | MAE improvement vs zero | Rank IC | Non-zero direction |
|---|---:|---:|---:|---:|---:|---:|
| Balanced logistic / Ridge | 43.20% | 0.392 | 0.541 bps | +2.35% | 0.288 | 63.03% |
| LightGBM | **46.19%** | **0.456** | **0.518 bps** | **+6.67%** | **0.363** | **65.29%** |

At the frozen 0.10 threshold, LightGBM produced 0.289 bps gross edge per active
signal against 1.854 bps estimated cost, resulting in -1.564 bps net edge and a
15.62% break-even cost fraction.

The holdout covered approximately 15:16 to 16:00 and should be interpreted as
closing-period evidence within one day.

## 7. Raw Nasdaq ITCH reconstruction

| Metric | Result |
|---|---:|
| Market-wide messages | 368,366,634 |
| Payload bytes | 10,509,149,824 |
| AAPL book events | 1,656,597 |
| Adds | 752,975 |
| Executions | 89,735 |
| Partial cancels | 5,900 |
| Deletes | 685,024 |
| Replaces | 122,963 |
| Maximum open AAPL orders | 42,774 |
| Missing references | 0 |
| Duplicate references | 0 |
| Share underflows | 0 |
| Crossed or locked rows | 0 |
| Final open orders | 0 |
| Records per second | 40,470 |
| Python-tracked peak allocation | 303.6 MB |

The compressed session reached clean gzip EOF without truncation, payload-length
mismatches, or decompression errors. The sample did not expose a zero-length
BinaryFILE terminator.

## Final interpretation

The project establishes two defensible results:

1. Event-flow and order-book features contain short-horizon information about
   AAPL mid-price movement within the studied day.
2. A streaming Python implementation can parse the full official ITCH sample
   and maintain internally consistent order-ID and full-depth price-level state.

It does not establish a profitable aggressive strategy, cross-day predictive
stability, or cross-asset generalisation.
