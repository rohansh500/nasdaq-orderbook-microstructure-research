# v1.0.0 release notes

## Summary

This release completes the predictive-research and raw-market-data engineering
workflows.

The repository combines a leakage-controlled AAPL microstructure study with a
streaming Nasdaq ITCH 5.0 parser and independent order-book reconstruction
engine.

## Verified highlights

- 368,366,634 market-wide ITCH messages parsed.
- 1,656,597 AAPL displayed-book transitions reconstructed.
- Zero tracked missing references, duplicate references, share underflows,
  timestamp reversals, or crossed/locked snapshots.
- Frozen 50-event LightGBM rank IC of 0.363.
- Frozen LightGBM balanced accuracy of 46.19%.
- Frozen LightGBM MAE improvement of 6.67% versus zero return.
- Gross signal edge of 0.289 bps against estimated aggressive cost of 1.854 bps.

## Interpretation

The release demonstrates short-horizon predictive information and robust
market-data engineering. It does not claim a deployable profitable strategy or
cross-day model generalisation.

## Suggested GitHub description

Streaming Nasdaq ITCH order-book reconstruction and leakage-controlled
microstructure prediction research in Python.

## Suggested topics

```text
market-microstructure
nasdaq-itch
order-book
quantitative-finance
lightgbm
time-series
python
algorithmic-trading
```
