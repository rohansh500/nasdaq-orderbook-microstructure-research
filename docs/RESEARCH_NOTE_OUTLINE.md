# Research note outline

## 1. Executive summary
- Research question
- Data and observation unit
- Frozen test result
- Cost-aware conclusion
- Main limitation

## 2. Data and reconstruction
- LOBSTER relationship to NASDAQ ITCH
- Message and snapshot schema
- Alignment checks
- Limited-depth reconstruction audit
- Data-quality findings

## 3. Features and targets
- Mid, spread, microprice
- Queue and depth imbalance
- Best-quote OFI
- Add, cancel, and trade pressure
- Event intensity and volatility
- Event-horizon targets

## 4. Leakage-safe methodology
- Purged chronological blocks
- Training-only transformations
- Baselines and model sequence
- Statistical metrics

## 5. Results
- Majority versus logistic classification
- Ridge and LightGBM regression
- Confusion matrices
- Rank IC by horizon
- Feature importance or coefficients
- Spread and liquidity buckets

## 6. Economic evaluation
- Signal construction
- Spread-cost assumptions
- Gross and net return
- Drawdown
- Why apparent alpha can disappear

## 7. Failure modes and limitations
- Flat-class dominance
- One-day sample
- Nonstationarity
- Fill and queue uncertainty
- Market impact and capacity
- Limited-depth versus raw ITCH reconstruction

## 8. Next improvements
- More dates and symbols
- Calibrated probabilities
- Multi-horizon model
- Raw ITCH parser and full order-ID reconstruction
