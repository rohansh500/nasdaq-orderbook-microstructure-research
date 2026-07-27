# Project execution plan

End each phase with a focused Git commit.

## Phase 0 - Reproducible repository
- Create and activate the virtual environment
- Download AAPL 10-level sample files
- Run audit, tests, and synthetic smoke test
- Commit: `chore: initialise order-book research repository`

## Phase 1 - Data integrity and microstructure audit
- Verify message and order-book row alignment
-  Inspect event-type frequencies
-  Inspect timestamp monotonicity and duplicate timestamps
-  Detect crossed or locked books
-  Clean dummy empty-level values
-  Validate visible-event snapshot transitions
- Commit: `research: audit lobster event and snapshot data`

## Phase 2 - Price-level reconstruction audit
-  Seed the local price-level book from a snapshot
-  Apply submissions, cancellations, deletions, and visible executions
-  Compare reconstructed and supplied top-of-book states
-  Document why limited-depth input cannot guarantee full reconstruction
- Commit: `feat: add price-level reconstruction audit`

## Phase 3 - Leakage-safe features and targets
-  Spread, mid, microprice, queue imbalance
-  Multi-level depth imbalance
-  Best-quote order-flow imbalance
-  Add/cancel/trade pressure
-  Event intensity and inter-arrival time
-  Rolling volatility and time-of-day features
-  Targets at 10, 50, and 100 events
- Commit: `feat: build event-level microstructure features`

## Phase 4 - Baselines
-  Majority classifier
-  Logistic regression
-  Zero-return and Ridge regression
-  Freeze validation metrics and thresholds
- Commit: `model: establish linear event-horizon baselines`

## Phase 5 - Non-linear models
-  LightGBM classifier and regressor
-  Feature-family ablation
-  Seed and hyperparameter stability
- Commit: `model: add gradient-boosted order-book models`

## Phase 6 - Economic evaluation
-  Non-overlapping signal observations
-  Confidence threshold
-  Gross future mid-price return
-  Half-spread entry and exit cost
-  Net PnL and drawdown
-  Performance by spread and liquidity regime
- Commit: `research: add spread-aware signal simulation`

## Phase 7 - Failure analysis
-  Class imbalance and flat-state dominance
-  Time-of-day dependence
-  Spread and depth dependence
-  Latency and fill-probability limitations
-  One-day generalisation limitation
- Commit: `research: document robustness and failure modes`

## Phase 8 - Raw NASDAQ ITCH extension
-  Download an official sample binary
-  Parse length-prefixed ITCH 5.0 messages
-  Build order-ID state and full price levels
-  Validate against independent aggregates
- Commit: `feat: reconstruct book from raw itch messages`

## Phase 9 - Publication
-  Final README with verified numbers only
-  Three-to-five-page research note
-  Chart pack and tables
-  GitHub repository creation and push
- Commit: `docs: publish reproducible microstructure study`
