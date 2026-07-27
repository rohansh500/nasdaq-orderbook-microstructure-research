from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orderbook_research.features import (
    add_snapshot_features,
    default_feature_columns,
)
from orderbook_research.metrics import classification_metrics
from orderbook_research.splits import purged_chronological_split
from orderbook_research.targets import add_event_horizon_targets


def synthetic_lobster_data(
    rows: int = 5_000,
    levels: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], size=rows))
    spread = rng.choice([0.01, 0.02, 0.03], size=rows)
    bid_1 = mid - spread / 2.0
    ask_1 = mid + spread / 2.0

    data: dict[str, np.ndarray] = {
        "event_index": np.arange(rows),
        "time_seconds": 34_200.0 + np.cumsum(rng.exponential(0.02, rows)),
        "event_type": rng.choice([1, 2, 3, 4, 5], size=rows),
        "order_id": np.arange(rows) + 1,
        "size": rng.integers(1, 1_000, size=rows),
        "price": mid,
        "direction": rng.choice([-1, 1], size=rows),
    }

    for level in range(1, levels + 1):
        data[f"ask_price_{level}"] = ask_1 + (level - 1) * 0.01
        data[f"ask_size_{level}"] = rng.integers(100, 5_000, size=rows)
        data[f"bid_price_{level}"] = bid_1 - (level - 1) * 0.01
        data[f"bid_size_{level}"] = rng.integers(100, 5_000, size=rows)

    return pd.DataFrame(data)


def main() -> None:
    levels = 10
    horizon = 50
    data = synthetic_lobster_data(levels=levels)
    data = add_snapshot_features(data, levels=levels)
    data = add_event_horizon_targets(data)

    features = default_feature_columns(levels=levels)
    split = purged_chronological_split(
        len(data),
        purge_events=100,
    )

    train = data.loc[split.train_indices].dropna(subset=[f"future_move_{horizon}"])
    validation = data.loc[split.validation_indices].dropna(subset=[f"future_move_{horizon}"])

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=300,
                ),
            ),
        ]
    )
    model.fit(
        train[features],
        train[f"future_move_{horizon}"].astype(int),
    )
    prediction = model.predict(validation[features])
    metrics = classification_metrics(
        validation[f"future_move_{horizon}"].astype(int),
        prediction,
    )

    print("Synthetic end-to-end smoke test")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
