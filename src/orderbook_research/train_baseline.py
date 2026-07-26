from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orderbook_research.features import (
    add_snapshot_features,
    default_feature_columns,
)
from orderbook_research.io import load_lobster_pair
from orderbook_research.metrics import (
    classification_metrics,
    regression_metrics,
)
from orderbook_research.simulation import (
    non_overlapping_signal_simulation,
)
from orderbook_research.splits import purged_chronological_split
from orderbook_research.targets import add_event_horizon_targets


def train_baselines(
    ticker: str,
    levels: int,
    horizon: int,
    max_rows: int | None,
    output_directory: Path,
) -> dict[str, object]:
    data = load_lobster_pair(
        ticker=ticker,
        levels=levels,
        nrows=max_rows,
        scale_prices=True,
    )
    data = add_snapshot_features(data, levels=levels)
    data = add_event_horizon_targets(
        data,
        horizons=(10, 50, 100),
    )

    feature_columns = default_feature_columns(levels=levels)
    class_target = f"future_move_{horizon}"
    return_target = f"future_return_bps_{horizon}"

    split = purged_chronological_split(
        len(data),
        purge_events=max(100, horizon),
    )

    train = data.loc[split.train_indices].copy()
    validation = data.loc[split.validation_indices].copy()
    test = data.loc[split.test_indices].copy()

    train = train.dropna(subset=[class_target, return_target])
    validation = validation.dropna(subset=[class_target, return_target])
    test = test.dropna(subset=[class_target, return_target])

    x_train = train[feature_columns]
    x_validation = validation[feature_columns]
    x_test = test[feature_columns]

    y_train_class = train[class_target].astype(int)
    y_validation_class = validation[class_target].astype(int)
    y_test_class = test[class_target].astype(int)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train_class)

    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    max_iter=500,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train_class)

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    ridge.fit(x_train, train[return_target])

    validation_dummy_prediction = dummy.predict(x_validation)
    validation_logistic_prediction = logistic.predict(x_validation)
    validation_ridge_prediction = ridge.predict(x_validation)

    test_dummy_prediction = dummy.predict(x_test)
    test_logistic_prediction = logistic.predict(x_test)
    test_ridge_prediction = ridge.predict(x_test)

    validation_probabilities = logistic.predict_proba(x_validation)
    test_probabilities = logistic.predict_proba(x_test)
    classes = logistic.named_steps["model"].classes_

    probability_columns = {
        int(class_label): position
        for position, class_label in enumerate(classes)
    }

    validation["probability_down"] = validation_probabilities[
        :, probability_columns[-1]
    ]
    validation["probability_up"] = validation_probabilities[
        :, probability_columns[1]
    ]
    test["probability_down"] = test_probabilities[
        :, probability_columns[-1]
    ]
    test["probability_up"] = test_probabilities[
        :, probability_columns[1]
    ]

    validation_simulation, validation_simulation_stats = (
        non_overlapping_signal_simulation(
            validation,
            horizon=horizon,
        )
    )
    test_simulation, test_simulation_stats = (
        non_overlapping_signal_simulation(
            test,
            horizon=horizon,
        )
    )

    metrics = {
        "configuration": {
            "ticker": ticker.upper(),
            "levels": levels,
            "horizon_events": horizon,
            "max_rows": max_rows,
            "feature_count": len(feature_columns),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "purge_events": split.purge_events,
        },
        "validation": {
            "majority_classifier": classification_metrics(
                y_validation_class,
                validation_dummy_prediction,
            ),
            "logistic_regression": classification_metrics(
                y_validation_class,
                validation_logistic_prediction,
            ),
            "ridge_regression": regression_metrics(
                validation[return_target].to_numpy(),
                validation_ridge_prediction,
            ),
            "signal_simulation": validation_simulation_stats,
        },
        "test": {
            "majority_classifier": classification_metrics(
                y_test_class,
                test_dummy_prediction,
            ),
            "logistic_regression": classification_metrics(
                y_test_class,
                test_logistic_prediction,
            ),
            "ridge_regression": regression_metrics(
                test[return_target].to_numpy(),
                test_ridge_prediction,
            ),
            "signal_simulation": test_simulation_stats,
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / f"baseline_h{horizon}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    model_directory = Path("models")
    model_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        logistic,
        model_directory / f"logistic_h{horizon}.joblib",
    )
    joblib.dump(
        ridge,
        model_directory / f"ridge_h{horizon}.joblib",
    )

    validation_simulation.to_parquet(
        output_directory / f"validation_h{horizon}_simulation.parquet",
        index=False,
    )
    test_simulation.to_parquet(
        output_directory / f"test_h{horizon}_simulation.parquet",
        index=False,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe majority, logistic, and Ridge baselines."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[10, 50, 100],
        default=50,
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/tables"),
    )
    args = parser.parse_args()

    metrics = train_baselines(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        max_rows=args.max_rows,
        output_directory=args.output_directory,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
