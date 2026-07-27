from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orderbook_research.features import (
    add_snapshot_features,
    default_feature_columns,
)
from orderbook_research.io import load_lobster_pair
from orderbook_research.metrics import classification_metrics
from orderbook_research.simulation import non_overlapping_signal_simulation
from orderbook_research.splits import purged_chronological_split
from orderbook_research.targets import add_event_horizon_targets

CLASS_LABELS = (-1, 0, 1)
CLASS_NAMES = {
    -1: "down",
    0: "flat",
    1: "up",
}


def class_distribution(target: pd.Series) -> dict[str, object]:
    """Return counts and proportions for down, flat, and up targets."""
    counts = target.value_counts().reindex(CLASS_LABELS, fill_value=0)
    total = int(counts.sum())

    proportions = counts / total if total > 0 else pd.Series(0.0, index=counts.index)

    return {
        "labels": list(CLASS_LABELS),
        "counts": {CLASS_NAMES[label]: int(counts.loc[label]) for label in CLASS_LABELS},
        "proportions": {
            CLASS_NAMES[label]: float(proportions.loc[label]) for label in CLASS_LABELS
        },
    }


def detailed_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Extend the repository metrics with per-class diagnostics."""
    result = classification_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_LABELS),
        zero_division=0,
    )

    result["per_class"] = {
        CLASS_NAMES[label]: {
            "label": label,
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "support": int(support[position]),
        }
        for position, label in enumerate(CLASS_LABELS)
    }
    return result


def detailed_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate return forecasts, including non-zero direction accuracy."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No finite target/prediction pairs.")

    if np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        rank_ic = 0.0
    else:
        rank_value = spearmanr(actual, predicted).statistic
        rank_ic = float(rank_value) if np.isfinite(rank_value) else 0.0

    directional_accuracy = float(np.mean(np.sign(actual) == np.sign(predicted)))

    nonzero_mask = actual != 0.0
    nonzero_directional_accuracy = (
        float(np.mean(np.sign(actual[nonzero_mask]) == np.sign(predicted[nonzero_mask])))
        if nonzero_mask.any()
        else 0.0
    )

    return {
        "observations": int(len(actual)),
        "nonzero_observations": int(nonzero_mask.sum()),
        "mae_bps": float(mean_absolute_error(actual, predicted)),
        "rmse_bps": float(mean_squared_error(actual, predicted) ** 0.5),
        "rank_ic": rank_ic,
        "directional_accuracy": directional_accuracy,
        "nonzero_directional_accuracy": nonzero_directional_accuracy,
        "mean_actual_return_bps": float(actual.mean()),
        "mean_predicted_return_bps": float(predicted.mean()),
    }


def detailed_simulation_metrics(
    simulation: pd.DataFrame,
    base_stats: dict[str, float],
) -> dict[str, float | int]:
    """Add per-signal economics and break-even spread information."""
    result: dict[str, float | int] = dict(base_stats)
    active = simulation["signal"] != 0
    active_count = int(active.sum())

    gross_total = float(simulation["gross_return_bps"].sum(skipna=True))
    cost_total = float(simulation["estimated_cost_bps"].sum(skipna=True))
    net_total = float(simulation["net_return_bps"].sum(skipna=True))

    result.update(
        {
            "observations": int(len(simulation)),
            "active_signals": active_count,
            "active_signal_fraction": float(active.mean()),
            "gross_return_bps": gross_total,
            "estimated_total_cost_bps": cost_total,
            "net_return_bps": net_total,
            "mean_gross_return_active_bps": (
                float(simulation.loc[active, "gross_return_bps"].mean())
                if active_count > 0
                else 0.0
            ),
            "mean_estimated_cost_active_bps": (
                float(simulation.loc[active, "estimated_cost_bps"].mean())
                if active_count > 0
                else 0.0
            ),
            "mean_net_return_active_bps": (
                float(simulation.loc[active, "net_return_bps"].mean()) if active_count > 0 else 0.0
            ),
            "break_even_cost_fraction": (
                float(gross_total / cost_total) if cost_total > 0 else 0.0
            ),
        }
    )
    return result


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

    train = data.iloc[split.train_indices].copy()
    validation = data.iloc[split.validation_indices].copy()
    exploratory_holdout = data.iloc[split.test_indices].copy()

    train = train.dropna(subset=[class_target, return_target])
    validation = validation.dropna(subset=[class_target, return_target])
    exploratory_holdout = exploratory_holdout.dropna(subset=[class_target, return_target])

    x_train = train[feature_columns]
    x_validation = validation[feature_columns]
    x_holdout = exploratory_holdout[feature_columns]

    y_train_class = train[class_target].astype(int)
    y_validation_class = validation[class_target].astype(int)
    y_holdout_class = exploratory_holdout[class_target].astype(int)

    majority_classifier = DummyClassifier(strategy="most_frequent")
    majority_classifier.fit(x_train, y_train_class)

    balanced_logistic = Pipeline(
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
                    random_state=42,
                ),
            ),
        ]
    )
    balanced_logistic.fit(x_train, y_train_class)

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    ridge.fit(x_train, train[return_target])

    validation_majority_prediction = majority_classifier.predict(x_validation)
    validation_logistic_prediction = balanced_logistic.predict(x_validation)
    validation_ridge_prediction = ridge.predict(x_validation)
    validation_zero_prediction = np.zeros(len(validation), dtype=float)

    holdout_majority_prediction = majority_classifier.predict(x_holdout)
    holdout_logistic_prediction = balanced_logistic.predict(x_holdout)
    holdout_ridge_prediction = ridge.predict(x_holdout)
    holdout_zero_prediction = np.zeros(
        len(exploratory_holdout),
        dtype=float,
    )

    validation_probabilities = balanced_logistic.predict_proba(x_validation)
    holdout_probabilities = balanced_logistic.predict_proba(x_holdout)
    fitted_classes = balanced_logistic.named_steps["model"].classes_

    probability_columns = {
        int(class_label): position for position, class_label in enumerate(fitted_classes)
    }
    required_probability_classes = {-1, 1}
    missing_probability_classes = required_probability_classes - set(probability_columns)
    if missing_probability_classes:
        raise ValueError(
            "The fitted classifier is missing required classes: "
            f"{sorted(missing_probability_classes)}"
        )

    validation["probability_down"] = validation_probabilities[:, probability_columns[-1]]
    validation["probability_up"] = validation_probabilities[:, probability_columns[1]]
    exploratory_holdout["probability_down"] = holdout_probabilities[:, probability_columns[-1]]
    exploratory_holdout["probability_up"] = holdout_probabilities[:, probability_columns[1]]

    validation_simulation, validation_base_simulation_stats = non_overlapping_signal_simulation(
        validation,
        horizon=horizon,
    )
    holdout_simulation, holdout_base_simulation_stats = non_overlapping_signal_simulation(
        exploratory_holdout,
        horizon=horizon,
    )

    validation_simulation_stats = detailed_simulation_metrics(
        validation_simulation,
        validation_base_simulation_stats,
    )
    holdout_simulation_stats = detailed_simulation_metrics(
        holdout_simulation,
        holdout_base_simulation_stats,
    )

    metrics: dict[str, object] = {
        "configuration": {
            "result_schema_version": "0.2.0",
            "ticker": ticker.upper(),
            "levels": levels,
            "horizon_events": horizon,
            "max_rows": max_rows,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "class_target": class_target,
            "return_target": return_target,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "exploratory_holdout_rows": len(exploratory_holdout),
            "purge_events": split.purge_events,
            "split_design": ("purged_chronological_train_validation_exploratory_holdout"),
            "logistic_parameters": {
                "C": 0.10,
                "max_iter": 500,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            },
            "ridge_parameters": {
                "alpha": 10.0,
            },
            "simulation_parameters": {
                "confidence_threshold": 0.10,
                "additional_fee_bps": 0.0,
                "sampling": "non_overlapping_every_horizon_events",
            },
        },
        "class_distributions": {
            "train": class_distribution(y_train_class),
            "validation": class_distribution(y_validation_class),
            "exploratory_holdout": class_distribution(y_holdout_class),
        },
        "validation": {
            "majority_classifier": detailed_classification_metrics(
                y_validation_class,
                validation_majority_prediction,
            ),
            "balanced_logistic_regression": (
                detailed_classification_metrics(
                    y_validation_class,
                    validation_logistic_prediction,
                )
            ),
            "zero_return_baseline": detailed_regression_metrics(
                validation[return_target],
                validation_zero_prediction,
            ),
            "ridge_regression": detailed_regression_metrics(
                validation[return_target],
                validation_ridge_prediction,
            ),
            "signal_simulation": validation_simulation_stats,
        },
        "exploratory_holdout": {
            "majority_classifier": detailed_classification_metrics(
                y_holdout_class,
                holdout_majority_prediction,
            ),
            "balanced_logistic_regression": (
                detailed_classification_metrics(
                    y_holdout_class,
                    holdout_logistic_prediction,
                )
            ),
            "zero_return_baseline": detailed_regression_metrics(
                exploratory_holdout[return_target],
                holdout_zero_prediction,
            ),
            "ridge_regression": detailed_regression_metrics(
                exploratory_holdout[return_target],
                holdout_ridge_prediction,
            ),
            "signal_simulation": holdout_simulation_stats,
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
        balanced_logistic,
        model_directory / f"balanced_logistic_h{horizon}.joblib",
    )
    joblib.dump(
        ridge,
        model_directory / f"ridge_h{horizon}.joblib",
    )

    validation_simulation.to_parquet(
        output_directory / f"validation_h{horizon}_simulation.parquet",
        index=False,
    )
    holdout_simulation.to_parquet(
        output_directory / f"exploratory_holdout_h{horizon}_simulation.parquet",
        index=False,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-safe majority, balanced logistic, zero-return, and Ridge baselines."
        )
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
