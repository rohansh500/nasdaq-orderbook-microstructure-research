from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from orderbook_research.simulation import non_overlapping_signal_simulation
from orderbook_research.targets import add_event_horizon_targets
from orderbook_research.train_baseline import (
    class_distribution,
    detailed_classification_metrics,
    detailed_regression_metrics,
    detailed_simulation_metrics,
)
from orderbook_research.walk_forward import (
    ExpandingWindowFold,
    expanding_window_folds,
)


SUMMARY_METRICS = [
    "balanced_accuracy",
    "macro_f1",
    "ridge_mae_improvement_bps",
    "ridge_mae_improvement_pct",
    "ridge_rmse_improvement_bps",
    "ridge_rmse_improvement_pct",
    "ridge_rank_ic",
    "ridge_directional_accuracy",
    "ridge_nonzero_directional_accuracy",
    "active_signal_fraction",
    "mean_gross_return_active_bps",
    "mean_estimated_cost_active_bps",
    "mean_net_return_active_bps",
    "break_even_cost_fraction",
    "net_return_bps",
    "max_drawdown_bps",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _percentage_improvement(baseline: float, model: float) -> float:
    if baseline == 0:
        return 0.0
    return float((baseline - model) / baseline * 100.0)


def _build_models() -> tuple[DummyClassifier, Pipeline, Pipeline]:
    majority_classifier = DummyClassifier(strategy="most_frequent")

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

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    return majority_classifier, balanced_logistic, ridge


def _fold_row(
    horizon: int,
    metadata: dict[str, Any],
    majority_metrics: dict[str, Any],
    logistic_metrics: dict[str, Any],
    zero_metrics: dict[str, Any],
    ridge_metrics: dict[str, Any],
    simulation_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "horizon": horizon,
        **metadata,
        "majority_accuracy": majority_metrics["accuracy"],
        "logistic_accuracy": logistic_metrics["accuracy"],
        "balanced_accuracy": logistic_metrics["balanced_accuracy"],
        "macro_f1": logistic_metrics["macro_f1"],
        "down_recall": logistic_metrics["per_class"]["down"]["recall"],
        "flat_recall": logistic_metrics["per_class"]["flat"]["recall"],
        "up_recall": logistic_metrics["per_class"]["up"]["recall"],
        "zero_mae_bps": zero_metrics["mae_bps"],
        "ridge_mae_bps": ridge_metrics["mae_bps"],
        "ridge_mae_improvement_bps": (
            zero_metrics["mae_bps"] - ridge_metrics["mae_bps"]
        ),
        "ridge_mae_improvement_pct": _percentage_improvement(
            zero_metrics["mae_bps"],
            ridge_metrics["mae_bps"],
        ),
        "zero_rmse_bps": zero_metrics["rmse_bps"],
        "ridge_rmse_bps": ridge_metrics["rmse_bps"],
        "ridge_rmse_improvement_bps": (
            zero_metrics["rmse_bps"] - ridge_metrics["rmse_bps"]
        ),
        "ridge_rmse_improvement_pct": _percentage_improvement(
            zero_metrics["rmse_bps"],
            ridge_metrics["rmse_bps"],
        ),
        "ridge_rank_ic": ridge_metrics["rank_ic"],
        "ridge_directional_accuracy": ridge_metrics[
            "directional_accuracy"
        ],
        "ridge_nonzero_directional_accuracy": ridge_metrics[
            "nonzero_directional_accuracy"
        ],
        "active_signals": simulation_metrics["active_signals"],
        "active_signal_fraction": simulation_metrics[
            "active_signal_fraction"
        ],
        "gross_return_bps": simulation_metrics["gross_return_bps"],
        "estimated_total_cost_bps": simulation_metrics[
            "estimated_total_cost_bps"
        ],
        "net_return_bps": simulation_metrics["net_return_bps"],
        "mean_gross_return_active_bps": simulation_metrics[
            "mean_gross_return_active_bps"
        ],
        "mean_estimated_cost_active_bps": simulation_metrics[
            "mean_estimated_cost_active_bps"
        ],
        "mean_net_return_active_bps": simulation_metrics[
            "mean_net_return_active_bps"
        ],
        "break_even_cost_fraction": simulation_metrics[
            "break_even_cost_fraction"
        ],
        "max_drawdown_bps": simulation_metrics["max_drawdown_bps"],
    }


def _metric_summary(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": 0.0,
            "std": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "positive_folds": 0,
            "folds": 0,
        }

    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
        "positive_folds": int((clean > 0).sum()),
        "folds": int(len(clean)),
    }


def summarize_fold_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        metric: _metric_summary(frame[metric])
        for metric in SUMMARY_METRICS
        if metric in frame.columns
    }


def _cross_horizon_summary(output_directory: Path) -> None:
    rows: list[dict[str, Any]] = []
    json_summary: dict[str, Any] = {
        "result_schema_version": "0.3.0",
        "horizons": {},
    }

    for horizon in (10, 50, 100):
        path = output_directory / f"walk_forward_h{horizon}_folds.csv"
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        horizon_summary = summarize_fold_frame(frame)
        json_summary["horizons"][str(horizon)] = horizon_summary

        row: dict[str, Any] = {
            "horizon": horizon,
            "folds": int(len(frame)),
        }
        for metric, stats in horizon_summary.items():
            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)

    if not rows:
        return

    pd.DataFrame(rows).sort_values("horizon").to_csv(
        output_directory / "walk_forward_summary.csv",
        index=False,
    )
    (
        output_directory / "walk_forward_summary_metrics.json"
    ).write_text(
        json.dumps(json_summary, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _evaluate_fold(
    data: pd.DataFrame,
    fold: ExpandingWindowFold,
    feature_columns: list[str],
    class_target: str,
    return_target: str,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = data.iloc[fold.train_indices].copy()
    validation = data.iloc[fold.validation_indices].copy()

    train = train.dropna(subset=[class_target, return_target])
    validation = validation.dropna(subset=[class_target, return_target])

    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.fold} is empty after target filtering.")

    x_train = train[feature_columns]
    x_validation = validation[feature_columns]
    y_train_class = train[class_target].astype(int)
    y_validation_class = validation[class_target].astype(int)

    required_classes = {-1, 0, 1}
    missing_train_classes = required_classes - set(y_train_class.unique())
    if missing_train_classes:
        raise ValueError(
            f"Fold {fold.fold} training data is missing classes: "
            f"{sorted(missing_train_classes)}"
        )

    majority, logistic, ridge = _build_models()
    majority.fit(x_train, y_train_class)
    logistic.fit(x_train, y_train_class)
    ridge.fit(x_train, train[return_target])

    majority_prediction = majority.predict(x_validation)
    logistic_prediction = logistic.predict(x_validation)
    ridge_prediction = ridge.predict(x_validation)
    zero_prediction = np.zeros(len(validation), dtype=float)

    probabilities = logistic.predict_proba(x_validation)
    fitted_classes = logistic.named_steps["model"].classes_
    probability_columns = {
        int(label): position
        for position, label in enumerate(fitted_classes)
    }
    for required_class in (-1, 1):
        if required_class not in probability_columns:
            raise ValueError(
                f"Fold {fold.fold} classifier did not fit class "
                f"{required_class}."
            )

    validation["probability_down"] = probabilities[
        :, probability_columns[-1]
    ]
    validation["probability_up"] = probabilities[
        :, probability_columns[1]
    ]

    majority_metrics = detailed_classification_metrics(
        y_validation_class,
        majority_prediction,
    )
    logistic_metrics = detailed_classification_metrics(
        y_validation_class,
        logistic_prediction,
    )
    zero_metrics = detailed_regression_metrics(
        validation[return_target],
        zero_prediction,
    )
    ridge_metrics = detailed_regression_metrics(
        validation[return_target],
        ridge_prediction,
    )

    simulation, base_simulation_stats = non_overlapping_signal_simulation(
        validation,
        horizon=horizon,
    )
    simulation_metrics = detailed_simulation_metrics(
        simulation,
        base_simulation_stats,
    )

    metadata: dict[str, Any] = fold.metadata()
    metadata.update(
        {
            "train_rows_after_target_filter": int(len(train)),
            "validation_rows_after_target_filter": int(len(validation)),
            "train_time_start_seconds": float(train["time_seconds"].min()),
            "train_time_end_seconds": float(train["time_seconds"].max()),
            "validation_time_start_seconds": float(
                validation["time_seconds"].min()
            ),
            "validation_time_end_seconds": float(
                validation["time_seconds"].max()
            ),
        }
    )

    row = _fold_row(
        horizon=horizon,
        metadata=metadata,
        majority_metrics=majority_metrics,
        logistic_metrics=logistic_metrics,
        zero_metrics=zero_metrics,
        ridge_metrics=ridge_metrics,
        simulation_metrics=simulation_metrics,
    )

    details = {
        "fold_metadata": metadata,
        "class_distributions": {
            "train": class_distribution(y_train_class),
            "validation": class_distribution(y_validation_class),
        },
        "majority_classifier": majority_metrics,
        "balanced_logistic_regression": logistic_metrics,
        "zero_return_baseline": zero_metrics,
        "ridge_regression": ridge_metrics,
        "signal_simulation": simulation_metrics,
    }
    return row, details


def run_walk_forward(
    ticker: str,
    levels: int,
    horizon: int,
    max_rows: int | None,
    output_directory: Path,
    n_folds: int = 5,
    purge_events: int = 100,
) -> dict[str, Any]:
    effective_purge = max(purge_events, horizon)

    data = load_lobster_pair(
        ticker=ticker,
        levels=levels,
        nrows=max_rows,
        scale_prices=True,
    )
    data = add_snapshot_features(data, levels=levels)
    data = add_event_horizon_targets(data, horizons=(10, 50, 100))

    feature_columns = default_feature_columns(levels=levels)
    class_target = f"future_move_{horizon}"
    return_target = f"future_return_bps_{horizon}"

    folds = expanding_window_folds(
        n_rows=len(data),
        n_folds=n_folds,
        development_fraction=0.80,
        initial_train_fraction=0.30,
        validation_fraction=0.10,
        purge_events=effective_purge,
    )

    rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []

    for fold in folds:
        print(
            f"Horizon {horizon}: fold {fold.fold}/{len(folds)} "
            f"({len(fold.train_indices):,} train, "
            f"{len(fold.validation_indices):,} validation rows before "
            "target filtering)"
        )
        row, details = _evaluate_fold(
            data=data,
            fold=fold,
            feature_columns=feature_columns,
            class_target=class_target,
            return_target=return_target,
            horizon=horizon,
        )
        rows.append(row)
        fold_details.append(details)

    fold_frame = pd.DataFrame(rows).sort_values("fold")
    summary = summarize_fold_frame(fold_frame)

    result: dict[str, Any] = {
        "configuration": {
            "result_schema_version": "0.3.0",
            "ticker": ticker.upper(),
            "levels": levels,
            "horizon_events": horizon,
            "max_rows": max_rows,
            "rows_loaded": int(len(data)),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "class_target": class_target,
            "return_target": return_target,
            "folds": n_folds,
            "development_fraction": 0.80,
            "exploratory_holdout_fraction": 0.20,
            "initial_train_fraction": 0.30,
            "validation_fraction": 0.10,
            "requested_purge_events": purge_events,
            "effective_purge_events": effective_purge,
            "split_design": "purged_expanding_window_first_80_percent",
            "logistic_parameters": {
                "C": 0.10,
                "max_iter": 500,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            },
            "ridge_parameters": {"alpha": 10.0},
            "simulation_parameters": {
                "confidence_threshold": 0.10,
                "additional_fee_bps": 0.0,
                "sampling": "non_overlapping_every_horizon_events",
            },
        },
        "summary": summary,
        "folds": fold_details,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(
        output_directory / f"walk_forward_h{horizon}_folds.csv",
        index=False,
    )
    (
        output_directory / f"walk_forward_h{horizon}_metrics.json"
    ).write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )

    _cross_horizon_summary(output_directory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run purged expanding-window validation for the current "
            "order-book baselines."
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/tables"),
    )
    args = parser.parse_args()

    result = run_walk_forward(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        max_rows=args.max_rows,
        output_directory=args.output_directory,
        n_folds=args.folds,
        purge_events=args.purge_events,
    )

    print("\nWalk-forward summary")
    print(json.dumps(result["summary"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
