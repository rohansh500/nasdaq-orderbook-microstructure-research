from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orderbook_research.features import add_snapshot_features, default_feature_columns
from orderbook_research.io import load_lobster_pair
from orderbook_research.model_comparison import (
    ablation_feature_sets,
    build_lightgbm_classifier,
    build_lightgbm_regressor,
    normalized_feature_importance,
    probability_positions,
)
from orderbook_research.simulation import non_overlapping_signal_simulation
from orderbook_research.targets import add_event_horizon_targets
from orderbook_research.train_baseline import (
    detailed_classification_metrics,
    detailed_regression_metrics,
    detailed_simulation_metrics,
)
from orderbook_research.walk_forward import ExpandingWindowFold, expanding_window_folds

SCHEMA_VERSION = "0.7.0"
PRIMARY_HORIZON = 50
DEFAULT_CONFIDENCE_THRESHOLD = 0.10


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _linear_models() -> tuple[Pipeline, Pipeline]:
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
    return logistic, ridge


def _percentage_improvement(baseline: float, model: float) -> float:
    return float((baseline - model) / baseline * 100.0) if baseline else 0.0


def _attach_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    positions = probability_positions(classes)
    result = frame.copy()
    result["probability_down"] = probabilities[:, positions[-1]]
    result["probability_up"] = probabilities[:, positions[1]]
    return result


def _simulation_metrics(
    validation: pd.DataFrame,
    horizon: int,
) -> dict[str, float | int]:
    simulation, base = non_overlapping_signal_simulation(
        validation,
        horizon=horizon,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    return detailed_simulation_metrics(simulation, base)


def _regression_comparison(
    actual: pd.Series,
    ridge_prediction: np.ndarray,
    lightgbm_prediction: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    zero = detailed_regression_metrics(actual, np.zeros(len(actual), dtype=float))
    ridge = detailed_regression_metrics(actual, ridge_prediction)
    lightgbm = detailed_regression_metrics(actual, lightgbm_prediction)
    return zero, ridge, lightgbm


def _comparison_fold(
    data: pd.DataFrame,
    fold: ExpandingWindowFold,
    feature_columns: list[str],
    class_target: str,
    return_target: str,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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

    logistic, ridge = _linear_models()
    lightgbm_classifier = build_lightgbm_classifier()
    lightgbm_regressor = build_lightgbm_regressor()

    logistic.fit(x_train, y_train_class)
    ridge.fit(x_train, train[return_target])
    lightgbm_classifier.fit(x_train, y_train_class)
    lightgbm_regressor.fit(x_train, train[return_target])

    logistic_prediction = logistic.predict(x_validation)
    lightgbm_class_prediction = lightgbm_classifier.predict(x_validation).astype(int)
    ridge_prediction = ridge.predict(x_validation)
    lightgbm_return_prediction = lightgbm_regressor.predict(x_validation)

    logistic_metrics = detailed_classification_metrics(y_validation_class, logistic_prediction)
    lightgbm_classifier_metrics = detailed_classification_metrics(
        y_validation_class, lightgbm_class_prediction
    )
    zero_metrics, ridge_metrics, lightgbm_regressor_metrics = _regression_comparison(
        validation[return_target], ridge_prediction, lightgbm_return_prediction
    )

    logistic_probabilities = logistic.predict_proba(x_validation)
    logistic_classes = logistic.named_steps["model"].classes_
    lightgbm_probabilities = lightgbm_classifier.predict_proba(x_validation)

    logistic_validation = _attach_probabilities(
        validation, logistic_probabilities, logistic_classes
    )
    lightgbm_validation = _attach_probabilities(
        validation, lightgbm_probabilities, lightgbm_classifier.classes_
    )
    logistic_economics = _simulation_metrics(logistic_validation, horizon)
    lightgbm_economics = _simulation_metrics(lightgbm_validation, horizon)

    metadata = fold.metadata()
    metadata.update(
        {
            "train_rows_after_target_filter": int(len(train)),
            "validation_rows_after_target_filter": int(len(validation)),
            "train_time_start_seconds": float(train["time_seconds"].min()),
            "train_time_end_seconds": float(train["time_seconds"].max()),
            "validation_time_start_seconds": float(validation["time_seconds"].min()),
            "validation_time_end_seconds": float(validation["time_seconds"].max()),
        }
    )

    row = {
        "horizon": horizon,
        **metadata,
        "logistic_balanced_accuracy": logistic_metrics["balanced_accuracy"],
        "lightgbm_balanced_accuracy": lightgbm_classifier_metrics["balanced_accuracy"],
        "lightgbm_balanced_accuracy_delta": (
            lightgbm_classifier_metrics["balanced_accuracy"] - logistic_metrics["balanced_accuracy"]
        ),
        "logistic_macro_f1": logistic_metrics["macro_f1"],
        "lightgbm_macro_f1": lightgbm_classifier_metrics["macro_f1"],
        "lightgbm_macro_f1_delta": (
            lightgbm_classifier_metrics["macro_f1"] - logistic_metrics["macro_f1"]
        ),
        "zero_mae_bps": zero_metrics["mae_bps"],
        "ridge_mae_bps": ridge_metrics["mae_bps"],
        "lightgbm_mae_bps": lightgbm_regressor_metrics["mae_bps"],
        "ridge_mae_improvement_pct": _percentage_improvement(
            zero_metrics["mae_bps"], ridge_metrics["mae_bps"]
        ),
        "lightgbm_mae_improvement_pct": _percentage_improvement(
            zero_metrics["mae_bps"], lightgbm_regressor_metrics["mae_bps"]
        ),
        "lightgbm_mae_delta_vs_ridge_bps": (
            ridge_metrics["mae_bps"] - lightgbm_regressor_metrics["mae_bps"]
        ),
        "ridge_rmse_bps": ridge_metrics["rmse_bps"],
        "lightgbm_rmse_bps": lightgbm_regressor_metrics["rmse_bps"],
        "ridge_rank_ic": ridge_metrics["rank_ic"],
        "lightgbm_rank_ic": lightgbm_regressor_metrics["rank_ic"],
        "lightgbm_rank_ic_delta": (
            lightgbm_regressor_metrics["rank_ic"] - ridge_metrics["rank_ic"]
        ),
        "ridge_nonzero_directional_accuracy": ridge_metrics["nonzero_directional_accuracy"],
        "lightgbm_nonzero_directional_accuracy": lightgbm_regressor_metrics[
            "nonzero_directional_accuracy"
        ],
        "logistic_active_signal_fraction": logistic_economics["active_signal_fraction"],
        "lightgbm_active_signal_fraction": lightgbm_economics["active_signal_fraction"],
        "logistic_mean_gross_active_bps": logistic_economics["mean_gross_return_active_bps"],
        "lightgbm_mean_gross_active_bps": lightgbm_economics["mean_gross_return_active_bps"],
        "logistic_mean_net_active_bps": logistic_economics["mean_net_return_active_bps"],
        "lightgbm_mean_net_active_bps": lightgbm_economics["mean_net_return_active_bps"],
        "logistic_break_even_cost_fraction": logistic_economics["break_even_cost_fraction"],
        "lightgbm_break_even_cost_fraction": lightgbm_economics["break_even_cost_fraction"],
    }

    details = {
        "fold_metadata": metadata,
        "classification": {
            "balanced_logistic": logistic_metrics,
            "lightgbm_classifier": lightgbm_classifier_metrics,
        },
        "regression": {
            "zero_return": zero_metrics,
            "ridge": ridge_metrics,
            "lightgbm_regressor": lightgbm_regressor_metrics,
        },
        "economics_at_threshold_0_10": {
            "balanced_logistic": logistic_economics,
            "lightgbm_classifier": lightgbm_economics,
        },
    }

    importance_rows: list[dict[str, Any]] = []
    classifier_importance = normalized_feature_importance(
        feature_columns, lightgbm_classifier.feature_importances_
    )
    regressor_importance = normalized_feature_importance(
        feature_columns, lightgbm_regressor.feature_importances_
    )
    for model_name, values in (
        ("lightgbm_classifier", classifier_importance),
        ("lightgbm_regressor", regressor_importance),
    ):
        for feature, importance in values.items():
            importance_rows.append(
                {
                    "horizon": horizon,
                    "fold": fold.fold,
                    "model": model_name,
                    "feature": feature,
                    "normalized_gain_importance": importance,
                }
            )
    return row, details, importance_rows


def _ablation_fold(
    data: pd.DataFrame,
    fold: ExpandingWindowFold,
    feature_set_name: str,
    feature_columns: list[str],
    class_target: str,
    return_target: str,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = data.iloc[fold.train_indices].copy()
    validation = data.iloc[fold.validation_indices].copy()
    train = train.dropna(subset=[class_target, return_target])
    validation = validation.dropna(subset=[class_target, return_target])

    classifier = build_lightgbm_classifier()
    regressor = build_lightgbm_regressor()
    classifier.fit(train[feature_columns], train[class_target].astype(int))
    regressor.fit(train[feature_columns], train[return_target])

    class_prediction = classifier.predict(validation[feature_columns]).astype(int)
    return_prediction = regressor.predict(validation[feature_columns])
    class_metrics = detailed_classification_metrics(
        validation[class_target].astype(int), class_prediction
    )
    zero_metrics = detailed_regression_metrics(
        validation[return_target], np.zeros(len(validation), dtype=float)
    )
    regression_metrics = detailed_regression_metrics(validation[return_target], return_prediction)

    probabilities = classifier.predict_proba(validation[feature_columns])
    economic_frame = _attach_probabilities(validation, probabilities, classifier.classes_)
    economics = _simulation_metrics(economic_frame, horizon)

    row = {
        "horizon": horizon,
        "fold": fold.fold,
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "balanced_accuracy": class_metrics["balanced_accuracy"],
        "macro_f1": class_metrics["macro_f1"],
        "mae_bps": regression_metrics["mae_bps"],
        "mae_improvement_pct": _percentage_improvement(
            zero_metrics["mae_bps"], regression_metrics["mae_bps"]
        ),
        "rank_ic": regression_metrics["rank_ic"],
        "nonzero_directional_accuracy": regression_metrics["nonzero_directional_accuracy"],
        "active_signal_fraction": economics["active_signal_fraction"],
        "mean_gross_return_active_bps": economics["mean_gross_return_active_bps"],
        "mean_net_return_active_bps": economics["mean_net_return_active_bps"],
        "break_even_cost_fraction": economics["break_even_cost_fraction"],
    }
    details = {
        "fold": fold.fold,
        "feature_set": feature_set_name,
        "feature_columns": feature_columns,
        "classification": class_metrics,
        "zero_return": zero_metrics,
        "regression": regression_metrics,
        "economics_at_threshold_0_10": economics,
    }
    return row, details


def _aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"fold", "horizon"}
    ]
    aggregations = {column: ["mean", "std", "min", "max"] for column in numeric_columns}
    result = frame.groupby(groups, dropna=False).agg(aggregations)
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    result = result.reset_index()

    positive_columns = [
        column
        for column in (
            "lightgbm_balanced_accuracy_delta",
            "lightgbm_macro_f1_delta",
            "lightgbm_mae_delta_vs_ridge_bps",
            "lightgbm_rank_ic_delta",
            "mae_improvement_pct",
            "rank_ic",
            "mean_gross_return_active_bps",
        )
        if column in frame.columns
    ]
    if positive_columns:
        counts = (
            frame.assign(**{f"{column}_positive": frame[column] > 0 for column in positive_columns})
            .groupby(groups, dropna=False)[[f"{column}_positive" for column in positive_columns]]
            .sum()
            .reset_index()
        )
        result = result.merge(counts, on=groups, how="left")
    return result


def _write_cross_horizon_summary(output_directory: Path) -> None:
    frames = []
    importance_frames = []
    for horizon in (10, 50, 100):
        comparison_path = output_directory / f"phase_c_h{horizon}_model_comparison.csv"
        importance_path = output_directory / f"phase_c_h{horizon}_feature_importance.csv"
        if comparison_path.exists():
            frames.append(pd.read_csv(comparison_path))
        if importance_path.exists():
            importance_frames.append(pd.read_csv(importance_path))

    payload: dict[str, Any] = {"result_schema_version": SCHEMA_VERSION}
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        summary = _aggregate(combined, ["horizon"])
        summary.to_csv(output_directory / "phase_c_model_summary.csv", index=False)
        payload["model_summary"] = summary.to_dict(orient="records")

    if importance_frames:
        importance = pd.concat(importance_frames, ignore_index=True)
        importance_summary = (
            importance.groupby(["horizon", "model", "feature"], dropna=False)[
                "normalized_gain_importance"
            ]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .sort_values(["horizon", "model", "mean"], ascending=[True, True, False])
        )
        importance_summary.to_csv(
            output_directory / "phase_c_feature_importance_summary.csv",
            index=False,
        )
        payload["feature_importance_summary"] = importance_summary.to_dict(orient="records")

    ablation_path = output_directory / "phase_c_h50_ablation.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation_summary = _aggregate(ablation, ["feature_set"])
        ablation_summary.to_csv(output_directory / "phase_c_ablation_summary.csv", index=False)
        payload["ablation_summary"] = ablation_summary.to_dict(orient="records")

    if len(payload) > 1:
        (output_directory / "phase_c_summary_metrics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )


def run_model_comparison(
    ticker: str,
    levels: int,
    horizon: int,
    output_directory: Path,
    n_folds: int = 5,
    purge_events: int = 100,
    max_rows: int | None = None,
    run_ablation: bool = False,
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

    comparison_rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold in folds:
        print(f"Horizon {horizon}: model comparison fold {fold.fold}/{len(folds)}")
        row, details, fold_importance = _comparison_fold(
            data=data,
            fold=fold,
            feature_columns=feature_columns,
            class_target=class_target,
            return_target=return_target,
            horizon=horizon,
        )
        comparison_rows.append(row)
        fold_details.append(details)
        importance_rows.extend(fold_importance)

    output_directory.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(
        output_directory / f"phase_c_h{horizon}_model_comparison.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        output_directory / f"phase_c_h{horizon}_feature_importance.csv", index=False
    )

    result: dict[str, Any] = {
        "configuration": {
            "result_schema_version": SCHEMA_VERSION,
            "ticker": ticker.upper(),
            "levels": levels,
            "horizon_events": horizon,
            "max_rows": max_rows,
            "rows_loaded": int(len(data)),
            "folds": n_folds,
            "development_fraction": 0.80,
            "exploratory_holdout_fraction": 0.20,
            "requested_purge_events": purge_events,
            "effective_purge_events": effective_purge,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "fixed_models_no_validation_tuning": True,
            "run_ablation": run_ablation,
            "ablation_primary_horizon": PRIMARY_HORIZON,
            "split_design": "purged_expanding_window_first_80_percent",
        },
        "model_parameters": {
            "balanced_logistic": {
                "C": 0.10,
                "max_iter": 500,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            },
            "ridge": {"alpha": 10.0},
            "lightgbm_classifier": build_lightgbm_classifier().get_params(),
            "lightgbm_regressor": build_lightgbm_regressor().get_params(),
        },
        "folds": fold_details,
    }

    if run_ablation:
        if horizon != PRIMARY_HORIZON:
            raise ValueError(f"Feature-family ablation is restricted to {PRIMARY_HORIZON} events.")
        ablation_rows: list[dict[str, Any]] = []
        ablation_details: list[dict[str, Any]] = []
        for feature_set in ablation_feature_sets(levels=levels):
            for fold in folds:
                print(
                    f"Horizon {horizon}: {feature_set.name} ablation fold {fold.fold}/{len(folds)}"
                )
                row, details = _ablation_fold(
                    data=data,
                    fold=fold,
                    feature_set_name=feature_set.name,
                    feature_columns=list(feature_set.columns),
                    class_target=class_target,
                    return_target=return_target,
                    horizon=horizon,
                )
                ablation_rows.append(row)
                ablation_details.append(details)
        pd.DataFrame(ablation_rows).to_csv(
            output_directory / "phase_c_h50_ablation.csv", index=False
        )
        result["ablation_feature_sets"] = [
            feature_set.as_dict() for feature_set in ablation_feature_sets(levels=levels)
        ]
        result["ablation_folds"] = ablation_details

    (output_directory / f"phase_c_h{horizon}_metrics.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_cross_horizon_summary(output_directory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare linear baselines with fixed LightGBM models using purged "
            "walk-forward validation and optional 50-event feature ablation."
        )
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, choices=[10, 50, 100], default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    result = run_model_comparison(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        output_directory=args.output_directory,
        n_folds=args.folds,
        purge_events=args.purge_events,
        max_rows=args.max_rows,
        run_ablation=args.run_ablation,
    )
    print(json.dumps(result["configuration"], indent=2))


if __name__ == "__main__":
    main()
