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
from orderbook_research.regimes import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    DEFAULT_COST_FRACTIONS,
    add_validation_regimes,
    economic_metrics,
    prepare_non_overlapping_signals,
    validate_grid,
)
from orderbook_research.targets import add_event_horizon_targets
from orderbook_research.train_baseline import detailed_regression_metrics
from orderbook_research.walk_forward import expanding_window_folds


REGIME_COLUMNS = {
    "spread": "spread_regime",
    "depth": "depth_regime",
    "volatility": "volatility_regime",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _models() -> tuple[Pipeline, Pipeline]:
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


def _regression_row(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    zero = detailed_regression_metrics(actual, np.zeros(len(actual), dtype=float))
    ridge = detailed_regression_metrics(actual, predicted)
    return {
        "validation_observations": int(ridge["observations"]),
        "zero_mae_bps": float(zero["mae_bps"]),
        "ridge_mae_bps": float(ridge["mae_bps"]),
        "ridge_mae_improvement_pct": _percentage_improvement(
            float(zero["mae_bps"]), float(ridge["mae_bps"])
        ),
        "ridge_rmse_bps": float(ridge["rmse_bps"]),
        "ridge_rank_ic": float(ridge["rank_ic"]),
        "ridge_nonzero_directional_accuracy": float(
            ridge["nonzero_directional_accuracy"]
        ),
    }


def _economic_row(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    result["sampled_observations"] = result.pop("observations")
    return result


def _aggregate_group(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "validation_observations",
        "sampled_observations",
        "ridge_mae_improvement_pct",
        "ridge_rank_ic",
        "ridge_nonzero_directional_accuracy",
        "active_signal_fraction",
        "mean_gross_return_active_bps",
        "mean_full_estimated_cost_active_bps",
        "mean_applied_cost_active_bps",
        "mean_net_return_active_bps",
        "active_hit_rate",
        "break_even_cost_fraction",
    ]
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std", "min", "max"]
        for metric in metrics
        if metric in frame.columns
    }
    summary = frame.groupby(group_columns, dropna=False).agg(aggregations)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    if "mean_net_return_active_bps" in frame.columns:
        positive = (
            frame.assign(
                positive_net_fold=frame["mean_net_return_active_bps"] > 0,
                positive_gross_fold=frame["mean_gross_return_active_bps"] > 0,
            )
            .groupby(group_columns, dropna=False)[
                ["positive_net_fold", "positive_gross_fold"]
            ]
            .sum()
            .reset_index()
        )
        summary = summary.merge(positive, on=group_columns, how="left")
    return summary


def _cross_horizon_summary(output_directory: Path) -> None:
    regime_frames: list[pd.DataFrame] = []
    sensitivity_frames: list[pd.DataFrame] = []
    intersection_frames: list[pd.DataFrame] = []

    for horizon in (10, 50, 100):
        regime_path = output_directory / f"phase_b_h{horizon}_regimes.csv"
        sensitivity_path = output_directory / f"phase_b_h{horizon}_threshold_cost_grid.csv"
        intersection_path = output_directory / f"phase_b_h{horizon}_spread_confidence.csv"
        if regime_path.exists():
            regime_frames.append(pd.read_csv(regime_path))
        if sensitivity_path.exists():
            sensitivity_frames.append(pd.read_csv(sensitivity_path))
        if intersection_path.exists():
            intersection_frames.append(pd.read_csv(intersection_path))

    payload: dict[str, Any] = {"result_schema_version": "0.6.0"}

    if regime_frames:
        regimes = pd.concat(regime_frames, ignore_index=True)
        regime_summary = _aggregate_group(
            regimes,
            ["horizon", "regime_type", "regime_label"],
        )
        regime_summary.to_csv(
            output_directory / "phase_b_regime_summary.csv",
            index=False,
        )
        payload["regime_summary"] = regime_summary.to_dict(orient="records")

    if sensitivity_frames:
        sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
        sensitivity_summary = _aggregate_group(
            sensitivity,
            ["horizon", "confidence_threshold", "cost_fraction"],
        )
        sensitivity_summary.to_csv(
            output_directory / "phase_b_sensitivity_summary.csv",
            index=False,
        )
        payload["sensitivity_summary"] = sensitivity_summary.to_dict(
            orient="records"
        )

    if intersection_frames:
        intersections = pd.concat(intersection_frames, ignore_index=True)
        intersection_summary = _aggregate_group(
            intersections,
            ["horizon", "spread_regime", "confidence_threshold"],
        )
        intersection_summary.to_csv(
            output_directory / "phase_b_spread_confidence_summary.csv",
            index=False,
        )
        payload["spread_confidence_summary"] = intersection_summary.to_dict(
            orient="records"
        )

    if len(payload) > 1:
        (output_directory / "phase_b_summary_metrics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _evaluate_fold(
    data: pd.DataFrame,
    fold: Any,
    feature_columns: list[str],
    class_target: str,
    return_target: str,
    horizon: int,
    levels: int,
    confidence_thresholds: tuple[float, ...],
    cost_fractions: tuple[float, ...],
    tick_size: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = data.iloc[fold.train_indices].copy()
    validation = data.iloc[fold.validation_indices].copy()
    train = train.dropna(subset=[class_target, return_target])
    validation = validation.dropna(subset=[class_target, return_target])

    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.fold} is empty after target filtering.")

    logistic, ridge = _models()
    logistic.fit(train[feature_columns], train[class_target].astype(int))
    ridge.fit(train[feature_columns], train[return_target])

    probabilities = logistic.predict_proba(validation[feature_columns])
    classes = logistic.named_steps["model"].classes_
    positions = {int(label): index for index, label in enumerate(classes)}
    if -1 not in positions or 1 not in positions:
        raise ValueError(f"Fold {fold.fold} classifier is missing up/down classes.")

    validation["probability_down"] = probabilities[:, positions[-1]]
    validation["probability_up"] = probabilities[:, positions[1]]
    validation["predicted_return_bps"] = ridge.predict(
        validation[feature_columns]
    )
    validation, regime_metadata = add_validation_regimes(
        train=train,
        validation=validation,
        levels=levels,
        volatility_window=50,
        tick_size=tick_size,
    )
    sampled = prepare_non_overlapping_signals(validation, horizon=horizon)

    regime_rows: list[dict[str, Any]] = []
    for regime_type, regime_column in REGIME_COLUMNS.items():
        labels = sorted(
            label
            for label in validation[regime_column].dropna().unique()
            if label != "unknown"
        )
        for label in labels:
            validation_subset = validation.loc[validation[regime_column] == label]
            sampled_subset = sampled.loc[sampled[regime_column] == label]
            if validation_subset.empty:
                continue

            regression = _regression_row(
                validation_subset[return_target],
                validation_subset["predicted_return_bps"].to_numpy(),
            )
            economics = economic_metrics(
                sampled_subset,
                horizon=horizon,
                confidence_threshold=0.10,
                cost_fraction=1.0,
            )
            regime_rows.append(
                {
                    "horizon": horizon,
                    "fold": fold.fold,
                    "regime_type": regime_type,
                    "regime_label": label,
                    "validation_fraction": float(
                        len(validation_subset) / len(validation)
                    ),
                    **regression,
                    **_economic_row(economics),
                }
            )

    sensitivity_rows: list[dict[str, Any]] = []
    for threshold in confidence_thresholds:
        for cost_fraction in cost_fractions:
            economics = economic_metrics(
                sampled,
                horizon=horizon,
                confidence_threshold=threshold,
                cost_fraction=cost_fraction,
            )
            sensitivity_rows.append(
                {
                    "horizon": horizon,
                    "fold": fold.fold,
                    "confidence_threshold": threshold,
                    "cost_fraction": cost_fraction,
                    **_economic_row(economics),
                }
            )

    spread_rows: list[dict[str, Any]] = []
    for spread_label in sorted(
        label
        for label in sampled["spread_regime"].dropna().unique()
        if label != "unknown"
    ):
        spread_subset = sampled.loc[sampled["spread_regime"] == spread_label]
        for threshold in confidence_thresholds:
            economics = economic_metrics(
                spread_subset,
                horizon=horizon,
                confidence_threshold=threshold,
                cost_fraction=1.0,
            )
            spread_rows.append(
                {
                    "horizon": horizon,
                    "fold": fold.fold,
                    "spread_regime": spread_label,
                    "confidence_threshold": threshold,
                    **_economic_row(economics),
                }
            )

    details = {
        "fold_metadata": {
            **fold.metadata(),
            "train_rows_after_target_filter": int(len(train)),
            "validation_rows_after_target_filter": int(len(validation)),
            "train_time_start_seconds": float(train["time_seconds"].min()),
            "train_time_end_seconds": float(train["time_seconds"].max()),
            "validation_time_start_seconds": float(
                validation["time_seconds"].min()
            ),
            "validation_time_end_seconds": float(validation["time_seconds"].max()),
        },
        "regime_metadata": regime_metadata,
    }
    return regime_rows, sensitivity_rows, spread_rows, details


def run_regime_analysis(
    ticker: str,
    levels: int,
    horizon: int,
    output_directory: Path,
    n_folds: int = 5,
    purge_events: int = 100,
    max_rows: int | None = None,
    tick_size: float = 0.01,
    confidence_thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
    cost_fractions: tuple[float, ...] = DEFAULT_COST_FRACTIONS,
) -> dict[str, Any]:
    confidence_thresholds = validate_grid(
        confidence_thresholds, "confidence_thresholds"
    )
    cost_fractions = validate_grid(cost_fractions, "cost_fractions")
    if any(value < 0 for value in confidence_thresholds):
        raise ValueError("Confidence thresholds cannot be negative.")
    if any(value < 0 or value > 1 for value in cost_fractions):
        raise ValueError("Cost fractions must be between zero and one.")

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

    regime_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []

    for fold in folds:
        print(f"Horizon {horizon}: Phase B fold {fold.fold}/{len(folds)}")
        fold_regimes, fold_sensitivity, fold_spread, details = _evaluate_fold(
            data=data,
            fold=fold,
            feature_columns=feature_columns,
            class_target=class_target,
            return_target=return_target,
            horizon=horizon,
            levels=levels,
            confidence_thresholds=confidence_thresholds,
            cost_fractions=cost_fractions,
            tick_size=tick_size,
        )
        regime_rows.extend(fold_regimes)
        sensitivity_rows.extend(fold_sensitivity)
        spread_rows.extend(fold_spread)
        fold_details.append(details)

    output_directory.mkdir(parents=True, exist_ok=True)
    regime_frame = pd.DataFrame(regime_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    spread_frame = pd.DataFrame(spread_rows)

    regime_frame.to_csv(
        output_directory / f"phase_b_h{horizon}_regimes.csv", index=False
    )
    sensitivity_frame.to_csv(
        output_directory / f"phase_b_h{horizon}_threshold_cost_grid.csv",
        index=False,
    )
    spread_frame.to_csv(
        output_directory / f"phase_b_h{horizon}_spread_confidence.csv",
        index=False,
    )

    result: dict[str, Any] = {
        "configuration": {
            "result_schema_version": "0.6.0",
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
            "tick_size": tick_size,
            "confidence_thresholds": list(confidence_thresholds),
            "cost_fractions": list(cost_fractions),
            "default_regime_threshold": 0.10,
            "default_regime_cost_fraction": 1.0,
            "split_design": "purged_expanding_window_first_80_percent",
        },
        "folds": fold_details,
        "row_counts": {
            "regime_rows": int(len(regime_frame)),
            "threshold_cost_rows": int(len(sensitivity_frame)),
            "spread_confidence_rows": int(len(spread_frame)),
        },
    }
    (output_directory / f"phase_b_h{horizon}_metrics.json").write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _cross_horizon_summary(output_directory)
    return result


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run spread, depth, volatility, confidence-threshold, and "
            "execution-cost regime analysis."
        )
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, choices=[10, 50, 100], default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument(
        "--confidence-thresholds",
        type=_parse_float_list,
        default=DEFAULT_CONFIDENCE_THRESHOLDS,
    )
    parser.add_argument(
        "--cost-fractions",
        type=_parse_float_list,
        default=DEFAULT_COST_FRACTIONS,
    )
    parser.add_argument(
        "--output-directory", type=Path, default=Path("reports/tables")
    )
    args = parser.parse_args()

    result = run_regime_analysis(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        output_directory=args.output_directory,
        n_folds=args.folds,
        purge_events=args.purge_events,
        max_rows=args.max_rows,
        tick_size=args.tick_size,
        confidence_thresholds=args.confidence_thresholds,
        cost_fractions=args.cost_fractions,
    )
    print(json.dumps(result["row_counts"], indent=2))


if __name__ == "__main__":
    main()
