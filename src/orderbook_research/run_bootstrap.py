from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from orderbook_research.bootstrap import (
    BOOTSTRAP_BENCHMARKS,
    BootstrapResult,
    bootstrap_fold_metrics,
    summarize_draws,
)
from orderbook_research.features import (
    add_snapshot_features,
    default_feature_columns,
)
from orderbook_research.io import load_lobster_pair
from orderbook_research.run_walk_forward import _build_models
from orderbook_research.simulation import non_overlapping_signal_simulation
from orderbook_research.targets import add_event_horizon_targets
from orderbook_research.walk_forward import expanding_window_folds


METRIC_NAMES = tuple(BOOTSTRAP_BENCHMARKS)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not serializable")


def _fit_fold_and_bootstrap(
    data: pd.DataFrame,
    fold: Any,
    feature_columns: list[str],
    class_target: str,
    return_target: str,
    horizon: int,
    n_bootstrap: int,
    event_block_length: int,
    confidence_level: float,
    random_seed: int,
) -> BootstrapResult:
    train = data.iloc[fold.train_indices].copy()
    validation = data.iloc[fold.validation_indices].copy()
    train = train.dropna(subset=[class_target, return_target])
    validation = validation.dropna(subset=[class_target, return_target])

    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.fold} is empty after target filtering.")

    _, logistic, ridge = _build_models()
    logistic.fit(train[feature_columns], train[class_target].astype(int))
    ridge.fit(train[feature_columns], train[return_target])

    ridge_prediction = ridge.predict(validation[feature_columns])
    probabilities = logistic.predict_proba(validation[feature_columns])
    fitted_classes = logistic.named_steps["model"].classes_
    probability_columns = {
        int(label): position
        for position, label in enumerate(fitted_classes)
    }
    if -1 not in probability_columns or 1 not in probability_columns:
        raise ValueError(f"Fold {fold.fold} is missing up/down probabilities.")

    validation["probability_down"] = probabilities[
        :, probability_columns[-1]
    ]
    validation["probability_up"] = probabilities[
        :, probability_columns[1]
    ]
    simulation, _ = non_overlapping_signal_simulation(
        validation,
        horizon=horizon,
    )

    return bootstrap_fold_metrics(
        actual=validation[return_target].to_numpy(dtype=float),
        predicted=np.asarray(ridge_prediction, dtype=float),
        simulation=simulation,
        horizon=horizon,
        n_bootstrap=n_bootstrap,
        event_block_length=event_block_length,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )


def _wide_row(
    horizon: int,
    scope: str,
    fold: int | None,
    intervals: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "horizon": horizon,
        "scope": scope,
        "fold": fold if fold is not None else "",
    }
    for metric, summary in intervals.items():
        row[f"{metric}_estimate"] = summary["estimate"]
        row[f"{metric}_ci_lower"] = summary["ci_lower"]
        row[f"{metric}_ci_upper"] = summary["ci_upper"]
        row[f"{metric}_probability_above_benchmark"] = summary[
            "probability_above_benchmark"
        ]
    return row


def run_bootstrap(
    ticker: str,
    levels: int,
    horizon: int,
    output_directory: Path,
    n_folds: int = 5,
    purge_events: int = 100,
    n_bootstrap: int = 1_000,
    event_block_length: int = 1_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    max_rows: int | None = None,
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

    fold_results: list[dict[str, Any]] = []
    fold_draws: dict[str, list[np.ndarray]] = {
        metric: [] for metric in METRIC_NAMES
    }
    csv_rows: list[dict[str, Any]] = []

    for fold in folds:
        print(
            f"Horizon {horizon}: bootstrap fold {fold.fold}/{len(folds)}"
        )
        result = _fit_fold_and_bootstrap(
            data=data,
            fold=fold,
            feature_columns=feature_columns,
            class_target=class_target,
            return_target=return_target,
            horizon=horizon,
            n_bootstrap=n_bootstrap,
            event_block_length=event_block_length,
            confidence_level=confidence_level,
            random_seed=random_seed + horizon * 100 + fold.fold,
        )

        fold_results.append(
            {
                "fold": fold.fold,
                "fold_metadata": fold.metadata(),
                "bootstrap_sample": {
                    "event_rows_used": result.event_rows_used,
                    "event_block_length": result.event_block_length,
                    "simulation_rows_used": result.simulation_rows_used,
                    "simulation_block_length": result.simulation_block_length,
                },
                "metrics": result.intervals,
            }
        )
        csv_rows.append(
            _wide_row(
                horizon=horizon,
                scope="fold",
                fold=fold.fold,
                intervals=result.intervals,
            )
        )
        for metric in METRIC_NAMES:
            fold_draws[metric].append(result.draws[metric])

    horizon_observed = {
        metric: float(
            np.mean(
                [
                    fold_result["metrics"][metric]["estimate"]
                    for fold_result in fold_results
                ]
            )
        )
        for metric in METRIC_NAMES
    }
    horizon_draws = {
        metric: np.vstack(draw_list).mean(axis=0)
        for metric, draw_list in fold_draws.items()
    }
    horizon_intervals = summarize_draws(
        observed=horizon_observed,
        draws=horizon_draws,
        confidence_level=confidence_level,
    )
    csv_rows.append(
        _wide_row(
            horizon=horizon,
            scope="horizon_mean",
            fold=None,
            intervals=horizon_intervals,
        )
    )

    payload: dict[str, Any] = {
        "configuration": {
            "result_schema_version": "0.4.0",
            "ticker": ticker.upper(),
            "levels": levels,
            "horizon_events": horizon,
            "folds": n_folds,
            "purge_events": effective_purge,
            "bootstrap_draws": n_bootstrap,
            "event_block_length": event_block_length,
            "confidence_level": confidence_level,
            "random_seed": random_seed,
            "rank_ic_bootstrap": (
                "moving blocks over fold-level rank-transformed observations"
            ),
            "economic_bootstrap": (
                "moving blocks over active non-overlapping signal observations"
            ),
            "exploratory_holdout_used": False,
        },
        "folds": fold_results,
        "horizon_mean": horizon_intervals,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(
        output_directory / f"bootstrap_h{horizon}_intervals.csv",
        index=False,
    )
    (
        output_directory / f"bootstrap_h{horizon}_metrics.json"
    ).write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _cross_horizon_summary(output_directory)
    return payload


def _cross_horizon_summary(output_directory: Path) -> None:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "result_schema_version": "0.4.0",
        "horizons": {},
    }

    for horizon in (10, 50, 100):
        path = output_directory / f"bootstrap_h{horizon}_metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        intervals = data["horizon_mean"]
        payload["horizons"][str(horizon)] = intervals
        rows.append(
            _wide_row(
                horizon=horizon,
                scope="horizon_mean",
                fold=None,
                intervals=intervals,
            )
        )

    if not rows:
        return

    pd.DataFrame(rows).sort_values("horizon").to_csv(
        output_directory / "bootstrap_summary.csv",
        index=False,
    )
    (output_directory / "bootstrap_summary_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Moving-block bootstrap for walk-forward metrics."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, choices=[10, 50, 100], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    parser.add_argument("--block-length", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/tables"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bootstrap(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        output_directory=args.output_directory,
        n_folds=args.folds,
        purge_events=args.purge_events,
        n_bootstrap=args.bootstrap_draws,
        event_block_length=args.block_length,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
