from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orderbook_research.diagnostics import (
    autocorrelation_table,
    descriptive_residual_metrics,
    ljung_box_table,
    non_overlapping_residuals,
    time_bucket_diagnostics,
)
from orderbook_research.features import (
    add_snapshot_features,
    default_feature_columns,
)
from orderbook_research.io import load_lobster_pair
from orderbook_research.targets import add_event_horizon_targets
from orderbook_research.walk_forward import expanding_window_folds


MARKET_ACF_LAGS = (1, 5, 10, 20, 50, 100)
RESIDUAL_ACF_LAGS = (1, 2, 5, 10, 20)
LJUNG_BOX_LAGS = (5, 10, 20)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not serializable")


def _ridge_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def _acf_lookup(table: pd.DataFrame, lag: int) -> float:
    match = table.loc[table["lag"] == lag, "autocorrelation"]
    return float(match.iloc[0]) if len(match) else 0.0


def _ljung_lookup(table: pd.DataFrame, max_lag: int, field: str) -> float | int:
    match = table.loc[table["max_lag"] == max_lag, field]
    if not len(match):
        return 0.0
    value = match.iloc[0]
    return int(value) if field == "reject_at_5pct" else float(value)


def _evaluate_fold(
    data: pd.DataFrame,
    fold: Any,
    feature_columns: list[str],
    return_target: str,
    horizon: int,
    market_acf_lags: tuple[int, ...],
    residual_acf_lags: tuple[int, ...],
    ljung_box_lags: tuple[int, ...],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train = data.iloc[fold.train_indices].copy()
    validation = data.iloc[fold.validation_indices].copy()
    train = train.dropna(subset=[return_target])
    validation = validation.dropna(subset=[return_target])

    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.fold} is empty after target filtering.")

    ridge = _ridge_model()
    ridge.fit(train[feature_columns], train[return_target])
    predicted = ridge.predict(validation[feature_columns])
    validation["predicted_return_bps"] = predicted
    validation["residual_bps"] = (
        validation[return_target].to_numpy(dtype=float) - predicted
    )
    validation["fold"] = fold.fold

    diagnostics = descriptive_residual_metrics(
        validation[return_target],
        predicted,
    )

    market_return = validation["mid_log_return"].to_numpy(dtype=float)
    market_return = market_return[np.isfinite(market_return)]
    squared_market_return = market_return**2
    non_overlap_residual = non_overlapping_residuals(
        validation[return_target],
        predicted,
        horizon=horizon,
    )

    market_acf = autocorrelation_table(
        market_return,
        market_acf_lags,
        series_name="one_event_mid_log_return",
    )
    squared_acf = autocorrelation_table(
        squared_market_return,
        market_acf_lags,
        series_name="squared_one_event_mid_log_return",
    )
    residual_acf = autocorrelation_table(
        non_overlap_residual,
        residual_acf_lags,
        series_name="non_overlapping_ridge_residual_bps",
    )
    autocorrelation = pd.concat(
        [market_acf, squared_acf, residual_acf],
        ignore_index=True,
    )
    autocorrelation.insert(0, "fold", fold.fold)
    autocorrelation.insert(0, "horizon", horizon)

    ljung_box = ljung_box_table(
        non_overlap_residual,
        max_lags=ljung_box_lags,
    )
    ljung_box.insert(0, "series", "non_overlapping_ridge_residual_bps")
    ljung_box.insert(0, "fold", fold.fold)
    ljung_box.insert(0, "horizon", horizon)

    metadata = fold.metadata()
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

    fold_row: dict[str, Any] = {
        "horizon": horizon,
        **metadata,
        **diagnostics,
        "non_overlapping_residual_observations": int(
            len(non_overlap_residual)
        ),
    }
    for lag in market_acf_lags:
        fold_row[f"mid_return_acf_lag_{lag}"] = _acf_lookup(
            market_acf,
            lag,
        )
        fold_row[f"squared_mid_return_acf_lag_{lag}"] = _acf_lookup(
            squared_acf,
            lag,
        )
    for lag in residual_acf_lags:
        fold_row[f"residual_acf_lag_{lag}"] = _acf_lookup(
            residual_acf,
            lag,
        )
    for lag in ljung_box_lags:
        fold_row[f"ljung_box_q_{lag}"] = _ljung_lookup(
            ljung_box,
            lag,
            "q_statistic",
        )
        fold_row[f"ljung_box_p_value_{lag}"] = _ljung_lookup(
            ljung_box,
            lag,
            "p_value",
        )
        fold_row[f"ljung_box_reject_5pct_{lag}"] = _ljung_lookup(
            ljung_box,
            lag,
            "reject_at_5pct",
        )

    details = {
        "fold_metadata": metadata,
        "regression_and_residual_diagnostics": diagnostics,
        "non_overlapping_residual_observations": int(
            len(non_overlap_residual)
        ),
        "autocorrelation": autocorrelation.to_dict(orient="records"),
        "ljung_box": ljung_box.to_dict(orient="records"),
    }

    oos_columns = [
        "time_seconds",
        "spread_bps",
        "mid_log_return",
        "event_interarrival_us",
        return_target,
        "predicted_return_bps",
        "residual_bps",
        "fold",
    ]
    oos_predictions = validation[oos_columns].copy()
    return fold_row, details, autocorrelation, ljung_box, oos_predictions


def _cross_horizon_summary(output_directory: Path) -> None:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "result_schema_version": "0.5.0",
        "horizons": {},
    }

    for horizon in (10, 50, 100):
        fold_path = output_directory / f"diagnostics_h{horizon}_folds.csv"
        time_path = output_directory / f"diagnostics_h{horizon}_time_buckets.csv"
        if not fold_path.exists():
            continue

        folds = pd.read_csv(fold_path)
        time_buckets = pd.read_csv(time_path) if time_path.exists() else pd.DataFrame()
        row: dict[str, Any] = {
            "horizon": horizon,
            "folds": int(len(folds)),
            "mean_rank_ic": float(folds["rank_ic"].mean()),
            "mean_mae_improvement_pct": float(
                folds["mae_improvement_pct"].mean()
            ),
            "mae_improvement_positive_folds": int(
                (folds["mae_improvement_pct"] > 0.0).sum()
            ),
            "mean_nonzero_directional_accuracy": float(
                folds["nonzero_directional_accuracy"].mean()
            ),
            "mean_calibration_slope": float(
                folds["calibration_slope"].mean()
            ),
            "mean_calibration_intercept_bps": float(
                folds["calibration_intercept_bps"].mean()
            ),
            "mean_residual_mean_bps": float(
                folds["residual_mean_bps"].mean()
            ),
            "mean_residual_std_bps": float(
                folds["residual_std_bps"].mean()
            ),
            "mean_residual_acf_lag_1": float(
                folds["residual_acf_lag_1"].mean()
            ),
            "mean_absolute_residual_acf_lag_1": float(
                folds["residual_acf_lag_1"].abs().mean()
            ),
            "residual_ljung_box_20_rejections_5pct": int(
                folds["ljung_box_reject_5pct_20"].sum()
            ),
            "mean_mid_return_acf_lag_1": float(
                folds["mid_return_acf_lag_1"].mean()
            ),
            "mean_squared_mid_return_acf_lag_1": float(
                folds["squared_mid_return_acf_lag_1"].mean()
            ),
            "time_buckets": int(len(time_buckets)),
            "time_buckets_positive_mae_improvement": int(
                (time_buckets["mae_improvement_pct"] > 0.0).sum()
            )
            if len(time_buckets)
            else 0,
            "time_buckets_positive_rank_ic": int(
                (time_buckets["rank_ic"] > 0.0).sum()
            )
            if len(time_buckets)
            else 0,
        }
        rows.append(row)
        payload["horizons"][str(horizon)] = row

    if not rows:
        return

    pd.DataFrame(rows).sort_values("horizon").to_csv(
        output_directory / "diagnostics_summary.csv",
        index=False,
    )
    (output_directory / "diagnostics_summary_metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def run_diagnostics(
    ticker: str,
    levels: int,
    horizon: int,
    output_directory: Path,
    n_folds: int = 5,
    purge_events: int = 100,
    bucket_minutes: int = 30,
    minimum_bucket_observations: int = 100,
    max_rows: int | None = None,
    market_acf_lags: tuple[int, ...] = MARKET_ACF_LAGS,
    residual_acf_lags: tuple[int, ...] = RESIDUAL_ACF_LAGS,
    ljung_box_lags: tuple[int, ...] = LJUNG_BOX_LAGS,
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
    return_target = f"future_return_bps_{horizon}"
    folds = expanding_window_folds(
        n_rows=len(data),
        n_folds=n_folds,
        development_fraction=0.80,
        initial_train_fraction=0.30,
        validation_fraction=0.10,
        purge_events=effective_purge,
    )

    fold_rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []
    autocorrelation_frames: list[pd.DataFrame] = []
    ljung_box_frames: list[pd.DataFrame] = []
    oos_frames: list[pd.DataFrame] = []

    for fold in folds:
        print(
            f"Horizon {horizon}: diagnostics fold {fold.fold}/{len(folds)}"
        )
        row, details, autocorrelation, ljung_box, oos_predictions = (
            _evaluate_fold(
                data=data,
                fold=fold,
                feature_columns=feature_columns,
                return_target=return_target,
                horizon=horizon,
                market_acf_lags=market_acf_lags,
                residual_acf_lags=residual_acf_lags,
                ljung_box_lags=ljung_box_lags,
            )
        )
        fold_rows.append(row)
        fold_details.append(details)
        autocorrelation_frames.append(autocorrelation)
        ljung_box_frames.append(ljung_box)
        oos_frames.append(oos_predictions)

    fold_frame = pd.DataFrame(fold_rows)
    autocorrelation_frame = pd.concat(
        autocorrelation_frames,
        ignore_index=True,
    )
    ljung_box_frame = pd.concat(ljung_box_frames, ignore_index=True)
    oos_frame = pd.concat(oos_frames, ignore_index=True).sort_values(
        "time_seconds"
    )
    time_bucket_frame = time_bucket_diagnostics(
        oos_frame,
        actual_column=return_target,
        predicted_column="predicted_return_bps",
        bucket_minutes=bucket_minutes,
        minimum_observations=minimum_bucket_observations,
    )
    if len(time_bucket_frame):
        time_bucket_frame.insert(0, "horizon", horizon)

    payload: dict[str, Any] = {
        "configuration": {
            "result_schema_version": "0.5.0",
            "ticker": ticker.upper(),
            "levels": levels,
            "horizon_events": horizon,
            "folds": n_folds,
            "purge_events": effective_purge,
            "development_fraction": 0.80,
            "exploratory_holdout_used": False,
            "ridge_alpha": 10.0,
            "market_acf_lags": list(market_acf_lags),
            "residual_acf_lags": list(residual_acf_lags),
            "ljung_box_lags": list(ljung_box_lags),
            "residual_sampling": (
                "every horizon-th validation observation to remove mechanical "
                "target overlap"
            ),
            "time_bucket_minutes": bucket_minutes,
            "minimum_bucket_observations": minimum_bucket_observations,
        },
        "folds": fold_details,
        "time_buckets": time_bucket_frame.to_dict(orient="records"),
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(
        output_directory / f"diagnostics_h{horizon}_folds.csv",
        index=False,
    )
    autocorrelation_frame.to_csv(
        output_directory / f"diagnostics_h{horizon}_autocorrelation.csv",
        index=False,
    )
    ljung_box_frame.to_csv(
        output_directory / f"diagnostics_h{horizon}_ljung_box.csv",
        index=False,
    )
    time_bucket_frame.to_csv(
        output_directory / f"diagnostics_h{horizon}_time_buckets.csv",
        index=False,
    )
    (output_directory / f"diagnostics_h{horizon}_metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _cross_horizon_summary(output_directory)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Econometric residual, autocorrelation, volatility-clustering, "
            "and time-of-day diagnostics."
        )
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[10, 50, 100],
        required=True,
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--bucket-minutes", type=int, default=30)
    parser.add_argument("--minimum-bucket-observations", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/tables"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diagnostics(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        output_directory=args.output_directory,
        n_folds=args.folds,
        purge_events=args.purge_events,
        bucket_minutes=args.bucket_minutes,
        minimum_bucket_observations=args.minimum_bucket_observations,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
