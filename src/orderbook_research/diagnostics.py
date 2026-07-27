from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, skew, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

SESSION_OPEN_SECONDS = 34_200.0


def _finite_array(values: pd.Series | np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def sample_autocorrelation(
    values: pd.Series | np.ndarray | Iterable[float],
    lag: int,
) -> float:
    """Return Pearson autocorrelation at one positive lag.

    A value of zero is returned when the lag is invalid, the sample is too
    small, or either side has no variation.
    """
    if lag <= 0:
        raise ValueError("lag must be positive")

    array = _finite_array(values)
    if len(array) <= lag:
        return 0.0

    left = array[:-lag]
    right = array[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0

    value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def autocorrelation_table(
    values: pd.Series | np.ndarray | Iterable[float],
    lags: Iterable[int],
    series_name: str,
) -> pd.DataFrame:
    """Return a tidy autocorrelation table."""
    rows = [
        {
            "series": series_name,
            "lag": int(lag),
            "autocorrelation": sample_autocorrelation(values, int(lag)),
        }
        for lag in lags
    ]
    return pd.DataFrame(rows)


def ljung_box_table(
    values: pd.Series | np.ndarray | Iterable[float],
    max_lags: Iterable[int],
) -> pd.DataFrame:
    """Compute Ljung-Box Q statistics without requiring statsmodels."""
    array = _finite_array(values)
    n = len(array)
    rows: list[dict[str, float | int]] = []

    for max_lag in max_lags:
        max_lag = int(max_lag)
        if max_lag <= 0:
            raise ValueError("Ljung-Box lags must be positive")

        effective_lag = min(max_lag, max(0, n - 1))
        if effective_lag == 0:
            q_statistic = 0.0
            p_value = 1.0
        else:
            terms = []
            for lag in range(1, effective_lag + 1):
                rho = sample_autocorrelation(array, lag)
                terms.append((rho**2) / (n - lag))
            q_statistic = float(n * (n + 2) * np.sum(terms))
            p_value = float(chi2.sf(q_statistic, df=effective_lag))

        rows.append(
            {
                "max_lag": max_lag,
                "effective_lag": effective_lag,
                "observations": n,
                "q_statistic": q_statistic,
                "p_value": p_value,
                "reject_at_5pct": int(p_value < 0.05),
            }
        )

    return pd.DataFrame(rows)


def descriptive_residual_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Return residual shape and calibration diagnostics."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_array = actual_array[valid]
    predicted_array = predicted_array[valid]

    if len(actual_array) == 0:
        raise ValueError("No finite actual/predicted pairs.")

    residual = actual_array - predicted_array
    zero_prediction = np.zeros(len(actual_array), dtype=float)
    zero_mae = float(mean_absolute_error(actual_array, zero_prediction))
    ridge_mae = float(mean_absolute_error(actual_array, predicted_array))
    zero_rmse = float(mean_squared_error(actual_array, zero_prediction) ** 0.5)
    ridge_rmse = float(mean_squared_error(actual_array, predicted_array) ** 0.5)

    if np.unique(predicted_array).size >= 2:
        slope, intercept = np.polyfit(predicted_array, actual_array, deg=1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = 0.0
        intercept = float(np.mean(actual_array))

    if np.unique(actual_array).size >= 2 and np.unique(predicted_array).size >= 2:
        rank_value = spearmanr(actual_array, predicted_array).statistic
        rank_ic = float(rank_value) if np.isfinite(rank_value) else 0.0
    else:
        rank_ic = 0.0

    nonzero = actual_array != 0.0
    nonzero_directional_accuracy = (
        float(np.mean(np.sign(actual_array[nonzero]) == np.sign(predicted_array[nonzero])))
        if nonzero.any()
        else 0.0
    )

    return {
        "observations": int(len(actual_array)),
        "nonzero_observations": int(nonzero.sum()),
        "zero_mae_bps": zero_mae,
        "ridge_mae_bps": ridge_mae,
        "mae_improvement_bps": float(zero_mae - ridge_mae),
        "mae_improvement_pct": (
            float((zero_mae - ridge_mae) / zero_mae * 100.0) if zero_mae > 0.0 else 0.0
        ),
        "zero_rmse_bps": zero_rmse,
        "ridge_rmse_bps": ridge_rmse,
        "rmse_improvement_bps": float(zero_rmse - ridge_rmse),
        "rank_ic": rank_ic,
        "nonzero_directional_accuracy": nonzero_directional_accuracy,
        "mean_actual_bps": float(np.mean(actual_array)),
        "mean_predicted_bps": float(np.mean(predicted_array)),
        "residual_mean_bps": float(np.mean(residual)),
        "residual_std_bps": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "residual_skewness": float(skew(residual, bias=False)) if len(residual) > 2 else 0.0,
        "residual_excess_kurtosis": float(kurtosis(residual, fisher=True, bias=False))
        if len(residual) > 3
        else 0.0,
        "residual_q01_bps": float(np.quantile(residual, 0.01)),
        "residual_q05_bps": float(np.quantile(residual, 0.05)),
        "residual_median_bps": float(np.quantile(residual, 0.50)),
        "residual_q95_bps": float(np.quantile(residual, 0.95)),
        "residual_q99_bps": float(np.quantile(residual, 0.99)),
        "calibration_intercept_bps": intercept,
        "calibration_slope": slope,
    }


def non_overlapping_residuals(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return horizon-spaced residuals to avoid mechanical target overlap."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    residual = actual_array[valid] - predicted_array[valid]
    return residual[::horizon]


def _clock_label(seconds_from_midnight: float) -> str:
    base = datetime(2000, 1, 1)
    moment = base + timedelta(seconds=float(seconds_from_midnight))
    return moment.strftime("%H:%M")


def add_time_bucket(
    frame: pd.DataFrame,
    bucket_minutes: int = 30,
) -> pd.DataFrame:
    """Add fixed clock-time bucket identifiers and human-readable labels."""
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    if "time_seconds" not in frame.columns:
        raise KeyError("frame must contain time_seconds")

    result = frame.copy()
    bucket_seconds = float(bucket_minutes * 60)
    bucket_index = np.floor(
        (result["time_seconds"] - SESSION_OPEN_SECONDS) / bucket_seconds
    ).astype(int)
    bucket_start = SESSION_OPEN_SECONDS + bucket_index * bucket_seconds
    bucket_end = bucket_start + bucket_seconds

    result["time_bucket_index"] = bucket_index
    result["time_bucket_start_seconds"] = bucket_start
    result["time_bucket_end_seconds"] = bucket_end
    result["time_bucket"] = [
        f"{_clock_label(start)}-{_clock_label(end)}"
        for start, end in zip(bucket_start, bucket_end, strict=False)
    ]
    return result


def time_bucket_diagnostics(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    bucket_minutes: int = 30,
    minimum_observations: int = 100,
) -> pd.DataFrame:
    """Summarise out-of-sample forecast quality by fixed time bucket."""
    required = {
        "time_seconds",
        actual_column,
        predicted_column,
        "spread_bps",
        "mid_log_return",
        "event_interarrival_us",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing time-bucket columns: {sorted(missing)}")

    bucketed = add_time_bucket(frame, bucket_minutes=bucket_minutes)
    rows: list[dict[str, float | int | str]] = []

    for bucket, group in bucketed.groupby("time_bucket", sort=True):
        actual = group[actual_column].to_numpy(dtype=float)
        predicted = group[predicted_column].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[valid]
        predicted = predicted[valid]

        if len(actual) < minimum_observations:
            continue

        diagnostics = descriptive_residual_metrics(actual, predicted)
        one_event_return_bps = group["mid_log_return"].to_numpy(dtype=float) * 10_000.0
        one_event_return_bps = one_event_return_bps[np.isfinite(one_event_return_bps)]

        rows.append(
            {
                "time_bucket": str(bucket),
                "time_bucket_index": int(group["time_bucket_index"].iloc[0]),
                "time_start_seconds": float(group["time_seconds"].min()),
                "time_end_seconds": float(group["time_seconds"].max()),
                **diagnostics,
                "mean_spread_bps": float(group["spread_bps"].mean()),
                "median_spread_bps": float(group["spread_bps"].median()),
                "median_event_interarrival_us": float(group["event_interarrival_us"].median()),
                "one_event_return_std_bps": (
                    float(np.std(one_event_return_bps, ddof=1))
                    if len(one_event_return_bps) > 1
                    else 0.0
                ),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time_bucket_index").reset_index(drop=True)
