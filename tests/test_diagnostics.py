from __future__ import annotations

import numpy as np
import pandas as pd

from orderbook_research.diagnostics import (
    add_time_bucket,
    descriptive_residual_metrics,
    ljung_box_table,
    non_overlapping_residuals,
    sample_autocorrelation,
    time_bucket_diagnostics,
)


def test_sample_autocorrelation_detects_alternation() -> None:
    values = np.array([1.0, -1.0] * 100)
    assert sample_autocorrelation(values, lag=1) < -0.99
    assert sample_autocorrelation(values, lag=2) > 0.99


def test_ljung_box_detects_serial_dependence() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(size=2_000)
    series = np.zeros_like(noise)
    for index in range(1, len(series)):
        series[index] = 0.85 * series[index - 1] + noise[index]

    result = ljung_box_table(series, max_lags=(10,))
    assert int(result.loc[0, "reject_at_5pct"]) == 1
    assert float(result.loc[0, "p_value"]) < 0.05


def test_non_overlapping_residuals_use_horizon_spacing() -> None:
    actual = np.arange(20, dtype=float)
    predicted = np.zeros(20, dtype=float)
    residuals = non_overlapping_residuals(actual, predicted, horizon=5)
    assert residuals.tolist() == [0.0, 5.0, 10.0, 15.0]


def test_descriptive_residual_metrics_include_calibration() -> None:
    predicted = np.linspace(-2.0, 2.0, 500)
    actual = 0.25 + 1.5 * predicted
    result = descriptive_residual_metrics(actual, predicted)
    assert abs(float(result["calibration_intercept_bps"]) - 0.25) < 1e-10
    assert abs(float(result["calibration_slope"]) - 1.5) < 1e-10
    assert float(result["rank_ic"]) > 0.99


def test_time_bucket_diagnostics_create_clock_buckets() -> None:
    n_rows = 400
    frame = pd.DataFrame(
        {
            "time_seconds": np.concatenate(
                [
                    np.linspace(34_200, 35_999, 200),
                    np.linspace(36_000, 37_799, 200),
                ]
            ),
            "actual": np.linspace(-1.0, 1.0, n_rows),
            "predicted": np.linspace(-0.8, 0.8, n_rows),
            "spread_bps": np.full(n_rows, 1.5),
            "mid_log_return": np.linspace(-0.0001, 0.0001, n_rows),
            "event_interarrival_us": np.full(n_rows, 500.0),
        }
    )

    bucketed = add_time_bucket(frame, bucket_minutes=30)
    assert bucketed["time_bucket"].nunique() == 2

    diagnostics = time_bucket_diagnostics(
        frame,
        actual_column="actual",
        predicted_column="predicted",
        bucket_minutes=30,
        minimum_observations=100,
    )
    assert len(diagnostics) == 2
    assert set(diagnostics["time_bucket"]) == {
        "09:30-10:00",
        "10:00-10:30",
    }
