from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orderbook_research.regimes import (
    TertileCutpoints,
    add_validation_regimes,
    assign_spread_regime,
    assign_tertile_regime,
    economic_metrics,
    fit_tertile_cutpoints,
    prepare_non_overlapping_signals,
)


def test_spread_regimes_use_tick_counts() -> None:
    spread = pd.Series([0.01, 0.02, 0.03, 0.05, np.nan])
    labels = assign_spread_regime(spread, tick_size=0.01)
    assert labels.tolist() == [
        "one_tick",
        "two_ticks",
        "three_plus_ticks",
        "three_plus_ticks",
        "unknown",
    ]


def test_tertile_cutpoints_are_fit_from_training_values() -> None:
    train = pd.Series(np.arange(1.0, 10.0))
    cutpoints = fit_tertile_cutpoints(train)
    validation = pd.Series([1.0, 5.0, 9.0])
    labels = assign_tertile_regime(validation, cutpoints)
    assert labels.tolist() == ["low", "medium", "high"]


def test_validation_regimes_reuse_training_cutpoints() -> None:
    train = pd.DataFrame(
        {
            "spread": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
            "total_depth_10": [10, 20, 30, 40, 50, 60],
            "rolling_volatility_50": [1, 2, 3, 4, 5, 6],
        }
    )
    validation = pd.DataFrame(
        {
            "spread": [0.01, 0.02, 0.04],
            "total_depth_10": [5, 35, 100],
            "rolling_volatility_50": [0.5, 3.5, 8.0],
        }
    )
    result, metadata = add_validation_regimes(train, validation, levels=10)
    assert result["spread_regime"].tolist() == [
        "one_tick",
        "two_ticks",
        "three_plus_ticks",
    ]
    assert result["depth_regime"].tolist() == ["low", "medium", "high"]
    assert result["volatility_regime"].tolist() == [
        "low",
        "medium",
        "high",
    ]
    assert metadata["depth_column"] == "total_depth_10"


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spread_bps": np.ones(20),
            "probability_down": np.full(20, 0.20),
            "probability_up": np.linspace(0.20, 0.80, 20),
            "future_return_bps_5": np.linspace(-1.0, 1.0, 20),
            "spread_regime": ["one_tick"] * 20,
        }
    )


def test_non_overlapping_sampling_uses_horizon_stride() -> None:
    sampled = prepare_non_overlapping_signals(_sample_frame(), horizon=5)
    assert sampled.index.tolist() == [0, 5, 10, 15]
    assert "quoted_round_trip_cost_bps" in sampled.columns


def test_higher_threshold_does_not_increase_active_fraction() -> None:
    sampled = prepare_non_overlapping_signals(_sample_frame(), horizon=5)
    low = economic_metrics(sampled, 5, confidence_threshold=0.05, cost_fraction=1.0)
    high = economic_metrics(sampled, 5, confidence_threshold=0.40, cost_fraction=1.0)
    assert high["active_signal_fraction"] <= low["active_signal_fraction"]


def test_cost_fraction_zero_and_one_are_consistent() -> None:
    sampled = prepare_non_overlapping_signals(_sample_frame(), horizon=5)
    zero = economic_metrics(sampled, 5, confidence_threshold=0.05, cost_fraction=0.0)
    full = economic_metrics(sampled, 5, confidence_threshold=0.05, cost_fraction=1.0)
    assert zero["net_return_bps"] == pytest.approx(zero["gross_return_bps"])
    assert full["net_return_bps"] == pytest.approx(
        full["gross_return_bps"] - full["full_estimated_cost_bps"]
    )


def test_invalid_cutpoints_are_not_required_to_be_distinct() -> None:
    labels = assign_tertile_regime(
        pd.Series([0.0, 1.0, 2.0]),
        TertileCutpoints(low=1.0, high=1.0),
    )
    assert labels.tolist() == ["low", "low", "high"]
