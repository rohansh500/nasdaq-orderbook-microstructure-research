import numpy as np
import pandas as pd
import pytest

from orderbook_research.bootstrap import bootstrap_fold_metrics


def _synthetic_simulation(n: int = 400) -> pd.DataFrame:
    signal = np.where(np.arange(n) % 3 == 0, 0, 1)
    gross = np.where(signal != 0, 0.30, 0.0)
    cost = np.where(signal != 0, 1.50, 0.0)
    return pd.DataFrame(
        {
            "signal": signal,
            "gross_return_bps": gross,
            "estimated_cost_bps": cost,
            "net_return_bps": gross - cost,
        }
    )


def test_bootstrap_is_reproducible_and_returns_intervals():
    rng = np.random.default_rng(7)
    actual = rng.normal(size=5_000)
    predicted = 0.35 * actual + rng.normal(scale=0.8, size=5_000)
    simulation = _synthetic_simulation()

    first = bootstrap_fold_metrics(
        actual,
        predicted,
        simulation,
        horizon=10,
        n_bootstrap=200,
        event_block_length=100,
        random_seed=11,
    )
    second = bootstrap_fold_metrics(
        actual,
        predicted,
        simulation,
        horizon=10,
        n_bootstrap=200,
        event_block_length=100,
        random_seed=11,
    )

    assert first.intervals == second.intervals
    assert first.intervals["ridge_rank_ic"]["ci_lower"] > 0
    assert (
        first.intervals["ridge_nonzero_directional_accuracy"]
        ["probability_above_benchmark"]
        > 0.90
    )


def test_economic_bootstrap_detects_cost_failure():
    rng = np.random.default_rng(12)
    actual = rng.normal(size=4_000)
    predicted = actual + rng.normal(scale=0.2, size=4_000)
    result = bootstrap_fold_metrics(
        actual,
        predicted,
        _synthetic_simulation(),
        horizon=10,
        n_bootstrap=200,
        event_block_length=100,
        random_seed=9,
    )

    assert result.intervals["mean_gross_return_active_bps"]["ci_lower"] > 0
    assert result.intervals["mean_net_return_active_bps"]["ci_upper"] < 0
    assert result.intervals["break_even_cost_fraction"]["ci_upper"] < 1


def test_bootstrap_rejects_too_few_draws():
    with pytest.raises(ValueError):
        bootstrap_fold_metrics(
            np.arange(1_000, dtype=float),
            np.arange(1_000, dtype=float),
            _synthetic_simulation(),
            horizon=10,
            n_bootstrap=50,
            event_block_length=100,
        )
