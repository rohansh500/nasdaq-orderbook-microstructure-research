from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


BOOTSTRAP_BENCHMARKS: dict[str, float] = {
    "ridge_rank_ic": 0.0,
    "ridge_mae_improvement_bps": 0.0,
    "ridge_mae_improvement_pct": 0.0,
    "ridge_nonzero_directional_accuracy": 0.50,
    "mean_gross_return_active_bps": 0.0,
    "mean_estimated_cost_active_bps": 0.0,
    "mean_net_return_active_bps": 0.0,
    "break_even_cost_fraction": 1.0,
}


@dataclass(frozen=True)
class BootstrapResult:
    """Observed estimates, bootstrap intervals, and raw draws."""

    observed: dict[str, float]
    intervals: dict[str, dict[str, float | int]]
    draws: dict[str, np.ndarray]
    event_rows_used: int
    event_block_length: int
    simulation_rows_used: int
    simulation_block_length: int


def _overlapping_block_sums(
    values: np.ndarray,
    block_length: int,
) -> np.ndarray:
    """Return sums for every overlapping block of a one-dimensional array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if block_length < 1:
        raise ValueError("block_length must be at least one.")
    if len(array) < block_length:
        raise ValueError(
            "The sample must contain at least one complete bootstrap block."
        )

    cumulative = np.concatenate(([0.0], np.cumsum(array, dtype=float)))
    return cumulative[block_length:] - cumulative[:-block_length]


def _draw_block_totals(
    block_sums: dict[str, np.ndarray],
    n_blocks: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample overlapping blocks and add their sufficient statistics."""
    if not block_sums:
        raise ValueError("block_sums cannot be empty.")
    if n_blocks < 1:
        raise ValueError("n_blocks must be at least one.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least one.")

    lengths = {len(values) for values in block_sums.values()}
    if len(lengths) != 1:
        raise ValueError("All block-sum arrays must have equal length.")

    available_blocks = lengths.pop()
    sampled_starts = rng.integers(
        0,
        available_blocks,
        size=(n_bootstrap, n_blocks),
    )

    return {
        name: values[sampled_starts].sum(axis=1)
        for name, values in block_sums.items()
    }


def _safe_correlation_from_totals(
    count: int,
    x_sum: np.ndarray,
    y_sum: np.ndarray,
    x2_sum: np.ndarray,
    y2_sum: np.ndarray,
    xy_sum: np.ndarray,
) -> np.ndarray:
    numerator = count * xy_sum - x_sum * y_sum
    x_term = count * x2_sum - x_sum**2
    y_term = count * y2_sum - y_sum**2
    denominator = np.sqrt(np.maximum(x_term * y_term, 0.0))

    result = np.zeros_like(numerator, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _interval_summary(
    estimate: float,
    draws: np.ndarray,
    benchmark: float,
    confidence_level: float,
) -> dict[str, float | int]:
    clean = np.asarray(draws, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("No finite bootstrap draws were produced.")

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(clean, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "estimate": float(estimate),
        "bootstrap_mean": float(clean.mean()),
        "standard_error": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence_level),
        "benchmark": float(benchmark),
        "probability_above_benchmark": float(np.mean(clean > benchmark)),
        "bootstrap_draws": int(clean.size),
    }


def _event_bootstrap_draws(
    actual: np.ndarray,
    predicted: np.ndarray,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, np.ndarray], int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) < block_length:
        raise ValueError("Not enough event observations for the block length.")

    n_blocks = floor(len(actual) / block_length)
    rows_used = n_blocks * block_length
    actual = actual[:rows_used]
    predicted = predicted[:rows_used]

    actual_rank = rankdata(actual, method="average")
    predicted_rank = rankdata(predicted, method="average")
    zero_error = np.abs(actual)
    ridge_error = np.abs(actual - predicted)
    nonzero = actual != 0.0
    direction_correct = (
        nonzero & (np.sign(actual) == np.sign(predicted))
    ).astype(float)

    block_sums = {
        "rank_x": _overlapping_block_sums(actual_rank, block_length),
        "rank_y": _overlapping_block_sums(predicted_rank, block_length),
        "rank_x2": _overlapping_block_sums(actual_rank**2, block_length),
        "rank_y2": _overlapping_block_sums(predicted_rank**2, block_length),
        "rank_xy": _overlapping_block_sums(
            actual_rank * predicted_rank,
            block_length,
        ),
        "zero_error": _overlapping_block_sums(zero_error, block_length),
        "ridge_error": _overlapping_block_sums(ridge_error, block_length),
        "nonzero": _overlapping_block_sums(
            nonzero.astype(float),
            block_length,
        ),
        "direction_correct": _overlapping_block_sums(
            direction_correct,
            block_length,
        ),
    }
    totals = _draw_block_totals(
        block_sums,
        n_blocks=n_blocks,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    rank_ic_draws = _safe_correlation_from_totals(
        count=rows_used,
        x_sum=totals["rank_x"],
        y_sum=totals["rank_y"],
        x2_sum=totals["rank_x2"],
        y2_sum=totals["rank_y2"],
        xy_sum=totals["rank_xy"],
    )
    zero_mae_draws = totals["zero_error"] / rows_used
    ridge_mae_draws = totals["ridge_error"] / rows_used
    mae_improvement_draws = zero_mae_draws - ridge_mae_draws
    mae_improvement_pct_draws = np.divide(
        mae_improvement_draws,
        zero_mae_draws,
        out=np.zeros_like(mae_improvement_draws),
        where=zero_mae_draws > 0,
    ) * 100.0
    nonzero_direction_draws = np.divide(
        totals["direction_correct"],
        totals["nonzero"],
        out=np.zeros_like(totals["direction_correct"]),
        where=totals["nonzero"] > 0,
    )

    exact_rank_ic = spearmanr(actual, predicted).statistic
    exact_rank_ic = float(exact_rank_ic) if np.isfinite(exact_rank_ic) else 0.0
    zero_mae = float(zero_error.mean())
    ridge_mae = float(ridge_error.mean())
    mae_improvement = zero_mae - ridge_mae
    nonzero_count = int(nonzero.sum())

    observed = {
        "ridge_rank_ic": exact_rank_ic,
        "ridge_mae_improvement_bps": float(mae_improvement),
        "ridge_mae_improvement_pct": (
            float(mae_improvement / zero_mae * 100.0)
            if zero_mae > 0
            else 0.0
        ),
        "ridge_nonzero_directional_accuracy": (
            float(direction_correct.sum() / nonzero_count)
            if nonzero_count > 0
            else 0.0
        ),
    }
    draws = {
        "ridge_rank_ic": rank_ic_draws,
        "ridge_mae_improvement_bps": mae_improvement_draws,
        "ridge_mae_improvement_pct": mae_improvement_pct_draws,
        "ridge_nonzero_directional_accuracy": nonzero_direction_draws,
    }
    return observed, draws, rows_used


def _simulation_bootstrap_draws(
    simulation: pd.DataFrame,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, np.ndarray], int]:
    required = {
        "signal",
        "gross_return_bps",
        "estimated_cost_bps",
        "net_return_bps",
    }
    missing = required - set(simulation.columns)
    if missing:
        raise KeyError(f"Missing simulation columns: {sorted(missing)}")

    active = simulation["signal"].to_numpy() != 0
    gross = simulation["gross_return_bps"].to_numpy(dtype=float)
    cost = simulation["estimated_cost_bps"].to_numpy(dtype=float)
    net = simulation["net_return_bps"].to_numpy(dtype=float)
    valid = active & np.isfinite(gross) & np.isfinite(cost) & np.isfinite(net)

    gross = gross[valid]
    cost = cost[valid]
    net = net[valid]
    if len(gross) < block_length:
        raise ValueError("Not enough active simulation rows for the block length.")

    n_blocks = floor(len(gross) / block_length)
    rows_used = n_blocks * block_length
    gross = gross[:rows_used]
    cost = cost[:rows_used]
    net = net[:rows_used]

    block_sums = {
        "gross": _overlapping_block_sums(gross, block_length),
        "cost": _overlapping_block_sums(cost, block_length),
        "net": _overlapping_block_sums(net, block_length),
    }
    totals = _draw_block_totals(
        block_sums,
        n_blocks=n_blocks,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    mean_gross_draws = totals["gross"] / rows_used
    mean_cost_draws = totals["cost"] / rows_used
    mean_net_draws = totals["net"] / rows_used
    break_even_draws = np.divide(
        totals["gross"],
        totals["cost"],
        out=np.zeros_like(totals["gross"]),
        where=totals["cost"] > 0,
    )

    observed = {
        "mean_gross_return_active_bps": float(gross.mean()),
        "mean_estimated_cost_active_bps": float(cost.mean()),
        "mean_net_return_active_bps": float(net.mean()),
        "break_even_cost_fraction": (
            float(gross.sum() / cost.sum()) if cost.sum() > 0 else 0.0
        ),
    }
    draws = {
        "mean_gross_return_active_bps": mean_gross_draws,
        "mean_estimated_cost_active_bps": mean_cost_draws,
        "mean_net_return_active_bps": mean_net_draws,
        "break_even_cost_fraction": break_even_draws,
    }
    return observed, draws, rows_used


def bootstrap_fold_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    simulation: pd.DataFrame,
    horizon: int,
    n_bootstrap: int = 1_000,
    event_block_length: int = 1_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapResult:
    """Estimate block-bootstrap uncertainty for one validation fold.

    Event metrics use overlapping event blocks. Economic metrics use blocks of
    non-overlapping simulation observations with approximately the same event
    span. Rank IC is bootstrapped as Pearson correlation of fold-level ranks,
    avoiding repeated sorting while preserving serial blocks.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one.")
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    rng = np.random.default_rng(random_seed)
    event_observed, event_draws, event_rows_used = _event_bootstrap_draws(
        actual=actual,
        predicted=predicted,
        block_length=event_block_length,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    simulation_block_length = max(
        2,
        int(np.ceil(event_block_length / horizon)),
    )
    active_rows = int(
        (
            (simulation["signal"] != 0)
            & simulation["gross_return_bps"].notna()
            & simulation["estimated_cost_bps"].notna()
            & simulation["net_return_bps"].notna()
        ).sum()
    )
    if active_rows < simulation_block_length:
        simulation_block_length = max(1, active_rows)

    simulation_observed, simulation_draws, simulation_rows_used = (
        _simulation_bootstrap_draws(
            simulation=simulation,
            block_length=simulation_block_length,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    )

    observed = {**event_observed, **simulation_observed}
    draws = {**event_draws, **simulation_draws}
    intervals = {
        metric: _interval_summary(
            estimate=observed[metric],
            draws=metric_draws,
            benchmark=BOOTSTRAP_BENCHMARKS[metric],
            confidence_level=confidence_level,
        )
        for metric, metric_draws in draws.items()
    }

    return BootstrapResult(
        observed=observed,
        intervals=intervals,
        draws=draws,
        event_rows_used=event_rows_used,
        event_block_length=event_block_length,
        simulation_rows_used=simulation_rows_used,
        simulation_block_length=simulation_block_length,
    )


def summarize_draws(
    observed: dict[str, float],
    draws: dict[str, np.ndarray],
    confidence_level: float,
) -> dict[str, dict[str, float | int]]:
    """Summarize already aggregated bootstrap draws."""
    return {
        metric: _interval_summary(
            estimate=observed[metric],
            draws=metric_draws,
            benchmark=BOOTSTRAP_BENCHMARKS[metric],
            confidence_level=confidence_level,
        )
        for metric, metric_draws in draws.items()
    }
