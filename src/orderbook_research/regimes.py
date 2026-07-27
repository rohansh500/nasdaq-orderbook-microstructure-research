from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_COST_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class TertileCutpoints:
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"low": float(self.low), "high": float(self.high)}


def fit_tertile_cutpoints(values: pd.Series | np.ndarray) -> TertileCutpoints:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot fit regime cutpoints without finite values.")

    low, high = np.quantile(array, [1.0 / 3.0, 2.0 / 3.0])
    return TertileCutpoints(low=float(low), high=float(high))


def assign_tertile_regime(
    values: pd.Series | np.ndarray,
    cutpoints: TertileCutpoints,
) -> pd.Series:
    series = pd.Series(values, copy=False, dtype=float)
    labels = np.select(
        [
            series <= cutpoints.low,
            (series > cutpoints.low) & (series <= cutpoints.high),
            series > cutpoints.high,
        ],
        ["low", "medium", "high"],
        default="unknown",
    )
    labels = np.where(np.isfinite(series.to_numpy()), labels, "unknown")
    return pd.Series(labels, index=series.index, dtype="object")


def assign_spread_regime(
    spread: pd.Series | np.ndarray,
    tick_size: float = 0.01,
) -> pd.Series:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive.")

    series = pd.Series(spread, copy=False, dtype=float)
    spread_ticks = np.rint(series / tick_size)
    labels = np.select(
        [
            spread_ticks <= 1,
            spread_ticks == 2,
            spread_ticks >= 3,
        ],
        ["one_tick", "two_ticks", "three_plus_ticks"],
        default="unknown",
    )
    valid = np.isfinite(series.to_numpy()) & (series.to_numpy() > 0)
    labels = np.where(valid, labels, "unknown")
    return pd.Series(labels, index=series.index, dtype="object")


def add_validation_regimes(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    levels: int = 10,
    volatility_window: int = 50,
    tick_size: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, object]]:
    depth_column = f"total_depth_{levels}"
    volatility_column = f"rolling_volatility_{volatility_window}"

    missing = {
        column
        for column in ("spread", depth_column, volatility_column)
        if column not in train.columns or column not in validation.columns
    }
    if missing:
        raise KeyError(f"Missing regime columns: {sorted(missing)}")

    depth_cutpoints = fit_tertile_cutpoints(train[depth_column])
    volatility_cutpoints = fit_tertile_cutpoints(train[volatility_column])

    result = validation.copy()
    result["spread_regime"] = assign_spread_regime(
        result["spread"],
        tick_size=tick_size,
    )
    result["depth_regime"] = assign_tertile_regime(
        result[depth_column],
        depth_cutpoints,
    )
    result["volatility_regime"] = assign_tertile_regime(
        result[volatility_column],
        volatility_cutpoints,
    )

    metadata: dict[str, object] = {
        "tick_size": float(tick_size),
        "depth_column": depth_column,
        "depth_cutpoints": depth_cutpoints.as_dict(),
        "volatility_column": volatility_column,
        "volatility_cutpoints": volatility_cutpoints.as_dict(),
    }
    return result, metadata


def prepare_non_overlapping_signals(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    required = {
        "spread_bps",
        "probability_down",
        "probability_up",
        f"future_return_bps_{horizon}",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing signal columns: {sorted(missing)}")

    full = frame.copy()
    full["future_spread_bps"] = full["spread_bps"].shift(-horizon)
    sampled = full.iloc[::horizon].copy()
    sampled["score"] = sampled["probability_up"] - sampled["probability_down"]
    sampled["absolute_score"] = sampled["score"].abs()
    sampled["quoted_round_trip_cost_bps"] = (
        0.5 * sampled["spread_bps"] + 0.5 * sampled["future_spread_bps"]
    )
    return sampled


def economic_metrics(
    sampled: pd.DataFrame,
    horizon: int,
    confidence_threshold: float,
    cost_fraction: float,
    additional_fee_bps: float = 0.0,
) -> dict[str, float | int]:
    if confidence_threshold < 0:
        raise ValueError("confidence_threshold cannot be negative.")
    if not 0 <= cost_fraction <= 1:
        raise ValueError("cost_fraction must be between zero and one.")

    target = f"future_return_bps_{horizon}"
    required = {
        target,
        "score",
        "quoted_round_trip_cost_bps",
    }
    missing = required - set(sampled.columns)
    if missing:
        raise KeyError(f"Missing economic columns: {sorted(missing)}")

    frame = sampled.copy()
    finite_mask = (
        np.isfinite(frame[target].to_numpy(dtype=float))
        & np.isfinite(frame["score"].to_numpy(dtype=float))
        & np.isfinite(frame["quoted_round_trip_cost_bps"].to_numpy(dtype=float))
    )
    frame = frame.loc[finite_mask].copy()
    frame["signal"] = np.where(
        frame["score"] > confidence_threshold,
        1,
        np.where(frame["score"] < -confidence_threshold, -1, 0),
    )
    frame["gross_return_bps"] = frame["signal"] * frame[target]
    frame["full_estimated_cost_bps"] = frame["signal"].abs() * (
        frame["quoted_round_trip_cost_bps"] + additional_fee_bps
    )
    frame["applied_cost_bps"] = cost_fraction * frame["full_estimated_cost_bps"]
    frame["net_return_bps"] = frame["gross_return_bps"] - frame["applied_cost_bps"]
    frame["cumulative_net_bps"] = frame["net_return_bps"].fillna(0).cumsum()
    frame["drawdown_bps"] = frame["cumulative_net_bps"] - frame["cumulative_net_bps"].cummax()

    active = frame["signal"] != 0
    active_count = int(active.sum())
    gross_total = float(frame["gross_return_bps"].sum(skipna=True))
    full_cost_total = float(frame["full_estimated_cost_bps"].sum(skipna=True))
    applied_cost_total = float(frame["applied_cost_bps"].sum(skipna=True))
    net_total = float(frame["net_return_bps"].sum(skipna=True))

    return {
        "observations": int(len(frame)),
        "active_signals": active_count,
        "active_signal_fraction": float(active.mean()) if len(frame) else 0.0,
        "gross_return_bps": gross_total,
        "full_estimated_cost_bps": full_cost_total,
        "applied_cost_bps": applied_cost_total,
        "net_return_bps": net_total,
        "mean_gross_return_active_bps": (
            float(frame.loc[active, "gross_return_bps"].mean()) if active_count else 0.0
        ),
        "mean_full_estimated_cost_active_bps": (
            float(frame.loc[active, "full_estimated_cost_bps"].mean()) if active_count else 0.0
        ),
        "mean_applied_cost_active_bps": (
            float(frame.loc[active, "applied_cost_bps"].mean()) if active_count else 0.0
        ),
        "mean_net_return_active_bps": (
            float(frame.loc[active, "net_return_bps"].mean()) if active_count else 0.0
        ),
        "active_hit_rate": (
            float((frame.loc[active, "gross_return_bps"] > 0).mean()) if active_count else 0.0
        ),
        "mean_absolute_score_active": (
            float(frame.loc[active, "absolute_score"].mean())
            if active_count and "absolute_score" in frame.columns
            else 0.0
        ),
        "break_even_cost_fraction": (
            float(gross_total / full_cost_total) if full_cost_total > 0 else 0.0
        ),
        "max_drawdown_bps": (float(frame["drawdown_bps"].min(skipna=True)) if len(frame) else 0.0),
    }


def validate_grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    if any(not np.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values.")
    return result
