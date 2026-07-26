from __future__ import annotations

import numpy as np
import pandas as pd


def non_overlapping_signal_simulation(
    frame: pd.DataFrame,
    horizon: int,
    confidence_threshold: float = 0.10,
    additional_fee_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {
        "mid_price",
        "spread_bps",
        f"future_mid_{horizon}",
        f"future_return_bps_{horizon}",
        "probability_down",
        "probability_up",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing simulation columns: {sorted(missing)}")

    sampled = frame.iloc[::horizon].copy()
    sampled["score"] = (
        sampled["probability_up"] - sampled["probability_down"]
    )
    sampled["signal"] = np.where(
        sampled["score"] > confidence_threshold,
        1,
        np.where(
            sampled["score"] < -confidence_threshold,
            -1,
            0,
        ),
    )

    future_spread_bps = sampled["spread_bps"].shift(-horizon)
    # The data is event-indexed. Shift on the full frame before sampling
    # to obtain the future spread at the same target horizon.
    full_future_spread = frame["spread_bps"].shift(-horizon)
    sampled["future_spread_bps"] = full_future_spread.loc[sampled.index]

    sampled["gross_return_bps"] = (
        sampled["signal"] * sampled[f"future_return_bps_{horizon}"]
    )
    sampled["estimated_cost_bps"] = (
        sampled["signal"].abs()
        * (
            0.5 * sampled["spread_bps"]
            + 0.5 * sampled["future_spread_bps"]
            + additional_fee_bps
        )
    )
    sampled["net_return_bps"] = (
        sampled["gross_return_bps"] - sampled["estimated_cost_bps"]
    )
    sampled["cumulative_net_bps"] = sampled["net_return_bps"].fillna(0).cumsum()
    peak = sampled["cumulative_net_bps"].cummax()
    sampled["drawdown_bps"] = sampled["cumulative_net_bps"] - peak

    active = sampled["signal"] != 0
    stats = {
        "observations": float(len(sampled)),
        "active_signal_fraction": float(active.mean()),
        "gross_return_bps": float(
            sampled["gross_return_bps"].sum(skipna=True)
        ),
        "net_return_bps": float(
            sampled["net_return_bps"].sum(skipna=True)
        ),
        "mean_net_return_active_bps": float(
            sampled.loc[active, "net_return_bps"].mean()
            if active.any()
            else 0.0
        ),
        "max_drawdown_bps": float(
            sampled["drawdown_bps"].min(skipna=True)
        ),
    }
    return sampled, stats
