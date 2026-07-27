from __future__ import annotations

import numpy as np
import pandas as pd

from orderbook_research.features import add_snapshot_features, safe_divide
from orderbook_research.itch_orderbook import ITCH_EVENT_REPLACE


def add_itch_snapshot_features(
    frame: pd.DataFrame,
    *,
    levels: int = 10,
    rolling_windows: tuple[int, ...] = (20, 50, 100),
) -> pd.DataFrame:
    """Create the existing research features while accounting for replaces.

    ITCH Order Replace is a single atomic message that removes the old order
    and inserts a new order.  The original LOBSTER feature pipeline does not
    have a replace event type, so the rolling add/cancel pressure series are
    recomputed using both legs without changing the established feature set.
    """
    df = add_snapshot_features(
        frame,
        levels=levels,
        rolling_windows=rolling_windows,
    )

    replace = df["event_type"] == ITCH_EVENT_REPLACE
    old_size = pd.to_numeric(
        df.get("old_size", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)

    df["signed_add_event"] = np.where(
        df["event_type"].isin([1, ITCH_EVENT_REPLACE]),
        df["direction"] * df["size"],
        0.0,
    )
    df["signed_cancel_event"] = np.where(
        df["event_type"].isin([2, 3]),
        -df["direction"] * df["size"],
        np.where(replace, -df["direction"] * old_size, 0.0),
    )

    normaliser = df[f"total_depth_{levels}"]
    for window in rolling_windows:
        minimum = max(2, window // 4)
        df[f"add_pressure_{window}"] = safe_divide(
            df["signed_add_event"].rolling(window, min_periods=minimum).sum(),
            normaliser,
        )
        df[f"cancel_pressure_{window}"] = safe_divide(
            df["signed_cancel_event"].rolling(window, min_periods=minimum).sum(),
            normaliser,
        )

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_columns] = df.loc[:, numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return df
