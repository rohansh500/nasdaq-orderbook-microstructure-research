from __future__ import annotations

import numpy as np
import pandas as pd


def add_event_horizon_targets(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (10, 50, 100),
) -> pd.DataFrame:
    df = frame.copy()

    for horizon in horizons:
        future_mid = df["mid_price"].shift(-horizon)
        future_return_bps = 10_000.0 * (future_mid / df["mid_price"] - 1.0)
        future_move = np.sign(future_mid - df["mid_price"])

        df[f"future_mid_{horizon}"] = future_mid
        df[f"future_return_bps_{horizon}"] = future_return_bps
        df[f"future_move_{horizon}"] = pd.Series(
            future_move,
            index=df.index,
        ).astype("Int8")

    return df
