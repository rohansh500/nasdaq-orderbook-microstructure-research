import numpy as np
import pandas as pd

from orderbook_research.features import add_snapshot_features


def test_mid_spread_and_queue_imbalance():
    frame = pd.DataFrame(
        {
            "time_seconds": [34200.0, 34200.1],
            "event_type": [1, 4],
            "size": [100, 50],
            "direction": [1, -1],
            "ask_price_1": [100.02, 100.03],
            "ask_size_1": [300, 200],
            "bid_price_1": [100.00, 100.01],
            "bid_size_1": [100, 400],
        }
    )
    for level in range(2, 11):
        frame[f"ask_price_{level}"] = 100.02 + level * 0.01
        frame[f"ask_size_{level}"] = 100
        frame[f"bid_price_{level}"] = 100.00 - level * 0.01
        frame[f"bid_size_{level}"] = 100

    result = add_snapshot_features(
        frame,
        levels=10,
        rolling_windows=(2,),
    )

    assert np.isclose(result.loc[0, "mid_price"], 100.01)
    assert np.isclose(result.loc[0, "spread"], 0.02)
    assert np.isclose(result.loc[0, "queue_imbalance_l1"], -0.5)
