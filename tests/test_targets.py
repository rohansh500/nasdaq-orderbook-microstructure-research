import pandas as pd

from orderbook_research.targets import add_event_horizon_targets


def test_event_horizon_target_uses_future_mid():
    frame = pd.DataFrame(
        {
            "mid_price": [100.0, 100.0, 101.0, 99.0],
        }
    )
    result = add_event_horizon_targets(frame, horizons=(2,))

    assert result.loc[0, "future_move_2"] == 1
    assert result.loc[1, "future_move_2"] == -1
    assert pd.isna(result.loc[2, "future_move_2"])
