import pandas as pd

from orderbook_research.reconstruction import reconstruction_audit


def test_price_level_reconstruction_matches_simple_sequence():
    frame = pd.DataFrame(
        [
            {
                "event_type": 1,
                "size": 100,
                "price": 10000,
                "direction": 1,
                "bid_price_1": 10000,
                "bid_size_1": 100,
                "ask_price_1": 10100,
                "ask_size_1": 200,
            },
            {
                "event_type": 1,
                "size": 50,
                "price": 10000,
                "direction": 1,
                "bid_price_1": 10000,
                "bid_size_1": 150,
                "ask_price_1": 10100,
                "ask_size_1": 200,
            },
            {
                "event_type": 4,
                "size": 50,
                "price": 10100,
                "direction": -1,
                "bid_price_1": 10000,
                "bid_size_1": 150,
                "ask_price_1": 10100,
                "ask_size_1": 150,
            },
        ]
    )

    result = reconstruction_audit(
        frame,
        levels=1,
        max_events=3,
    )

    assert result["top_price_match_fraction"] == 1.0
    assert result["top_state_match_fraction"] == 1.0
