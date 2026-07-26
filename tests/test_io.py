from orderbook_research.io import orderbook_columns


def test_orderbook_column_order():
    assert orderbook_columns(2) == [
        "ask_price_1",
        "ask_size_1",
        "bid_price_1",
        "bid_size_1",
        "ask_price_2",
        "ask_size_2",
        "bid_price_2",
        "bid_size_2",
    ]
