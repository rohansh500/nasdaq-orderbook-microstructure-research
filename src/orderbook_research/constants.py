MESSAGE_COLUMNS = [
    "time_seconds",
    "event_type",
    "order_id",
    "size",
    "price",
    "direction",
]

EVENT_LABELS = {
    1: "submission",
    2: "partial_cancel",
    3: "deletion",
    4: "visible_execution",
    5: "hidden_execution",
    6: "cross_trade",
    7: "trading_halt",
}

PRICE_SCALE = 10_000.0
DUMMY_ASK_PRICE = 9_999_999_999
DUMMY_BID_PRICE = -9_999_999_999
