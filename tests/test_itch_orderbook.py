from __future__ import annotations

from pathlib import Path

import pandas as pd

from orderbook_research.itch_binary import BinaryFileReader
from orderbook_research.itch_features import add_itch_snapshot_features
from orderbook_research.itch_fixture import write_synthetic_aapl_fixture
from orderbook_research.itch_orderbook import (
    FullDepthOrderBook,
    ItchSymbolReconstructor,
    Order,
)


def _reconstruct_fixture(tmp_path: Path) -> tuple[ItchSymbolReconstructor, pd.DataFrame]:
    path = write_synthetic_aapl_fixture(tmp_path / "fixture.itch")
    reconstructor = ItchSymbolReconstructor("AAPL", levels=3)
    events = []
    for record in BinaryFileReader(path):
        event = reconstructor.process(record)
        if event is not None:
            events.append(event)
    return reconstructor, pd.DataFrame(events)


def test_full_depth_order_lifecycle_and_aggregates() -> None:
    book = FullDepthOrderBook()
    first = Order(1, 1, "AAPL", "B", 100, 1_000_000)
    second = Order(2, 1, "AAPL", "B", 50, 1_000_000)
    book.add_order(first)
    book.add_order(second)

    assert book.bids.sizes[1_000_000] == 150
    book.reduce_order(1, 40)
    assert book.bids.sizes[1_000_000] == 110
    book.delete_order(2)
    assert book.bids.sizes[1_000_000] == 60
    assert book.validate()["valid"] is True


def test_replace_retains_side_and_changes_reference(tmp_path: Path) -> None:
    reconstructor, events = _reconstruct_fixture(tmp_path)
    replace = events.loc[events["message_type"] == "U"].iloc[0]

    assert replace["event_type"] == 8
    assert replace["old_order_id"] == 1_001
    assert replace["order_id"] == 1_002
    assert replace["direction"] == 1
    assert replace["old_size"] == 80
    assert replace["size"] == 60
    assert 1_001 not in reconstructor.book.orders
    assert 1_002 in reconstructor.book.orders


def test_fixture_reconstruction_produces_expected_final_book(tmp_path: Path) -> None:
    reconstructor, events = _reconstruct_fixture(tmp_path)
    final = events.iloc[-1]

    assert len(events) == 9
    assert final["bid_price_1"] == 100.0
    assert final["bid_size_1"] == 70
    assert final["ask_price_1"] == 100.02
    assert final["ask_size_1"] == 40
    assert reconstructor.book.validate()["valid"] is True
    assert reconstructor.book.stats.target_non_book_trades == 1


def test_itch_feature_adapter_counts_both_replace_legs(tmp_path: Path) -> None:
    _, events = _reconstruct_fixture(tmp_path)
    features = add_itch_snapshot_features(
        events,
        levels=3,
        rolling_windows=(4,),
    )
    replace = features.loc[features["message_type"] == "U"].iloc[0]

    assert replace["signed_add_event"] == 60
    assert replace["signed_cancel_event"] == -80
    assert "add_pressure_4" in features
    assert "cancel_pressure_4" in features
