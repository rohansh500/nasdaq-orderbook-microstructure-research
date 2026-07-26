from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PriceLevelBook:
    bids: dict[int, int] = field(default_factory=dict)
    asks: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: pd.Series,
        levels: int,
    ) -> "PriceLevelBook":
        bids: dict[int, int] = {}
        asks: dict[int, int] = {}

        for level in range(1, levels + 1):
            bid_price = snapshot.get(f"bid_price_{level}")
            bid_size = snapshot.get(f"bid_size_{level}")
            ask_price = snapshot.get(f"ask_price_{level}")
            ask_size = snapshot.get(f"ask_size_{level}")

            if pd.notna(bid_price) and pd.notna(bid_size) and bid_size > 0:
                bids[int(bid_price)] = int(bid_size)
            if pd.notna(ask_price) and pd.notna(ask_size) and ask_size > 0:
                asks[int(ask_price)] = int(ask_size)

        return cls(bids=bids, asks=asks)

    def apply_event(
        self,
        event_type: int,
        size: int,
        price: int,
        direction: int,
    ) -> None:
        if event_type in {5, 6, 7}:
            return
        if direction not in {-1, 1}:
            return

        side = self.bids if direction == 1 else self.asks
        price = int(price)
        size = int(size)

        if event_type == 1:
            side[price] = side.get(price, 0) + size
            return

        if event_type in {2, 3, 4}:
            remaining = side.get(price, 0) - size
            if remaining > 0:
                side[price] = remaining
            else:
                side.pop(price, None)

    def best_bid(self) -> tuple[int | None, int | None]:
        if not self.bids:
            return None, None
        price = max(self.bids)
        return price, self.bids[price]

    def best_ask(self) -> tuple[int | None, int | None]:
        if not self.asks:
            return None, None
        price = min(self.asks)
        return price, self.asks[price]


def reconstruction_audit(
    raw_frame: pd.DataFrame,
    levels: int,
    max_events: int = 10_000,
) -> dict[str, float | int | str | None]:
    """Validate each message-to-snapshot transition independently.

    The book is reseeded from the previous supplied snapshot for every event.
    This prevents one limited-depth mismatch from cascading through all later
    observations.
    """
    if len(raw_frame) < 2:
        raise ValueError("At least two aligned rows are required.")

    limit = min(len(raw_frame), max_events)

    exact_top_price_matches = 0
    exact_top_state_matches = 0
    checked = 0
    first_price_mismatch: int | None = None

    for position in range(1, limit):
        previous_snapshot = raw_frame.iloc[position - 1]
        current_row = raw_frame.iloc[position]

        # Independently reconstruct only this one transition.
        book = PriceLevelBook.from_snapshot(
            previous_snapshot,
            levels,
        )

        book.apply_event(
            int(current_row["event_type"]),
            int(current_row["size"]),
            int(current_row["price"]),
            int(current_row["direction"]),
        )

        reconstructed_bid_price, reconstructed_bid_size = (
            book.best_bid()
        )
        reconstructed_ask_price, reconstructed_ask_size = (
            book.best_ask()
        )

        actual_bid_price = (
            None
            if pd.isna(current_row["bid_price_1"])
            else int(current_row["bid_price_1"])
        )
        actual_ask_price = (
            None
            if pd.isna(current_row["ask_price_1"])
            else int(current_row["ask_price_1"])
        )
        actual_bid_size = (
            None
            if pd.isna(current_row["bid_size_1"])
            else int(current_row["bid_size_1"])
        )
        actual_ask_size = (
            None
            if pd.isna(current_row["ask_size_1"])
            else int(current_row["ask_size_1"])
        )

        price_match = (
            reconstructed_bid_price == actual_bid_price
            and reconstructed_ask_price == actual_ask_price
        )

        state_match = (
            price_match
            and reconstructed_bid_size == actual_bid_size
            and reconstructed_ask_size == actual_ask_size
        )

        exact_top_price_matches += int(price_match)
        exact_top_state_matches += int(state_match)
        checked += 1

        if not price_match and first_price_mismatch is None:
            first_price_mismatch = position

    return {
        "audit_method": "one_step_reseeded_transition",
        "events_checked": checked,
        "top_price_match_fraction": (
            exact_top_price_matches / checked
            if checked
            else 0.0
        ),
        "top_state_match_fraction": (
            exact_top_state_matches / checked
            if checked
            else 0.0
        ),
        "first_top_price_mismatch_event": first_price_mismatch,
    }