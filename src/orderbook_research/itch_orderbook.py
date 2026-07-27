from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from orderbook_research.constants import PRICE_SCALE
from orderbook_research.itch_binary import (
    BinaryRecord,
    decode_alpha,
    stock_locate,
    timestamp_ns,
)

ITCH_EVENT_REPLACE = 8


@dataclass(frozen=True)
class Order:
    order_id: int
    stock_locate: int
    stock: str
    side: str
    shares: int
    price: int
    attribution: str | None = None

    @property
    def direction(self) -> int:
        return 1 if self.side == "B" else -1


@dataclass
class ReconstructionStats:
    target_book_events: int = 0
    target_non_book_trades: int = 0
    adds: int = 0
    executions: int = 0
    cancels: int = 0
    deletes: int = 0
    replaces: int = 0
    duplicate_order_references: int = 0
    missing_target_order_references: int = 0
    share_underflows: int = 0
    timestamp_monotonicity_violations: int = 0
    crossed_book_rows: int = 0
    locked_book_rows: int = 0
    max_open_orders: int = 0
    final_open_orders: int = 0


@dataclass
class SortedPriceLevels:
    sizes: dict[int, int] = field(default_factory=dict)
    prices: list[int] = field(default_factory=list)

    def add(self, price: int, shares: int) -> None:
        if shares <= 0:
            raise ValueError("Price-level additions must be positive.")
        if price not in self.sizes:
            bisect.insort(self.prices, price)
            self.sizes[price] = 0
        self.sizes[price] += shares

    def remove(self, price: int, shares: int) -> None:
        if shares <= 0:
            raise ValueError("Price-level reductions must be positive.")
        current = self.sizes.get(price)
        if current is None:
            raise KeyError(f"Unknown price level: {price}")
        remaining = current - shares
        if remaining < 0:
            raise ValueError(f"Price-level underflow at {price}: {current} - {shares}.")
        if remaining == 0:
            del self.sizes[price]
            position = bisect.bisect_left(self.prices, price)
            if position >= len(self.prices) or self.prices[position] != price:
                raise AssertionError("Price index and size map are inconsistent.")
            self.prices.pop(position)
        else:
            self.sizes[price] = remaining

    def best(self, *, descending: bool) -> tuple[int | None, int | None]:
        if not self.prices:
            return None, None
        price = self.prices[-1] if descending else self.prices[0]
        return price, self.sizes[price]

    def top(self, levels: int, *, descending: bool) -> list[tuple[int, int]]:
        selected = self.prices[-levels:][::-1] if descending else self.prices[:levels]
        return [(price, self.sizes[price]) for price in selected]


@dataclass
class FullDepthOrderBook:
    strict: bool = True
    orders: dict[int, Order] = field(default_factory=dict)
    bids: SortedPriceLevels = field(default_factory=SortedPriceLevels)
    asks: SortedPriceLevels = field(default_factory=SortedPriceLevels)
    stats: ReconstructionStats = field(default_factory=ReconstructionStats)

    def _levels(self, side: str) -> SortedPriceLevels:
        if side == "B":
            return self.bids
        if side == "S":
            return self.asks
        raise ValueError(f"Unknown ITCH side: {side!r}")

    def add_order(self, order: Order) -> None:
        if order.order_id in self.orders:
            self.stats.duplicate_order_references += 1
            if self.strict:
                raise ValueError(f"Duplicate order reference: {order.order_id}")
            self.delete_order(order.order_id)
        if order.shares <= 0:
            raise ValueError("Order shares must be positive.")
        self.orders[order.order_id] = order
        self._levels(order.side).add(order.price, order.shares)
        self.stats.max_open_orders = max(
            self.stats.max_open_orders,
            len(self.orders),
        )

    def _missing_order(self, order_id: int) -> None:
        self.stats.missing_target_order_references += 1
        if self.strict:
            raise KeyError(f"Unknown target order reference: {order_id}")

    def reduce_order(self, order_id: int, shares: int) -> Order | None:
        order = self.orders.get(order_id)
        if order is None:
            self._missing_order(order_id)
            return None
        if shares <= 0:
            raise ValueError("Order reductions must be positive.")

        reduction = shares
        if shares > order.shares:
            self.stats.share_underflows += 1
            if self.strict:
                raise ValueError(f"Order underflow for {order_id}: {order.shares} - {shares}.")
            reduction = order.shares

        self._levels(order.side).remove(order.price, reduction)
        remaining = order.shares - reduction
        if remaining == 0:
            del self.orders[order_id]
        else:
            self.orders[order_id] = Order(
                order_id=order.order_id,
                stock_locate=order.stock_locate,
                stock=order.stock,
                side=order.side,
                shares=remaining,
                price=order.price,
                attribution=order.attribution,
            )
        return order

    def delete_order(self, order_id: int) -> Order | None:
        order = self.orders.get(order_id)
        if order is None:
            self._missing_order(order_id)
            return None
        self._levels(order.side).remove(order.price, order.shares)
        del self.orders[order_id]
        return order

    def replace_order(
        self,
        original_order_id: int,
        new_order_id: int,
        shares: int,
        price: int,
    ) -> tuple[Order | None, Order | None]:
        original = self.orders.get(original_order_id)
        if original is None:
            self._missing_order(original_order_id)
            return None, None
        self.delete_order(original_order_id)
        replacement = Order(
            order_id=new_order_id,
            stock_locate=original.stock_locate,
            stock=original.stock,
            side=original.side,
            shares=shares,
            price=price,
            attribution=original.attribution,
        )
        self.add_order(replacement)
        return original, replacement

    def snapshot(self, levels: int) -> dict[str, float | int | None]:
        row: dict[str, float | int | None] = {}
        asks = self.asks.top(levels, descending=False)
        bids = self.bids.top(levels, descending=True)

        for level in range(1, levels + 1):
            ask = asks[level - 1] if level <= len(asks) else None
            bid = bids[level - 1] if level <= len(bids) else None
            row[f"ask_price_{level}"] = ask[0] / PRICE_SCALE if ask is not None else None
            row[f"ask_size_{level}"] = ask[1] if ask is not None else None
            row[f"bid_price_{level}"] = bid[0] / PRICE_SCALE if bid is not None else None
            row[f"bid_size_{level}"] = bid[1] if bid is not None else None
        return row

    def validate(self) -> dict[str, Any]:
        bid_from_orders: Counter[int] = Counter()
        ask_from_orders: Counter[int] = Counter()
        nonpositive_orders = 0

        for order in self.orders.values():
            if order.shares <= 0:
                nonpositive_orders += 1
            target = bid_from_orders if order.side == "B" else ask_from_orders
            target[order.price] += order.shares

        bid_match = dict(bid_from_orders) == self.bids.sizes
        ask_match = dict(ask_from_orders) == self.asks.sizes
        sorted_bid_prices = self.bids.prices == sorted(self.bids.prices)
        sorted_ask_prices = self.asks.prices == sorted(self.asks.prices)

        return {
            "order_count": len(self.orders),
            "bid_level_count": len(self.bids.prices),
            "ask_level_count": len(self.asks.prices),
            "bid_aggregate_matches_orders": bid_match,
            "ask_aggregate_matches_orders": ask_match,
            "bid_price_index_sorted": sorted_bid_prices,
            "ask_price_index_sorted": sorted_ask_prices,
            "nonpositive_orders": nonpositive_orders,
            "valid": bool(
                bid_match
                and ask_match
                and sorted_bid_prices
                and sorted_ask_prices
                and nonpositive_orders == 0
            ),
        }


class ItchSymbolReconstructor:
    def __init__(
        self,
        symbol: str,
        *,
        levels: int = 10,
        strict: bool = True,
    ) -> None:
        self.symbol = symbol.upper()
        self.levels = levels
        self.strict = strict
        self.book = FullDepthOrderBook(strict=strict)
        self.message_counts: Counter[str] = Counter()
        self.stock_directory: dict[int, str] = {}
        self.target_locates: set[int] = set()
        self.last_target_timestamp_ns: int | None = None
        self.event_index = 0

    def _is_target_locate(self, locate: int) -> bool:
        return locate in self.target_locates

    def _register_symbol(self, locate: int, stock: str) -> None:
        self.stock_directory[locate] = stock
        if stock == self.symbol:
            self.target_locates.add(locate)

    def _record_timestamp(self, value: int) -> None:
        if self.last_target_timestamp_ns is not None and value < self.last_target_timestamp_ns:
            self.book.stats.timestamp_monotonicity_violations += 1
        self.last_target_timestamp_ns = value

    def _event_row(
        self,
        *,
        record: BinaryRecord,
        event_type: int,
        order: Order,
        size: int,
        price: int,
        old_order_id: int | None = None,
        old_size: int | None = None,
        old_price: int | None = None,
        execution_price: int | None = None,
    ) -> dict[str, Any]:
        timestamp = timestamp_ns(record.payload)
        self._record_timestamp(timestamp)
        self.event_index += 1
        self.book.stats.target_book_events += 1

        best_bid_price, _ = self.book.bids.best(descending=True)
        best_ask_price, _ = self.book.asks.best(descending=False)
        if best_bid_price is not None and best_ask_price is not None:
            if best_bid_price > best_ask_price:
                self.book.stats.crossed_book_rows += 1
            elif best_bid_price == best_ask_price:
                self.book.stats.locked_book_rows += 1

        row: dict[str, Any] = {
            "event_index": self.event_index - 1,
            "source_message_sequence": record.sequence,
            "source_file_offset": record.file_offset,
            "timestamp_ns": timestamp,
            "time_seconds": timestamp / 1_000_000_000.0,
            "message_type": record.message_type,
            "event_type": event_type,
            "order_id": order.order_id,
            "old_order_id": old_order_id,
            "size": size,
            "old_size": old_size,
            "price": price / PRICE_SCALE,
            "old_price": old_price / PRICE_SCALE if old_price is not None else None,
            "execution_price": (
                execution_price / PRICE_SCALE if execution_price is not None else None
            ),
            "direction": order.direction,
            "stock_locate": order.stock_locate,
            "stock": order.stock,
        }
        row.update(self.book.snapshot(self.levels))
        return row

    def process(self, record: BinaryRecord) -> dict[str, Any] | None:
        payload = record.payload
        kind = record.message_type
        self.message_counts[kind] += 1

        if kind == "R":
            locate = stock_locate(payload)
            stock = decode_alpha(payload[11:19]).upper()
            self._register_symbol(locate, stock)
            return None

        if kind in {"A", "F"}:
            locate = stock_locate(payload)
            order_id = int.from_bytes(payload[11:19], "big")
            side = chr(payload[19])
            shares = int.from_bytes(payload[20:24], "big")
            stock = decode_alpha(payload[24:32]).upper()
            price = int.from_bytes(payload[32:36], "big")
            attribution = decode_alpha(payload[36:40]) if kind == "F" else None
            self._register_symbol(locate, stock)
            if stock != self.symbol:
                return None

            order = Order(
                order_id=order_id,
                stock_locate=locate,
                stock=stock,
                side=side,
                shares=shares,
                price=price,
                attribution=attribution,
            )
            self.book.add_order(order)
            self.book.stats.adds += 1
            return self._event_row(
                record=record,
                event_type=1,
                order=order,
                size=shares,
                price=price,
            )

        locate = stock_locate(payload) if len(payload) >= 3 else 0

        if kind == "P":
            stock = decode_alpha(payload[24:32]).upper()
            if stock == self.symbol:
                self.book.stats.target_non_book_trades += 1
            return None

        if kind == "E":
            order_id = int.from_bytes(payload[11:19], "big")
            if order_id not in self.book.orders:
                if self._is_target_locate(locate):
                    self.book._missing_order(order_id)
                return None
            shares = int.from_bytes(payload[19:23], "big")
            order = self.book.reduce_order(order_id, shares)
            if order is None:
                return None
            self.book.stats.executions += 1
            return self._event_row(
                record=record,
                event_type=4,
                order=order,
                size=shares,
                price=order.price,
            )

        if kind == "C":
            order_id = int.from_bytes(payload[11:19], "big")
            if order_id not in self.book.orders:
                if self._is_target_locate(locate):
                    self.book._missing_order(order_id)
                return None
            shares = int.from_bytes(payload[19:23], "big")
            execution_price = int.from_bytes(payload[32:36], "big")
            order = self.book.reduce_order(order_id, shares)
            if order is None:
                return None
            self.book.stats.executions += 1
            return self._event_row(
                record=record,
                event_type=4,
                order=order,
                size=shares,
                price=order.price,
                execution_price=execution_price,
            )

        if kind == "X":
            order_id = int.from_bytes(payload[11:19], "big")
            if order_id not in self.book.orders:
                if self._is_target_locate(locate):
                    self.book._missing_order(order_id)
                return None
            shares = int.from_bytes(payload[19:23], "big")
            order = self.book.reduce_order(order_id, shares)
            if order is None:
                return None
            self.book.stats.cancels += 1
            return self._event_row(
                record=record,
                event_type=2,
                order=order,
                size=shares,
                price=order.price,
            )

        if kind == "D":
            order_id = int.from_bytes(payload[11:19], "big")
            if order_id not in self.book.orders:
                if self._is_target_locate(locate):
                    self.book._missing_order(order_id)
                return None
            order = self.book.delete_order(order_id)
            if order is None:
                return None
            self.book.stats.deletes += 1
            return self._event_row(
                record=record,
                event_type=3,
                order=order,
                size=order.shares,
                price=order.price,
            )

        if kind == "U":
            original_order_id = int.from_bytes(payload[11:19], "big")
            if original_order_id not in self.book.orders:
                if self._is_target_locate(locate):
                    self.book._missing_order(original_order_id)
                return None
            new_order_id = int.from_bytes(payload[19:27], "big")
            shares = int.from_bytes(payload[27:31], "big")
            price = int.from_bytes(payload[31:35], "big")
            original, replacement = self.book.replace_order(
                original_order_id,
                new_order_id,
                shares,
                price,
            )
            if original is None or replacement is None:
                return None
            self.book.stats.replaces += 1
            return self._event_row(
                record=record,
                event_type=ITCH_EVENT_REPLACE,
                order=replacement,
                size=shares,
                price=price,
                old_order_id=original_order_id,
                old_size=original.shares,
                old_price=original.price,
            )

        return None

    def metrics(self) -> dict[str, Any]:
        self.book.stats.final_open_orders = len(self.book.orders)
        return {
            "symbol": self.symbol,
            "levels": self.levels,
            "target_locates": sorted(self.target_locates),
            "message_counts": dict(sorted(self.message_counts.items())),
            "reconstruction": asdict(self.book.stats),
            "final_integrity": self.book.validate(),
        }
