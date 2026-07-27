from __future__ import annotations

from pathlib import Path

from orderbook_research.itch_binary import encode_timestamp_ns, write_binary_file


def _header(
    message_type: str,
    *,
    stock_locate: int,
    tracking_number: int,
    timestamp_ns: int,
) -> bytearray:
    payload = bytearray()
    payload.extend(message_type.encode("ascii"))
    payload.extend(int(stock_locate).to_bytes(2, "big"))
    payload.extend(int(tracking_number).to_bytes(2, "big"))
    payload.extend(encode_timestamp_ns(timestamp_ns))
    return payload


def pack_system_event(timestamp_ns: int, event_code: str) -> bytes:
    payload = _header(
        "S",
        stock_locate=0,
        tracking_number=1,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(event_code.encode("ascii"))
    return bytes(payload)


def pack_stock_directory(
    *,
    timestamp_ns: int,
    stock_locate: int,
    stock: str,
) -> bytes:
    payload = _header(
        "R",
        stock_locate=stock_locate,
        tracking_number=2,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(stock.upper().ljust(8).encode("ascii"))
    payload.extend(b"Q")  # Market Category
    payload.extend(b"N")  # Financial Status
    payload.extend((100).to_bytes(4, "big"))
    payload.extend(b"N")
    payload.extend(b"C")
    payload.extend(b"  ")
    payload.extend(b"P")
    payload.extend(b"N")
    payload.extend(b"N")
    payload.extend(b"1")
    payload.extend(b"N")
    payload.extend((1).to_bytes(4, "big"))
    payload.extend(b"N")
    if len(payload) != 39:
        raise AssertionError(f"Stock Directory fixture length is {len(payload)}.")
    return bytes(payload)


def pack_add_order(
    *,
    timestamp_ns: int,
    stock_locate: int,
    order_id: int,
    side: str,
    shares: int,
    stock: str,
    price: int,
    mpid: str | None = None,
) -> bytes:
    kind = "F" if mpid is not None else "A"
    payload = _header(
        kind,
        stock_locate=stock_locate,
        tracking_number=3,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(int(order_id).to_bytes(8, "big"))
    payload.extend(side.encode("ascii"))
    payload.extend(int(shares).to_bytes(4, "big"))
    payload.extend(stock.upper().ljust(8).encode("ascii"))
    payload.extend(int(price).to_bytes(4, "big"))
    if mpid is not None:
        payload.extend(mpid.upper().ljust(4).encode("ascii"))
    expected = 40 if mpid is not None else 36
    if len(payload) != expected:
        raise AssertionError(f"Add fixture length is {len(payload)}.")
    return bytes(payload)


def pack_execute(
    *,
    timestamp_ns: int,
    stock_locate: int,
    order_id: int,
    shares: int,
    match_number: int,
    execution_price: int | None = None,
) -> bytes:
    kind = "C" if execution_price is not None else "E"
    payload = _header(
        kind,
        stock_locate=stock_locate,
        tracking_number=4,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(int(order_id).to_bytes(8, "big"))
    payload.extend(int(shares).to_bytes(4, "big"))
    payload.extend(int(match_number).to_bytes(8, "big"))
    if execution_price is not None:
        payload.extend(b"Y")
        payload.extend(int(execution_price).to_bytes(4, "big"))
    return bytes(payload)


def pack_cancel(
    *,
    timestamp_ns: int,
    stock_locate: int,
    order_id: int,
    shares: int,
) -> bytes:
    payload = _header(
        "X",
        stock_locate=stock_locate,
        tracking_number=5,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(int(order_id).to_bytes(8, "big"))
    payload.extend(int(shares).to_bytes(4, "big"))
    return bytes(payload)


def pack_delete(
    *,
    timestamp_ns: int,
    stock_locate: int,
    order_id: int,
) -> bytes:
    payload = _header(
        "D",
        stock_locate=stock_locate,
        tracking_number=6,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(int(order_id).to_bytes(8, "big"))
    return bytes(payload)


def pack_replace(
    *,
    timestamp_ns: int,
    stock_locate: int,
    original_order_id: int,
    new_order_id: int,
    shares: int,
    price: int,
) -> bytes:
    payload = _header(
        "U",
        stock_locate=stock_locate,
        tracking_number=7,
        timestamp_ns=timestamp_ns,
    )
    payload.extend(int(original_order_id).to_bytes(8, "big"))
    payload.extend(int(new_order_id).to_bytes(8, "big"))
    payload.extend(int(shares).to_bytes(4, "big"))
    payload.extend(int(price).to_bytes(4, "big"))
    return bytes(payload)


def pack_trade(
    *,
    timestamp_ns: int,
    stock_locate: int,
    stock: str,
    shares: int,
    price: int,
    match_number: int,
) -> bytes:
    payload = _header(
        "P",
        stock_locate=stock_locate,
        tracking_number=8,
        timestamp_ns=timestamp_ns,
    )
    payload.extend((0).to_bytes(8, "big"))
    payload.extend(b"B")
    payload.extend(int(shares).to_bytes(4, "big"))
    payload.extend(stock.upper().ljust(8).encode("ascii"))
    payload.extend(int(price).to_bytes(4, "big"))
    payload.extend(int(match_number).to_bytes(8, "big"))
    return bytes(payload)


def synthetic_aapl_payloads() -> list[bytes]:
    locate = 1
    second = 1_000_000_000
    return [
        pack_system_event(8 * 60 * 60 * second, "O"),
        pack_stock_directory(
            timestamp_ns=8 * 60 * 60 * second + 1,
            stock_locate=locate,
            stock="AAPL",
        ),
        pack_add_order(
            timestamp_ns=34_200 * second + 1,
            stock_locate=locate,
            order_id=1_001,
            side="B",
            shares=100,
            stock="AAPL",
            price=1_000_000,
        ),
        pack_add_order(
            timestamp_ns=34_200 * second + 2,
            stock_locate=locate,
            order_id=2_001,
            side="S",
            shares=120,
            stock="AAPL",
            price=1_000_100,
            mpid="TEST",
        ),
        pack_cancel(
            timestamp_ns=34_200 * second + 3,
            stock_locate=locate,
            order_id=1_001,
            shares=20,
        ),
        pack_execute(
            timestamp_ns=34_200 * second + 4,
            stock_locate=locate,
            order_id=2_001,
            shares=40,
            match_number=9_001,
        ),
        pack_replace(
            timestamp_ns=34_200 * second + 5,
            stock_locate=locate,
            original_order_id=1_001,
            new_order_id=1_002,
            shares=60,
            price=999_900,
        ),
        pack_add_order(
            timestamp_ns=34_200 * second + 6,
            stock_locate=locate,
            order_id=1_003,
            side="B",
            shares=70,
            stock="AAPL",
            price=1_000_000,
        ),
        pack_delete(
            timestamp_ns=34_200 * second + 7,
            stock_locate=locate,
            order_id=2_001,
        ),
        pack_add_order(
            timestamp_ns=34_200 * second + 8,
            stock_locate=locate,
            order_id=2_002,
            side="S",
            shares=50,
            stock="AAPL",
            price=1_000_200,
        ),
        pack_execute(
            timestamp_ns=34_200 * second + 9,
            stock_locate=locate,
            order_id=2_002,
            shares=10,
            match_number=9_002,
            execution_price=1_000_150,
        ),
        pack_trade(
            timestamp_ns=34_200 * second + 10,
            stock_locate=locate,
            stock="AAPL",
            shares=15,
            price=1_000_100,
            match_number=9_003,
        ),
        pack_system_event(20 * 60 * 60 * second, "C"),
    ]


def write_synthetic_aapl_fixture(path: Path | str) -> Path:
    return write_binary_file(path, synthetic_aapl_payloads())
