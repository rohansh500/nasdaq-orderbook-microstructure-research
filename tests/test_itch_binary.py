from __future__ import annotations

from pathlib import Path

import pytest

from orderbook_research.itch_binary import BinaryFileReader, write_binary_file
from orderbook_research.itch_fixture import (
    pack_add_order,
    pack_system_event,
    synthetic_aapl_payloads,
    write_synthetic_aapl_fixture,
)


def test_binary_file_reader_reads_records_and_end_marker(tmp_path: Path) -> None:
    path = write_synthetic_aapl_fixture(tmp_path / "fixture.itch")
    reader = BinaryFileReader(path)
    records = list(reader)

    assert len(records) == len(synthetic_aapl_payloads())
    assert records[0].message_type == "S"
    assert records[2].message_type == "A"
    assert reader.stats.end_marker_seen is True
    assert reader.stats.length_mismatches == 0


def test_binary_file_reader_supports_gzip(tmp_path: Path) -> None:
    path = write_synthetic_aapl_fixture(tmp_path / "fixture.itch.gz")
    records = list(BinaryFileReader(path))
    assert len(records) == len(synthetic_aapl_payloads())


def test_binary_file_reader_detects_wrong_known_length(tmp_path: Path) -> None:
    malformed = pack_system_event(1, "O") + b"x"
    path = write_binary_file(tmp_path / "bad.itch", [malformed])

    with pytest.raises(ValueError, match="expected 12"):
        list(BinaryFileReader(path, strict_lengths=True))


def test_binary_file_reader_detects_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated.itch"
    path.write_bytes(b"\x00\x24" + b"A" * 10)

    with pytest.raises(EOFError, match="Truncated BinaryFILE payload"):
        list(BinaryFileReader(path))


def test_fixture_add_message_uses_expected_length() -> None:
    payload = pack_add_order(
        timestamp_ns=1,
        stock_locate=1,
        order_id=10,
        side="B",
        shares=100,
        stock="AAPL",
        price=1_000_000,
    )
    assert len(payload) == 36
