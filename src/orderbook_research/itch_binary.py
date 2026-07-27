from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

_LENGTH_PREFIX = struct.Struct(">H")

# Payload lengths from Nasdaq TotalView-ITCH 5.0.  Only the message types
# required for order-book reconstruction are enforced here.
EXPECTED_MESSAGE_LENGTHS: dict[str, int] = {
    "S": 12,  # System Event
    "R": 39,  # Stock Directory
    "H": 25,  # Stock Trading Action
    "A": 36,  # Add Order
    "F": 40,  # Add Order with MPID attribution
    "E": 31,  # Order Executed
    "C": 36,  # Order Executed with Price
    "X": 23,  # Order Cancel
    "D": 19,  # Order Delete
    "U": 35,  # Order Replace
    "P": 44,  # Trade, non-cross
    "Q": 40,  # Cross Trade
    "B": 19,  # Broken Trade
}


@dataclass(frozen=True)
class BinaryRecord:
    sequence: int
    file_offset: int
    payload_length: int
    message_type: str
    payload: bytes


@dataclass
class BinaryFileReadStats:
    records_read: int = 0
    payload_bytes_read: int = 0
    end_marker_seen: bool = False
    stopped_early: bool = False
    length_mismatches: int = 0


def decode_alpha(value: bytes) -> str:
    return value.decode("ascii", errors="replace").rstrip()


def decode_timestamp_ns(value: bytes) -> int:
    if len(value) != 6:
        raise ValueError("ITCH timestamps must contain exactly six bytes.")
    return int.from_bytes(value, byteorder="big", signed=False)


def encode_timestamp_ns(value: int) -> bytes:
    if not 0 <= value < 1 << 48:
        raise ValueError("Timestamp must fit in an unsigned six-byte integer.")
    return int(value).to_bytes(6, byteorder="big", signed=False)


def message_type(payload: bytes) -> str:
    if not payload:
        raise ValueError("ITCH payload cannot be empty.")
    return chr(payload[0])


def stock_locate(payload: bytes) -> int:
    if len(payload) < 3:
        raise ValueError("Payload is too short to contain Stock Locate.")
    return int.from_bytes(payload[1:3], byteorder="big", signed=False)


def tracking_number(payload: bytes) -> int:
    if len(payload) < 5:
        raise ValueError("Payload is too short to contain Tracking Number.")
    return int.from_bytes(payload[3:5], byteorder="big", signed=False)


def timestamp_ns(payload: bytes) -> int:
    if len(payload) < 11:
        raise ValueError("Payload is too short to contain an ITCH timestamp.")
    return decode_timestamp_ns(payload[5:11])


def _open_binary_stream(path: Path) -> BinaryIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


class BinaryFileReader:
    """Stream Nasdaq BinaryFILE records without loading the session into memory."""

    def __init__(
        self,
        path: Path | str,
        *,
        strict_lengths: bool = True,
    ) -> None:
        self.path = Path(path)
        self.strict_lengths = strict_lengths
        self.stats = BinaryFileReadStats()

    def __iter__(self) -> Iterator[BinaryRecord]:
        if not self.path.exists():
            raise FileNotFoundError(f"ITCH BinaryFILE not found: {self.path}")

        with _open_binary_stream(self.path) as stream:
            sequence = 0
            file_offset = 0

            while True:
                prefix = stream.read(_LENGTH_PREFIX.size)
                if not prefix:
                    break
                if len(prefix) != _LENGTH_PREFIX.size:
                    raise EOFError("Truncated BinaryFILE length prefix.")

                payload_length = _LENGTH_PREFIX.unpack(prefix)[0]
                record_offset = file_offset
                file_offset += _LENGTH_PREFIX.size

                if payload_length == 0:
                    self.stats.end_marker_seen = True
                    break

                payload = stream.read(payload_length)
                if len(payload) != payload_length:
                    raise EOFError(
                        "Truncated BinaryFILE payload: expected "
                        f"{payload_length} bytes, received {len(payload)}."
                    )
                file_offset += payload_length

                kind = message_type(payload)
                expected = EXPECTED_MESSAGE_LENGTHS.get(kind)
                if expected is not None and expected != payload_length:
                    self.stats.length_mismatches += 1
                    if self.strict_lengths:
                        raise ValueError(
                            f"Message {kind!r} has length {payload_length}; expected {expected}."
                        )

                sequence += 1
                self.stats.records_read = sequence
                self.stats.payload_bytes_read += payload_length

                yield BinaryRecord(
                    sequence=sequence,
                    file_offset=record_offset,
                    payload_length=payload_length,
                    message_type=kind,
                    payload=payload,
                )

    def mark_stopped_early(self) -> None:
        self.stats.stopped_early = True


def write_binary_file(
    path: Path | str,
    payloads: Iterable[bytes],
    *,
    include_end_marker: bool = True,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opener = gzip.open if output_path.suffix.lower() == ".gz" else open
    with opener(output_path, "wb") as stream:
        for payload in payloads:
            if len(payload) > 65_535:
                raise ValueError("BinaryFILE payload exceeds two-byte length range.")
            stream.write(_LENGTH_PREFIX.pack(len(payload)))
            stream.write(payload)
        if include_end_marker:
            stream.write(_LENGTH_PREFIX.pack(0))

    return output_path
