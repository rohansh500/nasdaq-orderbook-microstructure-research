from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from orderbook_research.features import default_feature_columns
from orderbook_research.itch_binary import BinaryFileReader
from orderbook_research.itch_features import add_itch_snapshot_features
from orderbook_research.itch_orderbook import ItchSymbolReconstructor

SCHEMA_VERSION = "0.9.0"


class ParquetBatchWriter:
    """Write event dictionaries using one stable schema across all batches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._writer: Any | None = None
        self._pa: Any | None = None
        self._pq: Any | None = None
        self._schema: Any | None = None

        self.rows_written = 0

    def _load_pyarrow(self) -> None:
        if self._pa is not None and self._pq is not None:
            return

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Phase E Parquet export requires pyarrow. "
                "Install the repository requirements first."
            ) from exc

        self._pa = pa
        self._pq = pq

    def _build_schema(
        self,
        rows: list[dict[str, Any]],
    ) -> Any:
        """Create an explicit nullable schema instead of inferring per batch."""

        if self._pa is None:
            raise RuntimeError("PyArrow must be loaded before building the schema.")

        if not rows:
            raise ValueError("Cannot build a Parquet schema from an empty batch.")

        first_row = rows[0]

        level_numbers = {
            int(column.rsplit("_", 1)[1])
            for column in first_row
            if column.startswith(
                (
                    "ask_price_",
                    "ask_size_",
                    "bid_price_",
                    "bid_size_",
                )
            )
        }

        if not level_numbers:
            raise ValueError("No order-book level columns were found in the first batch.")

        levels = max(level_numbers)

        fields = [
            self._pa.field("event_index", self._pa.int64()),
            self._pa.field("source_message_sequence", self._pa.int64()),
            self._pa.field("source_file_offset", self._pa.int64()),
            self._pa.field("timestamp_ns", self._pa.int64()),
            self._pa.field("time_seconds", self._pa.float64()),
            self._pa.field("message_type", self._pa.string()),
            self._pa.field("event_type", self._pa.int64()),
            self._pa.field("order_id", self._pa.int64()),
            self._pa.field("old_order_id", self._pa.int64()),
            self._pa.field("size", self._pa.int64()),
            self._pa.field("old_size", self._pa.int64()),
            self._pa.field("price", self._pa.float64()),
            self._pa.field("old_price", self._pa.float64()),
            self._pa.field("execution_price", self._pa.float64()),
            self._pa.field("direction", self._pa.int64()),
            self._pa.field("stock_locate", self._pa.int64()),
            self._pa.field("stock", self._pa.string()),
        ]

        for level in range(1, levels + 1):
            fields.extend(
                [
                    self._pa.field(
                        f"ask_price_{level}",
                        self._pa.float64(),
                    ),
                    self._pa.field(
                        f"ask_size_{level}",
                        self._pa.int64(),
                    ),
                    self._pa.field(
                        f"bid_price_{level}",
                        self._pa.float64(),
                    ),
                    self._pa.field(
                        f"bid_size_{level}",
                        self._pa.int64(),
                    ),
                ]
            )

        return self._pa.schema(fields)

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        self._load_pyarrow()

        if self._pa is None or self._pq is None:
            raise RuntimeError("PyArrow could not be initialized.")

        if self._schema is None:
            self._schema = self._build_schema(rows)

        # Using the explicit schema guarantees that columns containing only
        # None in one batch retain their intended nullable numeric types.
        table = self._pa.Table.from_pylist(
            rows,
            schema=self._schema,
        )

        if self._writer is None:
            self._writer = self._pq.ParquetWriter(
                self.path,
                self._schema,
                compression="zstd",
            )

        self._writer.write_table(table)
        self.rows_written += len(rows)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def reconstruct_itch_file(
    *,
    input_path: Path,
    symbol: str,
    levels: int,
    output_directory: Path,
    max_messages: int | None = None,
    stop_after_target_events: int | None = None,
    batch_size: int = 50_000,
    sample_rows: int = 10_000,
    strict: bool = True,
    sample_start_seconds: float | None = 34_200.0,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if sample_rows < 0:
        raise ValueError("sample_rows cannot be negative.")

    output_directory.mkdir(parents=True, exist_ok=True)
    symbol = symbol.upper()
    parquet_path = output_directory / f"phase_e_itch_{symbol}_events.parquet"
    sample_path = output_directory / f"phase_e_itch_{symbol}_events_sample.csv"
    feature_sample_path = output_directory / f"phase_e_itch_{symbol}_features_sample.csv"
    counts_path = output_directory / f"phase_e_itch_{symbol}_message_counts.csv"
    metrics_path = output_directory / f"phase_e_itch_{symbol}_reconstruction_metrics.json"

    reader = BinaryFileReader(input_path, strict_lengths=strict)
    reconstructor = ItchSymbolReconstructor(
        symbol,
        levels=levels,
        strict=strict,
    )

    pending: list[dict[str, Any]] = []
    sample: list[dict[str, Any]] = []
    stop_reason: str | None = None

    tracemalloc.start()
    started = time.perf_counter()
    try:
        with ParquetBatchWriter(parquet_path) as writer:
            for record in reader:
                event = reconstructor.process(record)
                if event is not None:
                    pending.append(event)
                    within_sample_window = (
                        sample_start_seconds is None
                        or float(event["time_seconds"]) >= sample_start_seconds
                    )

                    if within_sample_window and len(sample) < sample_rows:
                        sample.append(event)
                    if len(pending) >= batch_size:
                        writer.write(pending)
                        pending.clear()

                    if (
                        stop_after_target_events is not None
                        and reconstructor.event_index >= stop_after_target_events
                    ):
                        reader.mark_stopped_early()
                        stop_reason = "stop_after_target_events"
                        break

                if max_messages is not None and record.sequence >= max_messages:
                    reader.mark_stopped_early()
                    stop_reason = "max_messages"
                    break

            writer.write(pending)
            pending.clear()
            rows_written = writer.rows_written
    finally:
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    sample_frame = pd.DataFrame(sample)
    sample_frame.to_csv(sample_path, index=False)

    if not sample_frame.empty:
        feature_sample = add_itch_snapshot_features(
            sample_frame,
            levels=levels,
        )
        useful_columns = [
            "event_index",
            "time_seconds",
            "message_type",
            "event_type",
            *default_feature_columns(levels=levels),
        ]
        available = [column for column in useful_columns if column in feature_sample]
        feature_sample[available].to_csv(feature_sample_path, index=False)
    else:
        pd.DataFrame().to_csv(feature_sample_path, index=False)

    metrics = reconstructor.metrics()
    counts = pd.DataFrame(
        [
            {"message_type": kind, "count": count}
            for kind, count in metrics["message_counts"].items()
        ]
    )
    counts.to_csv(counts_path, index=False)

    input_size = input_path.stat().st_size
    output_size = parquet_path.stat().st_size if parquet_path.exists() else 0
    records = reader.stats.records_read
    target_events = reconstructor.event_index

    result: dict[str, Any] = {
        "configuration": {
            "result_schema_version": SCHEMA_VERSION,
            "input_path": str(input_path),
            "input_filename": input_path.name,
            "symbol": symbol,
            "levels": levels,
            "strict": strict,
            "max_messages": max_messages,
            "stop_after_target_events": stop_after_target_events,
            "batch_size": batch_size,
            "sample_rows": sample_rows,
            "price_precision": 4,
            "replace_event_type": 8,
        },
        "binary_file": {
            **asdict(reader.stats),
            "input_size_bytes": input_size,
            "end_marker_required_only_for_complete_session": True,
            "stop_reason": stop_reason,
        },
        **metrics,
        "benchmark": {
            "elapsed_seconds": elapsed,
            "records_per_second": records / elapsed if elapsed else 0.0,
            "target_book_events_per_second": (target_events / elapsed if elapsed else 0.0),
            "compressed_input_megabytes_per_second": (
                input_size / 1_000_000.0 / elapsed if elapsed else 0.0
            ),
            "peak_tracemalloc_megabytes": peak_bytes / 1_000_000.0,
        },
        "outputs": {
            "parquet_path": str(parquet_path),
            "parquet_rows": rows_written,
            "parquet_size_bytes": output_size,
            "sample_csv_path": str(sample_path),
            "feature_sample_csv_path": str(feature_sample_path),
            "message_counts_csv_path": str(counts_path),
        },
    }

    metrics_path.write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return result
