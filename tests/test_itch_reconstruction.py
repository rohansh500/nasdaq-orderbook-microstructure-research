from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import orderbook_research.itch_reconstruction as reconstruction_module
from orderbook_research.itch_fixture import write_synthetic_aapl_fixture
from orderbook_research.itch_reconstruction import reconstruct_itch_file


class FakeParquetBatchWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows_written = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        self.rows_written += len(rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("test parquet placeholder", encoding="utf-8")

    def __enter__(self) -> "FakeParquetBatchWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


def test_reconstruction_runner_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reconstruction_module,
        "ParquetBatchWriter",
        FakeParquetBatchWriter,
    )
    input_path = write_synthetic_aapl_fixture(tmp_path / "fixture.itch.gz")
    output = tmp_path / "output"

    result = reconstruct_itch_file(
        input_path=input_path,
        symbol="AAPL",
        levels=3,
        output_directory=output,
        batch_size=3,
        sample_rows=20,
    )

    assert result["configuration"]["result_schema_version"] == "0.9.0"
    assert result["configuration"]["sample_start_seconds"] == 34_200.0
    assert result["reconstruction"]["target_book_events"] == 9
    assert result["final_integrity"]["valid"] is True
    assert result["outputs"]["parquet_rows"] == 9
    assert (output / "phase_e_itch_AAPL_events.parquet").exists()
    assert (output / "phase_e_itch_AAPL_events_sample.csv").exists()
    assert (output / "phase_e_itch_AAPL_features_sample.csv").exists()
    assert (output / "phase_e_itch_AAPL_message_counts.csv").exists()
    assert (output / "phase_e_itch_AAPL_reconstruction_metrics.json").exists()

    payload = json.loads((output / "phase_e_itch_AAPL_reconstruction_metrics.json").read_text())
    assert payload["binary_file"]["end_marker_seen"] is True


def test_reconstruction_can_stop_after_target_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reconstruction_module,
        "ParquetBatchWriter",
        FakeParquetBatchWriter,
    )
    input_path = write_synthetic_aapl_fixture(tmp_path / "fixture.itch")
    result = reconstruct_itch_file(
        input_path=input_path,
        symbol="AAPL",
        levels=2,
        output_directory=tmp_path / "partial",
        stop_after_target_events=4,
        batch_size=2,
    )

    assert result["reconstruction"]["target_book_events"] == 4
    assert result["binary_file"]["stopped_early"] is True
    assert result["binary_file"]["stop_reason"] == "stop_after_target_events"


def test_real_parquet_writer_when_pyarrow_is_available(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    input_path = write_synthetic_aapl_fixture(tmp_path / "fixture.itch")
    output = tmp_path / "real_parquet"

    reconstruct_itch_file(
        input_path=input_path,
        symbol="AAPL",
        levels=2,
        output_directory=output,
        batch_size=2,
    )

    frame = pd.read_parquet(output / "phase_e_itch_AAPL_events.parquet")
    assert len(frame) == 9
