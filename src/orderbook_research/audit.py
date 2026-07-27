from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orderbook_research.constants import EVENT_LABELS
from orderbook_research.io import load_lobster_pair
from orderbook_research.reconstruction import reconstruction_audit


def audit_data(
    ticker: str,
    levels: int,
    max_rows: int | None,
    reconstruction_events: int,
) -> dict[str, object]:
    raw = load_lobster_pair(
        ticker=ticker,
        levels=levels,
        nrows=max_rows,
        scale_prices=False,
    )

    crossed = raw["bid_price_1"] > raw["ask_price_1"]
    locked = raw["bid_price_1"] == raw["ask_price_1"]
    event_counts = raw["event_type"].value_counts().sort_index()

    summary: dict[str, object] = {
        "ticker": ticker.upper(),
        "levels": levels,
        "rows_loaded": int(len(raw)),
        "time_start_seconds": float(raw["time_seconds"].min()),
        "time_end_seconds": float(raw["time_seconds"].max()),
        "timestamps_monotonic": bool(raw["time_seconds"].is_monotonic_increasing),
        "duplicate_timestamp_fraction": float(raw["time_seconds"].duplicated().mean()),
        "crossed_book_rows": int(crossed.sum()),
        "locked_book_rows": int(locked.sum()),
        "missing_values_total": int(raw.isna().sum().sum()),
        "event_counts": {
            EVENT_LABELS.get(int(event_type), str(int(event_type))): int(count)
            for event_type, count in event_counts.items()
        },
        "median_interarrival_microseconds": float(
            raw["time_seconds"].diff().replace(0, np.nan).dropna().median() * 1_000_000.0
        ),
    }

    summary["limited_depth_reconstruction"] = reconstruction_audit(
        raw,
        levels=levels,
        max_events=reconstruction_events,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit aligned LOBSTER message and order-book files."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument(
        "--reconstruction-events",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/data_audit.json"),
    )
    args = parser.parse_args()

    summary = audit_data(
        ticker=args.ticker,
        levels=args.levels,
        max_rows=args.max_rows,
        reconstruction_events=args.reconstruction_events,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nAudit written to {args.output}")


if __name__ == "__main__":
    main()
