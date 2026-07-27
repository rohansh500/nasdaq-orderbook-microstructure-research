from __future__ import annotations

import argparse
import json
from pathlib import Path

from orderbook_research.itch_fixture import write_synthetic_aapl_fixture
from orderbook_research.itch_reconstruction import reconstruct_itch_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a Nasdaq TotalView-ITCH 5.0 BinaryFILE, reconstruct a "
            "single-symbol full-depth book, and export research snapshots."
        )
    )
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/tables"),
    )
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--stop-after-target-events", type=int)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--generate-fixture",
        type=Path,
        help="Write and process a small synthetic AAPL BinaryFILE fixture.",
    )
    parser.add_argument(
        "--sample-start-seconds",
        type=float,
        default=34_200.0,
        help="Earliest timestamp included in the lightweight output sample.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if args.generate_fixture is not None:
        input_path = write_synthetic_aapl_fixture(args.generate_fixture)
    if input_path is None:
        parser.error("Provide --input-path or --generate-fixture.")

    result = reconstruct_itch_file(
        input_path=input_path,
        symbol=args.symbol,
        levels=args.levels,
        output_directory=args.output_directory,
        max_messages=args.max_messages,
        stop_after_target_events=args.stop_after_target_events,
        batch_size=args.batch_size,
        sample_rows=args.sample_rows,
        strict=not args.non_strict,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
