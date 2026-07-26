from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "totalorganfailure/lobster-data"
SAMPLE_DATE = "2012-06-21"
START_MS = "34200000"
END_MS = "57600000"


def filenames(ticker: str, levels: int) -> tuple[str, str, str]:
    ticker = ticker.upper()
    folder = f"LOBSTER_SampleFile_{ticker}_{SAMPLE_DATE}_{levels}"
    stem = f"{ticker}_{SAMPLE_DATE}_{START_MS}_{END_MS}"
    message = f"{folder}/{stem}_message_{levels}.csv"
    orderbook = f"{folder}/{stem}_orderbook_{levels}.csv"
    return folder, message, orderbook


def download_sample(
    ticker: str,
    levels: int,
    output_root: Path,
) -> tuple[Path, Path]:
    ticker = ticker.upper()
    folder, message_filename, orderbook_filename = filenames(ticker, levels)
    destination = output_root / f"{ticker}_{levels}"
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Downloading public {ticker} {levels}-level LOBSTER sample...")
    message_cached = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=message_filename,
    )
    orderbook_cached = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=orderbook_filename,
    )

    message_destination = destination / Path(message_filename).name
    orderbook_destination = destination / Path(orderbook_filename).name

    message_destination.write_bytes(Path(message_cached).read_bytes())
    orderbook_destination.write_bytes(Path(orderbook_cached).read_bytes())

    print(f"Message file:   {message_destination}")
    print(f"Order-book file: {orderbook_destination}")
    return message_destination, orderbook_destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a public LOBSTER sample from Hugging Face."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/lobster"),
    )
    args = parser.parse_args()

    try:
        download_sample(args.ticker, args.levels, args.output_root)
    except Exception as exc:
        raise RuntimeError(
            "The public sample could not be downloaded. "
            "Check internet access and the ticker/level combination."
        ) from exc


if __name__ == "__main__":
    main()
