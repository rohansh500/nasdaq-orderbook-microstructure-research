from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from orderbook_research.constants import (
    DUMMY_ASK_PRICE,
    DUMMY_BID_PRICE,
    MESSAGE_COLUMNS,
    PRICE_SCALE,
)


def orderbook_columns(levels: int) -> list[str]:
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def expected_paths(
    ticker: str = "AAPL",
    levels: int = 10,
    root: Path = Path("data/raw/lobster"),
) -> tuple[Path, Path]:
    ticker = ticker.upper()
    directory = root / f"{ticker}_{levels}"
    stem = f"{ticker}_2012-06-21_34200000_57600000"
    return (
        directory / f"{stem}_message_{levels}.csv",
        directory / f"{stem}_orderbook_{levels}.csv",
    )


def read_message_file(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Message file not found: {path}. Run the download script first."
        )

    return pd.read_csv(
        path,
        header=None,
        names=MESSAGE_COLUMNS,
        nrows=nrows,
        dtype={
            "time_seconds": "float64",
            "event_type": "int8",
            "order_id": "int64",
            "size": "int64",
            "price": "int64",
            "direction": "int8",
        },
    )


def read_orderbook_file(
    path: Path,
    levels: int,
    nrows: int | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Order-book file not found: {path}. Run the download script first."
        )

    return pd.read_csv(
        path,
        header=None,
        names=orderbook_columns(levels),
        nrows=nrows,
        dtype="int64",
    )


def clean_dummy_levels(frame: pd.DataFrame, levels: int) -> pd.DataFrame:
    df = frame.copy()
    for level in range(1, levels + 1):
        ask_price = f"ask_price_{level}"
        ask_size = f"ask_size_{level}"
        bid_price = f"bid_price_{level}"
        bid_size = f"bid_size_{level}"

        ask_dummy = df[ask_price] == DUMMY_ASK_PRICE
        bid_dummy = df[bid_price] == DUMMY_BID_PRICE

        df.loc[ask_dummy, [ask_price, ask_size]] = np.nan
        df.loc[bid_dummy, [bid_price, bid_size]] = np.nan
    return df


def load_lobster_pair(
    ticker: str = "AAPL",
    levels: int = 10,
    root: Path = Path("data/raw/lobster"),
    nrows: int | None = None,
    scale_prices: bool = True,
) -> pd.DataFrame:
    message_path, orderbook_path = expected_paths(ticker, levels, root)
    messages = read_message_file(message_path, nrows=nrows)
    orderbook = read_orderbook_file(orderbook_path, levels, nrows=nrows)

    if len(messages) != len(orderbook):
        raise ValueError(
            "Message and order-book row counts differ: "
            f"{len(messages)} versus {len(orderbook)}."
        )

    df = pd.concat(
        [
            messages.reset_index(drop=True),
            orderbook.reset_index(drop=True),
        ],
        axis=1,
    )
    df.insert(0, "event_index", np.arange(len(df), dtype=np.int64))
    df = clean_dummy_levels(df, levels)

    if not df["time_seconds"].is_monotonic_increasing:
        raise ValueError("Event timestamps are not monotonically non-decreasing.")

    if scale_prices:
        price_columns = ["price"] + [
            column
            for column in orderbook_columns(levels)
            if "price" in column
        ]
        df[price_columns] = df[price_columns] / PRICE_SCALE

    return df
