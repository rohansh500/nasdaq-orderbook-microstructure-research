from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    epsilon: float = 1e-12,
) -> pd.Series:
    return numerator / (denominator.abs() + epsilon)


def best_quote_order_flow_imbalance(frame: pd.DataFrame) -> pd.Series:
    bid_price = frame["bid_price_1"]
    bid_size = frame["bid_size_1"]
    ask_price = frame["ask_price_1"]
    ask_size = frame["ask_size_1"]

    previous_bid_price = bid_price.shift(1)
    previous_bid_size = bid_size.shift(1)
    previous_ask_price = ask_price.shift(1)
    previous_ask_size = ask_size.shift(1)

    bid_contribution = (bid_price >= previous_bid_price).astype(float) * bid_size - (
        bid_price <= previous_bid_price
    ).astype(float) * previous_bid_size
    ask_contribution = (
        -(ask_price <= previous_ask_price).astype(float) * ask_size
        + (ask_price >= previous_ask_price).astype(float) * previous_ask_size
    )
    return bid_contribution + ask_contribution


def add_snapshot_features(
    frame: pd.DataFrame,
    levels: int = 10,
    rolling_windows: tuple[int, ...] = (20, 50, 100),
) -> pd.DataFrame:
    df = frame.copy()

    df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["spread_bps"] = 10_000.0 * safe_divide(df["spread"], df["mid_price"])

    top_depth = df["bid_size_1"] + df["ask_size_1"]
    df["queue_imbalance_l1"] = safe_divide(
        df["bid_size_1"] - df["ask_size_1"],
        top_depth,
    )

    df["microprice"] = safe_divide(
        (df["ask_price_1"] * df["bid_size_1"] + df["bid_price_1"] * df["ask_size_1"]),
        top_depth,
    )
    df["microprice_deviation_bps"] = 10_000.0 * safe_divide(
        df["microprice"] - df["mid_price"],
        df["mid_price"],
    )

    for depth in sorted({1, min(5, levels), levels}):
        bid_columns = [f"bid_size_{level}" for level in range(1, depth + 1)]
        ask_columns = [f"ask_size_{level}" for level in range(1, depth + 1)]

        bid_depth = df[bid_columns].sum(axis=1, min_count=1)
        ask_depth = df[ask_columns].sum(axis=1, min_count=1)
        total_depth = bid_depth + ask_depth

        df[f"bid_depth_{depth}"] = bid_depth
        df[f"ask_depth_{depth}"] = ask_depth
        df[f"total_depth_{depth}"] = total_depth
        df[f"depth_imbalance_{depth}"] = safe_divide(
            bid_depth - ask_depth,
            total_depth,
        )

    df["ofi_l1"] = best_quote_order_flow_imbalance(df)
    df["event_interarrival_us"] = df["time_seconds"].diff().clip(lower=0) * 1_000_000.0

    df["is_submission"] = (df["event_type"] == 1).astype("int8")
    df["is_partial_cancel"] = (df["event_type"] == 2).astype("int8")
    df["is_deletion"] = (df["event_type"] == 3).astype("int8")
    df["is_visible_execution"] = (df["event_type"] == 4).astype("int8")
    df["is_hidden_execution"] = (df["event_type"] == 5).astype("int8")

    df["signed_add_event"] = np.where(
        df["event_type"] == 1,
        df["direction"] * df["size"],
        0.0,
    )
    df["signed_cancel_event"] = np.where(
        df["event_type"].isin([2, 3]),
        -df["direction"] * df["size"],
        0.0,
    )
    df["signed_trade_event"] = np.where(
        df["event_type"].isin([4, 5]),
        -df["direction"] * df["size"],
        0.0,
    )

    log_mid = np.log(df["mid_price"])
    df["mid_log_return"] = log_mid.diff()
    df["seconds_from_open"] = df["time_seconds"] - 34_200.0
    df["session_fraction"] = df["seconds_from_open"] / 23_400.0
    df["time_sin"] = np.sin(2.0 * np.pi * df["session_fraction"])
    df["time_cos"] = np.cos(2.0 * np.pi * df["session_fraction"])

    normaliser = df[f"total_depth_{levels}"]

    for window in rolling_windows:
        elapsed = df["time_seconds"] - df["time_seconds"].shift(window)
        df[f"event_intensity_{window}"] = safe_divide(
            pd.Series(float(window), index=df.index),
            elapsed,
        )

        df[f"ofi_l1_sum_{window}"] = (
            df["ofi_l1"].rolling(window, min_periods=max(2, window // 4)).sum()
        )
        df[f"add_pressure_{window}"] = safe_divide(
            df["signed_add_event"]
            .rolling(
                window,
                min_periods=max(2, window // 4),
            )
            .sum(),
            normaliser,
        )
        df[f"cancel_pressure_{window}"] = safe_divide(
            df["signed_cancel_event"]
            .rolling(
                window,
                min_periods=max(2, window // 4),
            )
            .sum(),
            normaliser,
        )
        df[f"trade_pressure_{window}"] = safe_divide(
            df["signed_trade_event"]
            .rolling(
                window,
                min_periods=max(2, window // 4),
            )
            .sum(),
            normaliser,
        )
        df[f"rolling_volatility_{window}"] = (
            df["mid_log_return"].rolling(window, min_periods=max(2, window // 4)).std()
        )

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_columns] = df.loc[:, numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return df


def default_feature_columns(
    levels: int = 10,
    rolling_windows: tuple[int, ...] = (20, 50, 100),
) -> list[str]:
    columns = [
        "spread_bps",
        "queue_imbalance_l1",
        "microprice_deviation_bps",
        "depth_imbalance_1",
        f"depth_imbalance_{min(5, levels)}",
        f"depth_imbalance_{levels}",
        "ofi_l1",
        "event_interarrival_us",
        "is_submission",
        "is_partial_cancel",
        "is_deletion",
        "is_visible_execution",
        "is_hidden_execution",
        "seconds_from_open",
        "session_fraction",
        "time_sin",
        "time_cos",
    ]

    for window in rolling_windows:
        columns.extend(
            [
                f"event_intensity_{window}",
                f"ofi_l1_sum_{window}",
                f"add_pressure_{window}",
                f"cancel_pressure_{window}",
                f"trade_pressure_{window}",
                f"rolling_volatility_{window}",
            ]
        )

    return list(dict.fromkeys(columns))
