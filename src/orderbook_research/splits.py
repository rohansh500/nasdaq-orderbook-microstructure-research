from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    purge_events: int


def purged_chronological_split(
    n_rows: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    purge_events: int = 100,
) -> PurgedSplit:
    if n_rows < 1_000:
        raise ValueError("At least 1,000 rows are required for this split.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be below one."
        )
    if purge_events < 0:
        raise ValueError("purge_events cannot be negative.")

    train_boundary = int(n_rows * train_fraction)
    validation_boundary = int(
        n_rows * (train_fraction + validation_fraction)
    )

    train_end = train_boundary - purge_events
    validation_end = validation_boundary - purge_events
    test_end = n_rows - purge_events

    if min(train_end, validation_end - train_boundary, test_end - validation_boundary) <= 0:
        raise ValueError("Split blocks are too small after applying the purge.")

    split = PurgedSplit(
        train_indices=np.arange(0, train_end),
        validation_indices=np.arange(train_boundary, validation_end),
        test_indices=np.arange(validation_boundary, test_end),
        purge_events=purge_events,
    )

    assert split.train_indices.max() < split.validation_indices.min()
    assert split.validation_indices.max() < split.test_indices.min()
    assert (
        split.validation_indices.min() - split.train_indices.max() - 1
        >= purge_events
    )
    assert (
        split.test_indices.min() - split.validation_indices.max() - 1
        >= purge_events
    )

    return split
