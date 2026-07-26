from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpandingWindowFold:
    """Indices and metadata for one purged expanding-window fold."""

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    purge_events: int
    development_end: int

    def metadata(self) -> dict[str, int]:
        return {
            "fold": self.fold,
            "train_start": int(self.train_indices[0]),
            "train_end": int(self.train_indices[-1]),
            "validation_start": int(self.validation_indices[0]),
            "validation_end": int(self.validation_indices[-1]),
            "train_rows": int(len(self.train_indices)),
            "validation_rows": int(len(self.validation_indices)),
            "purge_events": int(self.purge_events),
            "development_end_exclusive": int(self.development_end),
        }


def expanding_window_folds(
    n_rows: int,
    n_folds: int = 5,
    development_fraction: float = 0.80,
    initial_train_fraction: float = 0.30,
    validation_fraction: float = 0.10,
    purge_events: int = 100,
) -> list[ExpandingWindowFold]:
    """Create purged expanding-window folds in chronological event order.

    The default layout uses the first 80% of observations for development:

    - fold 1 trains on approximately 0-30% and validates on 30-40%;
    - fold 2 trains on approximately 0-40% and validates on 40-50%;
    - ...;
    - fold 5 trains on approximately 0-70% and validates on 70-80%.

    A purge is removed from the end of each training block so that a training
    label with a future event horizon cannot reach into its validation block.
    The final 20% is not included in these folds.
    """
    if n_rows < 1_000:
        raise ValueError("At least 1,000 rows are required.")
    if n_folds < 1:
        raise ValueError("n_folds must be at least one.")
    if purge_events < 0:
        raise ValueError("purge_events cannot be negative.")

    for name, value in {
        "development_fraction": development_fraction,
        "initial_train_fraction": initial_train_fraction,
        "validation_fraction": validation_fraction,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between zero and one.")

    required_fraction = initial_train_fraction + n_folds * validation_fraction
    if required_fraction > development_fraction + 1e-12:
        raise ValueError(
            "initial_train_fraction + n_folds * validation_fraction "
            "cannot exceed development_fraction."
        )

    development_end = int(n_rows * development_fraction)
    first_validation_start = int(n_rows * initial_train_fraction)
    validation_size = int(n_rows * validation_fraction)

    if validation_size <= 0:
        raise ValueError("validation_fraction produces an empty block.")

    folds: list[ExpandingWindowFold] = []

    for fold_number in range(1, n_folds + 1):
        validation_start = (
            first_validation_start + (fold_number - 1) * validation_size
        )
        validation_end = min(
            validation_start + validation_size,
            development_end,
        )
        train_end_exclusive = validation_start - purge_events

        if train_end_exclusive <= 0:
            raise ValueError(
                f"Fold {fold_number} has no training rows after purging."
            )
        if validation_end <= validation_start:
            raise ValueError(
                f"Fold {fold_number} has an empty validation block."
            )

        train_indices = np.arange(0, train_end_exclusive, dtype=np.int64)
        validation_indices = np.arange(
            validation_start,
            validation_end,
            dtype=np.int64,
        )

        gap = (
            int(validation_indices.min())
            - int(train_indices.max())
            - 1
        )
        if gap < purge_events:
            raise AssertionError(
                f"Fold {fold_number} purge gap is {gap}, expected at least "
                f"{purge_events}."
            )

        folds.append(
            ExpandingWindowFold(
                fold=fold_number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                purge_events=purge_events,
                development_end=development_end,
            )
        )

    for previous, current in zip(folds, folds[1:]):
        if len(current.train_indices) <= len(previous.train_indices):
            raise AssertionError("Training windows must expand by fold.")
        if (
            current.validation_indices.min()
            <= previous.validation_indices.min()
        ):
            raise AssertionError(
                "Validation windows must move forward chronologically."
            )

    if folds[-1].validation_indices[-1] >= development_end:
        raise AssertionError(
            "The final validation block must remain inside development data."
        )

    return folds
