import numpy as np
import pytest

from orderbook_research.walk_forward import expanding_window_folds


def test_generates_five_expanding_folds():
    folds = expanding_window_folds(
        n_rows=10_000,
        n_folds=5,
        purge_events=100,
    )

    assert len(folds) == 5
    assert len(folds[1].train_indices) > len(folds[0].train_indices)
    assert len(folds[4].train_indices) > len(folds[3].train_indices)


def test_each_fold_is_ordered_disjoint_and_purged():
    folds = expanding_window_folds(
        n_rows=10_000,
        n_folds=5,
        purge_events=100,
    )

    for fold in folds:
        assert (
            np.intersect1d(
                fold.train_indices,
                fold.validation_indices,
            ).size
            == 0
        )
        assert fold.train_indices.max() < fold.validation_indices.min()

        gap = fold.validation_indices.min() - fold.train_indices.max() - 1
        assert gap >= 100


def test_validation_blocks_move_forward_and_stop_at_eighty_percent():
    folds = expanding_window_folds(
        n_rows=10_000,
        n_folds=5,
        purge_events=100,
    )

    starts = [int(fold.validation_indices.min()) for fold in folds]
    assert starts == sorted(starts)
    assert len(set(starts)) == 5
    assert folds[-1].validation_indices.max() == 7_999
    assert folds[-1].development_end == 8_000


def test_fold_metadata_matches_indices():
    fold = expanding_window_folds(
        n_rows=10_000,
        n_folds=5,
        purge_events=100,
    )[0]
    metadata = fold.metadata()

    assert metadata["fold"] == 1
    assert metadata["train_start"] == 0
    assert metadata["train_end"] == 2_899
    assert metadata["validation_start"] == 3_000
    assert metadata["validation_end"] == 3_999
    assert metadata["train_rows"] == 2_900
    assert metadata["validation_rows"] == 1_000
    assert metadata["purge_events"] == 100


def test_rejects_invalid_fraction_layout():
    with pytest.raises(ValueError):
        expanding_window_folds(
            n_rows=10_000,
            n_folds=6,
            initial_train_fraction=0.30,
            validation_fraction=0.10,
            development_fraction=0.80,
        )
