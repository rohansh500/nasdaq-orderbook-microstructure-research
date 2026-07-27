import numpy as np
import pytest

from orderbook_research.features import default_feature_columns
from orderbook_research.final_evaluation import (
    final_holdout_split,
    selected_feature_columns,
    stable_configuration_hash,
    validate_run_request,
)
from orderbook_research.model_comparison import feature_families


def test_selected_feature_set_removes_only_time_features() -> None:
    selected = selected_feature_columns(levels=10)
    all_columns = default_feature_columns(levels=10)
    time_columns = feature_families(levels=10)["time"]

    assert len(all_columns) == 35
    assert len(selected) == 31
    assert set(selected) == set(all_columns) - set(time_columns)


def test_final_holdout_split_is_chronological_and_purged() -> None:
    split = final_holdout_split(
        400_000,
        development_fraction=0.80,
        purge_events=100,
        horizon=50,
    )

    assert split.boundary_index == 320_000
    assert split.development_indices[0] == 0
    assert split.development_indices[-1] == 319_899
    assert split.holdout_indices[0] == 320_000
    assert split.holdout_indices[-1] == 399_999
    assert split.holdout_indices.min() - split.development_indices.max() - 1 == 100
    assert np.intersect1d(split.development_indices, split.holdout_indices).size == 0


def test_final_holdout_split_uses_horizon_as_minimum_purge() -> None:
    split = final_holdout_split(
        10_000,
        purge_events=20,
        horizon=100,
    )
    assert split.purge_events == 100


def test_configuration_hash_is_stable_and_sensitive() -> None:
    first = stable_configuration_hash({"b": 2, "a": 1})
    second = stable_configuration_hash({"a": 1, "b": 2})
    changed = stable_configuration_hash({"a": 1, "b": 3})

    assert first == second
    assert first != changed


def test_final_run_requires_explicit_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm-final-holdout"):
        validate_run_request(
            smoke=False,
            confirm_final_holdout=False,
            max_rows=None,
        )

    validate_run_request(
        smoke=False,
        confirm_final_holdout=True,
        max_rows=None,
    )


def test_smoke_mode_requires_limited_rows() -> None:
    with pytest.raises(ValueError, match="max-rows"):
        validate_run_request(
            smoke=True,
            confirm_final_holdout=False,
            max_rows=None,
        )

    validate_run_request(
        smoke=True,
        confirm_final_holdout=False,
        max_rows=50_000,
    )
