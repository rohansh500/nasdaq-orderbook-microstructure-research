from __future__ import annotations

import numpy as np
import pytest

from orderbook_research.features import default_feature_columns
from orderbook_research.model_comparison import (
    FEATURE_FAMILIES,
    ablation_feature_sets,
    build_lightgbm_classifier,
    build_lightgbm_regressor,
    feature_families,
    normalized_feature_importance,
    probability_positions,
)


def test_feature_families_partition_default_features() -> None:
    families = feature_families(levels=10)
    assert tuple(families) == FEATURE_FAMILIES

    flattened = [column for columns in families.values() for column in columns]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(default_feature_columns(levels=10))


def test_ablation_sets_drop_exactly_one_family() -> None:
    families = feature_families(levels=10)
    all_features = set(default_feature_columns(levels=10))
    sets = {feature_set.name: feature_set for feature_set in ablation_feature_sets(10)}

    assert set(sets) == {
        "all_features",
        "without_book_state",
        "without_event_flow",
        "without_volatility",
        "without_time",
    }
    assert set(sets["all_features"].columns) == all_features
    for family_name, family_columns in families.items():
        retained = set(sets[f"without_{family_name}"].columns)
        assert retained == all_features - set(family_columns)


def test_lightgbm_builders_use_fixed_regularized_parameters() -> None:
    classifier = build_lightgbm_classifier()
    regressor = build_lightgbm_regressor()

    assert classifier.get_params()["n_estimators"] == 250
    assert classifier.get_params()["class_weight"] == "balanced"
    assert classifier.get_params()["num_leaves"] == 15
    assert regressor.get_params()["objective"] == "regression_l1"
    assert regressor.get_params()["reg_lambda"] == 5.0


def test_probability_positions_requires_up_and_down() -> None:
    assert probability_positions([-1, 0, 1]) == {-1: 0, 0: 1, 1: 2}
    with pytest.raises(ValueError):
        probability_positions([0, 1])


def test_normalized_feature_importance_sums_to_one() -> None:
    result = normalized_feature_importance(["a", "b", "c"], np.array([2.0, 1.0, 1.0]))
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["a"] == pytest.approx(0.5)


def test_normalized_feature_importance_handles_zero_gain() -> None:
    result = normalized_feature_importance(["a", "b"], np.array([0.0, np.nan]))
    assert result == {"a": 0.0, "b": 0.0}
