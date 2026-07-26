from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from orderbook_research.features import default_feature_columns


FEATURE_FAMILIES = ("book_state", "event_flow", "volatility", "time")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]
    removed_family: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "feature_count": len(self.columns),
            "columns": list(self.columns),
            "removed_family": self.removed_family,
        }


def feature_families(
    levels: int = 10,
    rolling_windows: tuple[int, ...] = (20, 50, 100),
) -> dict[str, list[str]]:
    book_state = [
        "spread_bps",
        "queue_imbalance_l1",
        "microprice_deviation_bps",
        "depth_imbalance_1",
        f"depth_imbalance_{min(5, levels)}",
        f"depth_imbalance_{levels}",
    ]

    event_flow = [
        "ofi_l1",
        "event_interarrival_us",
        "is_submission",
        "is_partial_cancel",
        "is_deletion",
        "is_visible_execution",
        "is_hidden_execution",
    ]
    volatility: list[str] = []
    for window in rolling_windows:
        event_flow.extend(
            [
                f"event_intensity_{window}",
                f"ofi_l1_sum_{window}",
                f"add_pressure_{window}",
                f"cancel_pressure_{window}",
                f"trade_pressure_{window}",
            ]
        )
        volatility.append(f"rolling_volatility_{window}")

    time = [
        "seconds_from_open",
        "session_fraction",
        "time_sin",
        "time_cos",
    ]

    families = {
        "book_state": list(dict.fromkeys(book_state)),
        "event_flow": list(dict.fromkeys(event_flow)),
        "volatility": list(dict.fromkeys(volatility)),
        "time": list(dict.fromkeys(time)),
    }
    validate_feature_families(families, levels=levels)
    return families


def validate_feature_families(
    families: dict[str, list[str]],
    levels: int = 10,
) -> None:
    if tuple(families) != FEATURE_FAMILIES:
        raise ValueError(
            f"Feature families must be ordered as {FEATURE_FAMILIES}."
        )

    flattened = [column for columns in families.values() for column in columns]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Feature families contain duplicate columns.")

    expected = default_feature_columns(levels=levels)
    if set(flattened) != set(expected):
        missing = sorted(set(expected) - set(flattened))
        unexpected = sorted(set(flattened) - set(expected))
        raise ValueError(
            "Feature families do not partition the default feature set. "
            f"Missing={missing}, unexpected={unexpected}."
        )


def ablation_feature_sets(levels: int = 10) -> list[FeatureSet]:
    families = feature_families(levels=levels)
    all_columns = tuple(default_feature_columns(levels=levels))
    result = [FeatureSet(name="all_features", columns=all_columns)]

    for family_name in FEATURE_FAMILIES:
        removed = set(families[family_name])
        retained = tuple(column for column in all_columns if column not in removed)
        if not retained:
            raise ValueError(f"Removing {family_name} leaves no features.")
        result.append(
            FeatureSet(
                name=f"without_{family_name}",
                columns=retained,
                removed_family=family_name,
            )
        )
    return result


def build_lightgbm_classifier(
    random_state: int = 42,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=500,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=5.0,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
    )


def build_lightgbm_regressor(
    random_state: int = 42,
) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=500,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
    )


def probability_positions(classes: Iterable[int]) -> dict[int, int]:
    result = {int(label): position for position, label in enumerate(classes)}
    missing = {-1, 1} - set(result)
    if missing:
        raise ValueError(
            f"Classifier probabilities are missing classes {sorted(missing)}."
        )
    return result


def normalized_feature_importance(
    feature_names: list[str] | tuple[str, ...],
    importances: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(importances, dtype=float)
    if len(feature_names) != len(values):
        raise ValueError("Feature names and importances have different lengths.")
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    total = float(values.sum())
    if total > 0:
        values = values / total
    return {
        str(feature): float(value)
        for feature, value in zip(feature_names, values, strict=True)
    }
