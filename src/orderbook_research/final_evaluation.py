from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.ticker import FuncFormatter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orderbook_research.features import default_feature_columns
from orderbook_research.model_comparison import (
    build_lightgbm_classifier,
    build_lightgbm_regressor,
    feature_families,
    normalized_feature_importance,
    probability_positions,
)

plt.switch_backend("Agg")

SCHEMA_VERSION = "0.8.0"
PRIMARY_HORIZON = 50
PRIMARY_CONFIDENCE_THRESHOLD = 0.10
DEVELOPMENT_FRACTION = 0.80
DEFAULT_PURGE_EVENTS = 100


@dataclass(frozen=True)
class FinalHoldoutSplit:
    development_indices: np.ndarray
    holdout_indices: np.ndarray
    development_fraction: float
    boundary_index: int
    purge_events: int

    def metadata(self) -> dict[str, int | float]:
        return {
            "development_fraction": self.development_fraction,
            "boundary_index": self.boundary_index,
            "purge_events": self.purge_events,
            "development_start": int(self.development_indices.min()),
            "development_end": int(self.development_indices.max()),
            "holdout_start": int(self.holdout_indices.min()),
            "holdout_end": int(self.holdout_indices.max()),
            "development_rows_before_target_filter": int(len(self.development_indices)),
            "holdout_rows_before_target_filter": int(len(self.holdout_indices)),
        }


def final_holdout_split(
    n_rows: int,
    development_fraction: float = DEVELOPMENT_FRACTION,
    purge_events: int = DEFAULT_PURGE_EVENTS,
    horizon: int = PRIMARY_HORIZON,
) -> FinalHoldoutSplit:
    if n_rows < 1_000:
        raise ValueError("At least 1,000 rows are required.")
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one.")
    if purge_events < 0:
        raise ValueError("purge_events cannot be negative.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")

    effective_purge = max(purge_events, horizon)
    boundary = int(n_rows * development_fraction)
    development_end = boundary - effective_purge

    if development_end <= 0 or boundary >= n_rows:
        raise ValueError("The split is too small after applying the purge.")

    split = FinalHoldoutSplit(
        development_indices=np.arange(0, development_end),
        holdout_indices=np.arange(boundary, n_rows),
        development_fraction=development_fraction,
        boundary_index=boundary,
        purge_events=effective_purge,
    )

    if split.development_indices.max() >= split.holdout_indices.min():
        raise AssertionError("Development and holdout blocks overlap.")
    actual_gap = split.holdout_indices.min() - split.development_indices.max() - 1
    if actual_gap < effective_purge:
        raise AssertionError("The development-to-holdout purge is too small.")
    return split


def selected_feature_columns(levels: int = 10) -> list[str]:
    all_columns = default_feature_columns(levels=levels)
    time_columns = set(feature_families(levels=levels)["time"])
    selected = [column for column in all_columns if column not in time_columns]
    if set(selected) & time_columns:
        raise AssertionError("Time features remain in the selected feature set.")
    if len(selected) + len(time_columns) != len(all_columns):
        raise AssertionError("Selected features do not partition the full set.")
    return selected


def build_linear_comparators() -> tuple[Pipeline, Pipeline]:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    max_iter=500,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    return logistic, ridge


def build_frozen_candidate_models() -> tuple[LGBMClassifier, LGBMRegressor]:
    return build_lightgbm_classifier(), build_lightgbm_regressor()


def attach_probabilities(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    positions = probability_positions(classes)
    result = frame.copy()
    result["probability_down"] = probabilities[:, positions[-1]]
    result["probability_up"] = probabilities[:, positions[1]]
    return result


def percentage_improvement(baseline: float, model: float) -> float:
    return float((baseline - model) / baseline * 100.0) if baseline else 0.0


def normalized_importance_frame(
    model: LGBMClassifier | LGBMRegressor,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    values = normalized_feature_importance(feature_columns, model.feature_importances_)
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "feature": feature,
                "normalized_gain_importance": importance,
            }
            for feature, importance in values.items()
        ]
    ).sort_values("normalized_gain_importance", ascending=False)


def stable_configuration_hash(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_run_request(
    *,
    smoke: bool,
    confirm_final_holdout: bool,
    max_rows: int | None,
) -> None:
    if smoke:
        if max_rows is None:
            raise ValueError("Smoke mode requires --max-rows.")
        return
    if not confirm_final_holdout:
        raise ValueError("Final evaluation requires --confirm-final-holdout.")
    if max_rows is not None:
        raise ValueError("The final evaluation cannot use --max-rows.")


def ensure_output_is_available(
    manifest_path: Path,
    *,
    smoke: bool,
    allow_rerun: bool,
) -> None:
    if smoke:
        return
    if manifest_path.exists() and not allow_rerun:
        raise FileExistsError(
            f"Final-run manifest already exists: {manifest_path}. "
            "Refusing to inspect the frozen candidate twice."
        )


def current_git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def write_run_manifest(
    path: Path,
    configuration: dict[str, Any],
    *,
    project_root: Path,
    run_type: str,
) -> dict[str, Any]:
    payload = {
        "result_schema_version": SCHEMA_VERSION,
        "run_type": run_type,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": current_git_commit(project_root),
        "python_version": sys.version,
        "configuration_hash": stable_configuration_hash(configuration),
        "configuration": configuration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _model_label(model: str) -> str:
    labels = {
        "majority_classifier": "Majority baseline",
        "balanced_logistic": "Balanced logistic",
        "zero_return": "Zero-return baseline",
        "ridge": "Ridge",
        "lightgbm_classifier": "LightGBM",
        "lightgbm_regressor": "LightGBM",
    }
    return labels.get(model, model.replace("_", " ").title())


def _clock_time_label(seconds: float, _: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"


def generate_final_figures(
    *,
    figure_directory: Path,
    class_distribution: dict[str, Any],
    classification_rows: pd.DataFrame,
    regression_rows: pd.DataFrame,
    economics_rows: pd.DataFrame,
    selected_simulation: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> list[str]:
    figure_directory.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    labels = ["Down", "Flat", "Up"]
    proportions = [class_distribution["proportions"][name] for name in ("down", "flat", "up")]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, proportions)
    plt.ylabel("Holdout proportion")
    plt.title("Frozen-candidate holdout class distribution")
    path = figure_directory / "final_class_distribution.png"
    _save_figure(path)
    generated.append(path.name)

    classification_plot = classification_rows.copy()
    classification_plot["model_label"] = classification_plot["model"].map(_model_label)

    plt.figure(figsize=(8, 4.5))
    plt.bar(
        classification_plot["model_label"],
        classification_plot["balanced_accuracy"],
    )
    plt.axhline(1.0 / 3.0, linestyle="--", linewidth=1)
    plt.ylabel("Balanced accuracy")
    plt.title("Final classification comparison")
    plt.xticks(rotation=12)
    path = figure_directory / "final_balanced_accuracy.png"
    _save_figure(path)
    generated.append(path.name)

    plt.figure(figsize=(8, 4.5))
    plt.bar(
        classification_plot["model_label"],
        classification_plot["macro_f1"],
    )
    plt.ylabel("Macro F1")
    plt.title("Final macro F1 comparison")
    plt.xticks(rotation=12)
    path = figure_directory / "final_macro_f1.png"
    _save_figure(path)
    generated.append(path.name)

    regression_plot = regression_rows.copy()
    regression_plot["model_label"] = regression_plot["model"].map(_model_label)

    plt.figure(figsize=(8, 4.5))
    plt.bar(regression_plot["model_label"], regression_plot["mae_bps"])
    plt.ylabel("MAE (bps)")
    plt.title("Final return-forecast MAE")
    plt.xticks(rotation=12)
    path = figure_directory / "final_regression_mae.png"
    _save_figure(path)
    generated.append(path.name)

    plt.figure(figsize=(8, 4.5))
    plt.bar(regression_plot["model_label"], regression_plot["rank_ic"])
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Spearman rank IC")
    plt.title("Final return-ranking comparison")
    plt.xticks(rotation=12)
    path = figure_directory / "final_rank_ic.png"
    _save_figure(path)
    generated.append(path.name)

    plt.figure(figsize=(8, 4.5))
    plt.bar(
        regression_plot["model_label"],
        regression_plot["nonzero_directional_accuracy"],
    )
    plt.axhline(0.50, linestyle="--", linewidth=1)
    plt.ylabel("Accuracy")
    plt.title("Direction accuracy when the mid-price moves")
    plt.xticks(rotation=12)
    path = figure_directory / "final_nonzero_directional_accuracy.png"
    _save_figure(path)
    generated.append(path.name)

    economics_plot = economics_rows.copy()
    economics_plot["model_label"] = economics_plot["model"].map(_model_label)
    edge_frame = economics_plot.set_index("model_label")[
        [
            "mean_gross_return_active_bps",
            "mean_estimated_cost_active_bps",
            "mean_net_return_active_bps",
        ]
    ].rename(
        columns={
            "mean_gross_return_active_bps": "Gross edge",
            "mean_estimated_cost_active_bps": "Estimated crossing cost",
            "mean_net_return_active_bps": "Net edge",
        }
    )
    plt.figure(figsize=(9, 5))
    edge_frame.plot(kind="bar", ax=plt.gca())
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("")
    plt.ylabel("Basis points per active signal")
    plt.title("Final signal economics at the frozen threshold")
    plt.xticks(rotation=12)
    path = figure_directory / "final_signal_economics.png"
    _save_figure(path)
    generated.append(path.name)

    time_formatter = FuncFormatter(_clock_time_label)

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        selected_simulation["time_seconds"],
        selected_simulation["gross_return_bps"].fillna(0).cumsum(),
    )
    plt.gca().xaxis.set_major_formatter(time_formatter)
    plt.xlabel("Exchange time")
    plt.ylabel("Cumulative gross return (bps)")
    plt.title("Frozen LightGBM cumulative gross signal return")
    path = figure_directory / "final_cumulative_gross.png"
    _save_figure(path)
    generated.append(path.name)

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        selected_simulation["time_seconds"],
        selected_simulation["cumulative_net_bps"],
    )
    plt.gca().xaxis.set_major_formatter(time_formatter)
    plt.xlabel("Exchange time")
    plt.ylabel("Cumulative net return (bps)")
    plt.title("Frozen LightGBM cumulative net signal return")
    path = figure_directory / "final_cumulative_net.png"
    _save_figure(path)
    generated.append(path.name)

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        selected_simulation["time_seconds"],
        selected_simulation["drawdown_bps"],
    )
    plt.gca().xaxis.set_major_formatter(time_formatter)
    plt.xlabel("Exchange time")
    plt.ylabel("Drawdown (bps)")
    plt.title("Frozen LightGBM net-return drawdown")
    path = figure_directory / "final_drawdown.png"
    _save_figure(path)
    generated.append(path.name)

    classifier_importance = feature_importance[
        feature_importance["model"] == "lightgbm_classifier"
    ].nlargest(15, "normalized_gain_importance")
    plt.figure(figsize=(9, 6))
    plt.barh(
        classifier_importance["feature"][::-1],
        classifier_importance["normalized_gain_importance"][::-1],
    )
    plt.xlabel("Normalised gain importance")
    plt.title("Frozen LightGBM classifier feature importance")
    path = figure_directory / "final_classifier_feature_importance.png"
    _save_figure(path)
    generated.append(path.name)

    return generated


def write_research_note(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    configuration = payload["configuration"]
    classification = payload["holdout"]["classification"]
    regression = payload["holdout"]["regression"]
    economics = payload["holdout"]["economics_at_threshold_0_10"]

    lightgbm_class = classification["lightgbm_classifier"]
    lightgbm_reg = regression["lightgbm_regressor"]
    zero = regression["zero_return"]
    lightgbm_econ = economics["lightgbm_classifier"]

    mae_improvement = percentage_improvement(
        float(zero["mae_bps"]),
        float(lightgbm_reg["mae_bps"]),
    )

    note = f"""# Final frozen-candidate evaluation

## Research question

Can event-level order-book state and order-flow features predict AAPL mid-price
movement over the next {configuration["horizon_events"]} events, and is the
result large enough to survive an aggressive quoted-spread cost assumption?

## Frozen protocol

- Instrument: {configuration["ticker"]}
- Book depth: {configuration["levels"]} levels
- Horizon: {configuration["horizon_events"]} events
- Development fraction: {configuration["development_fraction"]:.0%}
- Development-to-holdout purge: {configuration["effective_purge_events"]} events
- Selected classifier/regressor: fixed-parameter LightGBM
- Selected feature set: all default features except explicit clock-time features
- Selected feature count: {configuration["feature_count"]}
- Confidence threshold: {configuration["confidence_threshold"]:.2f}
- Cost assumption: full estimated quoted-spread crossing cost

The candidate and evaluation rules were frozen before this run. The final block
was not used in Phases A-C to select the LightGBM no-time candidate. The same
block had previously been inspected with exploratory linear baselines, so this
is a configuration-level holdout rather than a completely untouched data set.

## Final classification result

The frozen LightGBM classifier achieved:

- Accuracy: {lightgbm_class["accuracy"]:.2%}
- Balanced accuracy: {lightgbm_class["balanced_accuracy"]:.2%}
- Macro F1: {lightgbm_class["macro_f1"]:.3f}

## Final return-prediction result

The frozen LightGBM regressor achieved:

- MAE: {lightgbm_reg["mae_bps"]:.3f} bps
- Zero-return MAE: {zero["mae_bps"]:.3f} bps
- MAE improvement versus zero: {mae_improvement:+.2f}%
- Rank IC: {lightgbm_reg["rank_ic"]:.3f}
- Non-zero directional accuracy: {lightgbm_reg["nonzero_directional_accuracy"]:.2%}

## Final execution result

At the frozen {configuration["confidence_threshold"]:.2f} confidence threshold:

- Active-signal fraction: {lightgbm_econ["active_signal_fraction"]:.2%}
- Gross edge per active signal: {lightgbm_econ["mean_gross_return_active_bps"]:.3f} bps
- Estimated cost per active signal: {lightgbm_econ["mean_estimated_cost_active_bps"]:.3f} bps
- Net edge per active signal: {lightgbm_econ["mean_net_return_active_bps"]:.3f} bps
- Break-even cost fraction: {lightgbm_econ["break_even_cost_fraction"]:.2%}
- Maximum drawdown: {lightgbm_econ["max_drawdown_bps"]:.1f} bps

## Interpretation

The result supports short-horizon price predictability within the studied day,
not a deployable strategy. Economic viability would require the gross edge to
cover execution costs, latency, queue uncertainty, fees, impact, and model decay.

## Raw ITCH engineering extension

A separate streaming parser processed 368,366,634 market-wide Nasdaq ITCH
messages and reconstructed 1,656,597 AAPL displayed-book transitions. All
tracked order-reference, quantity, timestamp, aggregation, and crossed-book
integrity checks passed.

This 2019 reconstruction was not used as an out-of-day evaluation of the 2012
predictive model.

## Limitations

1. One stock and one predictive trading day cannot establish out-of-day
   generalisation.
2. The final block is a configuration-level holdout, not an independent day.
3. The execution model assumes immediate aggressive fills and does not model
   queue position, partial fills, latency, fees, impact, or adverse selection.
4. Feature importance is descriptive and is not a causal attribution.
5. Independent ITCH reconstruction demonstrates engineering correctness, not
   cross-day predictive stability.

## Potential extensions

Additional dates and instruments, probability calibration, passive execution,
queue position, latency, partial fills, and market impact remain valid future
research directions. They are not required for the v1.0.0 engineering and
research release.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note, encoding="utf-8")
