from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from orderbook_research.features import add_snapshot_features
from orderbook_research.final_evaluation import (
    DEVELOPMENT_FRACTION,
    PRIMARY_CONFIDENCE_THRESHOLD,
    PRIMARY_HORIZON,
    SCHEMA_VERSION,
    attach_probabilities,
    build_frozen_candidate_models,
    build_linear_comparators,
    ensure_output_is_available,
    final_holdout_split,
    generate_final_figures,
    normalized_importance_frame,
    percentage_improvement,
    selected_feature_columns,
    validate_run_request,
    write_research_note,
    write_run_manifest,
)
from orderbook_research.io import load_lobster_pair
from orderbook_research.simulation import non_overlapping_signal_simulation
from orderbook_research.targets import add_event_horizon_targets
from orderbook_research.train_baseline import (
    class_distribution,
    detailed_classification_metrics,
    detailed_regression_metrics,
    detailed_simulation_metrics,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not serializable")


def _simulation_metrics(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    simulation, base = non_overlapping_signal_simulation(
        frame,
        horizon=horizon,
        confidence_threshold=PRIMARY_CONFIDENCE_THRESHOLD,
    )
    return simulation, detailed_simulation_metrics(simulation, base)


def run_final_evaluation(
    *,
    ticker: str,
    levels: int,
    horizon: int,
    purge_events: int,
    max_rows: int | None,
    table_directory: Path,
    figure_directory: Path,
    report_path: Path,
    model_directory: Path,
    project_root: Path,
    smoke: bool,
    confirm_final_holdout: bool,
    allow_rerun: bool,
) -> dict[str, Any]:
    if horizon != PRIMARY_HORIZON:
        raise ValueError(
            f"The frozen Phase D horizon is {PRIMARY_HORIZON} events."
        )
    validate_run_request(
        smoke=smoke,
        confirm_final_holdout=confirm_final_holdout,
        max_rows=max_rows,
    )

    manifest_path = table_directory / "final_evaluation_manifest.json"
    ensure_output_is_available(
        manifest_path,
        smoke=smoke,
        allow_rerun=allow_rerun,
    )

    data = load_lobster_pair(
        ticker=ticker,
        levels=levels,
        nrows=max_rows,
        scale_prices=True,
    )
    data = add_snapshot_features(data, levels=levels)
    data = add_event_horizon_targets(data, horizons=(10, 50, 100))

    split = final_holdout_split(
        len(data),
        development_fraction=DEVELOPMENT_FRACTION,
        purge_events=purge_events,
        horizon=horizon,
    )
    class_target = f"future_move_{horizon}"
    return_target = f"future_return_bps_{horizon}"
    features = selected_feature_columns(levels=levels)

    development = data.iloc[split.development_indices].copy()
    holdout = data.iloc[split.holdout_indices].copy()
    development = development.dropna(subset=[class_target, return_target])
    holdout = holdout.dropna(subset=[class_target, return_target])
    if development.empty or holdout.empty:
        raise ValueError("Development or holdout block is empty after filtering.")

    x_development = development[features]
    x_holdout = holdout[features]
    y_development_class = development[class_target].astype(int)
    y_holdout_class = holdout[class_target].astype(int)

    majority = DummyClassifier(strategy="most_frequent")
    logistic, ridge = build_linear_comparators()
    lightgbm_classifier, lightgbm_regressor = build_frozen_candidate_models()

    majority.fit(x_development, y_development_class)
    logistic.fit(x_development, y_development_class)
    ridge.fit(x_development, development[return_target])
    lightgbm_classifier.fit(x_development, y_development_class)
    lightgbm_regressor.fit(x_development, development[return_target])

    majority_prediction = majority.predict(x_holdout)
    logistic_prediction = logistic.predict(x_holdout)
    lightgbm_class_prediction = lightgbm_classifier.predict(x_holdout).astype(int)
    ridge_prediction = ridge.predict(x_holdout)
    lightgbm_return_prediction = lightgbm_regressor.predict(x_holdout)
    zero_prediction = np.zeros(len(holdout), dtype=float)

    classification_metrics = {
        "majority_classifier": detailed_classification_metrics(
            y_holdout_class, majority_prediction
        ),
        "balanced_logistic": detailed_classification_metrics(
            y_holdout_class, logistic_prediction
        ),
        "lightgbm_classifier": detailed_classification_metrics(
            y_holdout_class, lightgbm_class_prediction
        ),
    }
    regression_metrics = {
        "zero_return": detailed_regression_metrics(
            holdout[return_target], zero_prediction
        ),
        "ridge": detailed_regression_metrics(
            holdout[return_target], ridge_prediction
        ),
        "lightgbm_regressor": detailed_regression_metrics(
            holdout[return_target], lightgbm_return_prediction
        ),
    }

    logistic_frame = attach_probabilities(
        holdout,
        logistic.predict_proba(x_holdout),
        logistic.named_steps["model"].classes_,
    )
    lightgbm_frame = attach_probabilities(
        holdout,
        lightgbm_classifier.predict_proba(x_holdout),
        lightgbm_classifier.classes_,
    )
    logistic_simulation, logistic_economics = _simulation_metrics(
        logistic_frame, horizon
    )
    lightgbm_simulation, lightgbm_economics = _simulation_metrics(
        lightgbm_frame, horizon
    )
    economics = {
        "balanced_logistic": logistic_economics,
        "lightgbm_classifier": lightgbm_economics,
    }

    split_metadata = split.metadata()
    split_metadata.update(
        {
            "development_rows_after_target_filter": int(len(development)),
            "holdout_rows_after_target_filter": int(len(holdout)),
            "development_time_start_seconds": float(
                development["time_seconds"].min()
            ),
            "development_time_end_seconds": float(
                development["time_seconds"].max()
            ),
            "holdout_time_start_seconds": float(holdout["time_seconds"].min()),
            "holdout_time_end_seconds": float(holdout["time_seconds"].max()),
        }
    )

    configuration = {
        "result_schema_version": SCHEMA_VERSION,
        "run_type": "smoke" if smoke else "frozen_candidate_holdout",
        "ticker": ticker.upper(),
        "levels": levels,
        "horizon_events": horizon,
        "max_rows": max_rows,
        "rows_loaded": int(len(data)),
        "development_fraction": DEVELOPMENT_FRACTION,
        "exploratory_holdout_fraction": 1.0 - DEVELOPMENT_FRACTION,
        "requested_purge_events": purge_events,
        "effective_purge_events": split.purge_events,
        "confidence_threshold": PRIMARY_CONFIDENCE_THRESHOLD,
        "feature_set": "without_time",
        "feature_count": len(features),
        "feature_columns": features,
        "candidate_frozen_before_holdout_run": True,
        "holdout_status": (
            "not_used_for_phase_a_to_c_lightgbm_candidate_selection; "
            "previously_inspected_with_exploratory_linear_baselines"
        ),
        "no_holdout_threshold_or_feature_tuning": True,
        "cost_assumption": "full_estimated_quoted_spread_crossing_cost",
        "split_design": "first_80_percent_development_final_20_percent_holdout",
    }

    payload: dict[str, Any] = {
        "configuration": configuration,
        "split": split_metadata,
        "model_parameters": {
            "balanced_logistic": logistic.named_steps["model"].get_params(),
            "ridge": ridge.named_steps["model"].get_params(),
            "lightgbm_classifier": lightgbm_classifier.get_params(),
            "lightgbm_regressor": lightgbm_regressor.get_params(),
        },
        "class_distributions": {
            "development": class_distribution(development[class_target]),
            "holdout": class_distribution(holdout[class_target]),
        },
        "holdout": {
            "classification": classification_metrics,
            "regression": regression_metrics,
            "economics_at_threshold_0_10": economics,
        },
    }

    table_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    classification_rows = pd.DataFrame(
        [
            {
                "model": model,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "down_recall": metrics["per_class"]["down"]["recall"],
                "flat_recall": metrics["per_class"]["flat"]["recall"],
                "up_recall": metrics["per_class"]["up"]["recall"],
            }
            for model, metrics in classification_metrics.items()
        ]
    )
    regression_rows = pd.DataFrame(
        [
            {
                "model": model,
                **metrics,
                "mae_improvement_pct_vs_zero": percentage_improvement(
                    regression_metrics["zero_return"]["mae_bps"],
                    metrics["mae_bps"],
                ),
            }
            for model, metrics in regression_metrics.items()
        ]
    )
    economics_rows = pd.DataFrame(
        [{"model": model, **metrics} for model, metrics in economics.items()]
    )

    classification_rows.to_csv(
        table_directory / "final_holdout_classification.csv", index=False
    )
    regression_rows.to_csv(
        table_directory / "final_holdout_regression.csv", index=False
    )
    economics_rows.to_csv(
        table_directory / "final_holdout_economics.csv", index=False
    )
    lightgbm_simulation.to_csv(
        table_directory / "final_holdout_lightgbm_simulation.csv", index=False
    )
    logistic_simulation.to_csv(
        table_directory / "final_holdout_logistic_simulation.csv", index=False
    )

    classifier_importance = normalized_importance_frame(
        lightgbm_classifier, features, "lightgbm_classifier"
    )
    regressor_importance = normalized_importance_frame(
        lightgbm_regressor, features, "lightgbm_regressor"
    )
    importance = pd.concat(
        [classifier_importance, regressor_importance], ignore_index=True
    )
    importance.to_csv(
        table_directory / "final_holdout_feature_importance.csv", index=False
    )

    generated_figures = generate_final_figures(
        figure_directory=figure_directory,
        class_distribution=payload["class_distributions"]["holdout"],
        classification_rows=classification_rows,
        regression_rows=regression_rows,
        economics_rows=economics_rows,
        selected_simulation=lightgbm_simulation,
        feature_importance=importance,
    )
    payload["generated_figures"] = generated_figures

    metrics_path = table_directory / "final_holdout_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    write_research_note(path=report_path, payload=payload)

    joblib.dump(
        lightgbm_classifier,
        model_directory / "final_lightgbm_classifier_h50_no_time.joblib",
    )
    joblib.dump(
        lightgbm_regressor,
        model_directory / "final_lightgbm_regressor_h50_no_time.joblib",
    )

    manifest = write_run_manifest(
        manifest_path,
        configuration,
        project_root=project_root,
        run_type="smoke" if smoke else "frozen_candidate_holdout",
    )
    payload["manifest"] = manifest
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen 50-event LightGBM no-time candidate on the "
            "final chronological holdout and generate the Phase D report."
        )
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-final-holdout", action="store_true")
    parser.add_argument("--allow-rerun", action="store_true")
    parser.add_argument(
        "--table-directory",
        type=Path,
        default=Path("reports/tables"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("reports/figures/final"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/final_research_note.md"),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("models"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args()

    result = run_final_evaluation(
        ticker=args.ticker,
        levels=args.levels,
        horizon=args.horizon,
        purge_events=args.purge_events,
        max_rows=args.max_rows,
        table_directory=args.table_directory,
        figure_directory=args.figure_directory,
        report_path=args.report_path,
        model_directory=args.model_directory,
        project_root=args.project_root,
        smoke=args.smoke,
        confirm_final_holdout=args.confirm_final_holdout,
        allow_rerun=args.allow_rerun,
    )
    print(json.dumps(result["configuration"], indent=2))


if __name__ == "__main__":
    main()
