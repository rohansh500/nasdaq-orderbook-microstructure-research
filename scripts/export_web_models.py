from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from orderbook_research.features import add_snapshot_features
from orderbook_research.final_evaluation import (
    DEVELOPMENT_FRACTION,
    PRIMARY_HORIZON,
    final_holdout_split,
    selected_feature_columns,
)
from orderbook_research.io import load_lobster_pair
from orderbook_research.targets import add_event_horizon_targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_features(
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        finite = values.dropna()
        if finite.empty:
            raise ValueError(f"Feature {feature!r} has no finite development values.")
        profile[feature] = {
            "p01": float(finite.quantile(0.01)),
            "median": float(finite.median()),
            "p99": float(finite.quantile(0.99)),
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
        }
    return profile


def _load_development_matrix(
    *,
    ticker: str,
    levels: int,
    purge_events: int,
    parity_rows: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    data = load_lobster_pair(ticker=ticker, levels=levels, scale_prices=True)
    data = add_snapshot_features(data, levels=levels)
    data = add_event_horizon_targets(data, horizons=(PRIMARY_HORIZON,))

    split = final_holdout_split(
        len(data),
        development_fraction=DEVELOPMENT_FRACTION,
        purge_events=purge_events,
        horizon=PRIMARY_HORIZON,
    )
    features = selected_feature_columns(levels=levels)
    target = f"future_return_bps_{PRIMARY_HORIZON}"
    development = data.iloc[split.development_indices].dropna(subset=[target]).copy()
    profile = _profile_features(development, features)

    usable = development[features].replace([np.inf, -np.inf], np.nan)
    if len(usable) > parity_rows:
        usable = usable.sample(n=parity_rows, random_state=42).sort_index()
    return usable, profile


def _validate_model_features(model: Any, expected: list[str], name: str) -> None:
    fitted = getattr(model, "feature_name_", None)
    if fitted is None:
        return
    fitted_names = [str(value) for value in fitted]
    if fitted_names != expected:
        raise ValueError(
            f"{name} feature order does not match the frozen manifest.\n"
            f"Expected: {expected}\nActual: {fitted_names}"
        )


def _compact_tree(node: dict[str, Any]) -> dict[str, Any]:
    if "leaf_value" in node:
        return {"v": float(node["leaf_value"])}

    decision_type = str(node.get("decision_type", "<="))
    if decision_type != "<=":
        raise ValueError(
            "The browser evaluator currently supports numerical LightGBM "
            f"splits only; found decision_type={decision_type!r}."
        )

    return {
        "f": int(node["split_feature"]),
        "t": float(node["threshold"]),
        "d": bool(node.get("default_left", True)),
        "m": str(node.get("missing_type", "None")),
        "l": _compact_tree(node["left_child"]),
        "r": _compact_tree(node["right_child"]),
    }


def _tree_value(node: dict[str, Any], row: np.ndarray) -> float:
    current = node
    while "v" not in current:
        value = float(row[int(current["f"])])
        missing_type = str(current.get("m", "None"))
        is_missing = np.isnan(value) or (missing_type == "Zero" and value == 0.0)
        if is_missing:
            go_left = bool(current["d"])
        else:
            go_left = value <= float(current["t"])
        current = current["l"] if go_left else current["r"]
    return float(current["v"])


def _raw_tree_predictions(
    matrix: np.ndarray,
    trees: list[dict[str, Any]],
    outputs: int,
) -> np.ndarray:
    result = np.zeros((len(matrix), outputs), dtype=np.float64)
    for tree_index, tree in enumerate(trees):
        output_index = tree_index % outputs
        result[:, output_index] += np.fromiter(
            (_tree_value(tree, row) for row in matrix),
            dtype=np.float64,
            count=len(matrix),
        )
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def _compact_model(model: Any, kind: str) -> dict[str, Any]:
    booster = model.booster_
    dump = booster.dump_model()
    trees = [_compact_tree(item["tree_structure"]) for item in dump["tree_info"]]

    if bool(dump.get("average_output", False)):
        raise ValueError("average_output LightGBM models are not supported.")

    payload: dict[str, Any] = {
        "format": "lightgbm-compact-v1",
        "kind": kind,
        "featureCount": int(booster.num_feature()),
        "treeCount": len(trees),
        "trees": trees,
    }
    if kind == "multiclass":
        payload["numClass"] = int(len(model.classes_))
        payload["classLabels"] = [int(value) for value in model.classes_]
    return payload


def export_models(
    *,
    classifier_path: Path,
    regressor_path: Path,
    final_manifest_path: Path,
    output_directory: Path,
    ticker: str,
    levels: int,
    purge_events: int,
    parity_rows: int,
    tolerance: float,
) -> dict[str, Any]:
    if not classifier_path.exists() or not regressor_path.exists():
        raise FileNotFoundError(
            "Frozen joblib models were not found. Re-run Phase D locally or pass explicit paths."
        )

    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    configuration = final_manifest["configuration"]
    expected_features = [str(value) for value in configuration["feature_columns"]]
    selected = selected_feature_columns(levels=levels)
    if expected_features != selected:
        raise ValueError("Repository feature order differs from final_evaluation_manifest.json.")

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    _validate_model_features(classifier, expected_features, "classifier")
    _validate_model_features(regressor, expected_features, "regressor")

    classifier_payload = _compact_model(classifier, "multiclass")
    regressor_payload = _compact_model(regressor, "regression")

    output_directory.mkdir(parents=True, exist_ok=True)
    classifier_output = output_directory / "classifier_h50_no_time.json"
    regressor_output = output_directory / "regressor_h50_no_time.json"
    classifier_output.write_text(
        json.dumps(classifier_payload, separators=(",", ":")),
        encoding="utf-8",
    )
    regressor_output.write_text(
        json.dumps(regressor_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    parity_frame, profile = _load_development_matrix(
        ticker=ticker,
        levels=levels,
        purge_events=purge_events,
        parity_rows=parity_rows,
    )
    matrix = parity_frame.to_numpy(dtype=np.float64)

    classifier_raw = _raw_tree_predictions(
        matrix,
        classifier_payload["trees"],
        outputs=len(classifier.classes_),
    )
    browser_probabilities = _softmax(classifier_raw)
    browser_regression = _raw_tree_predictions(
        matrix,
        regressor_payload["trees"],
        outputs=1,
    ).reshape(-1)

    python_probabilities = classifier.predict_proba(parity_frame)
    python_regression = regressor.predict(parity_frame)

    np.testing.assert_allclose(
        python_probabilities,
        browser_probabilities,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        python_regression,
        browser_regression,
        rtol=tolerance,
        atol=tolerance,
    )

    browser_positions = np.argmax(browser_probabilities, axis=1)
    browser_labels = np.asarray(classifier.classes_)[browser_positions]
    np.testing.assert_array_equal(classifier.predict(parity_frame), browser_labels)

    payload = {
        "modelsReady": True,
        "modelVersion": "1.1.0",
        "sourceConfigurationHash": final_manifest.get("configuration_hash"),
        "sourceGitCommit": final_manifest.get("git_commit"),
        "ticker": configuration["ticker"],
        "horizonEvents": configuration["horizon_events"],
        "featureSet": configuration["feature_set"],
        "featureCount": len(expected_features),
        "featureNames": expected_features,
        "featureRanges": profile,
        "inputType": "float64",
        "classLabels": [int(value) for value in classifier.classes_],
        "classifier": {
            "path": "/models/classifier_h50_no_time.json",
            "sha256": _sha256(classifier_output),
            "bytes": classifier_output.stat().st_size,
        },
        "regressor": {
            "path": "/models/regressor_h50_no_time.json",
            "sha256": _sha256(regressor_output),
            "bytes": regressor_output.stat().st_size,
        },
        "parity": {
            "rows": len(matrix),
            "rtol": tolerance,
            "atol": tolerance,
            "classifierMaximumAbsoluteDifference": float(
                np.max(np.abs(python_probabilities - browser_probabilities))
            ),
            "regressorMaximumAbsoluteDifference": float(
                np.max(np.abs(python_regression - browser_regression))
            ),
            "classLabelsExact": True,
        },
        "scope": (
            "Offline educational model trained on one historical AAPL stock-day. "
            "Synthetic scenario output is not a live forecast or trading "
            "recommendation."
        ),
    }
    manifest_output = output_directory / "feature_manifest.json"
    manifest_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Classifier JSON: {classifier_output} ({classifier_output.stat().st_size:,} bytes)")
    print(f"Regressor JSON: {regressor_output} ({regressor_output.stat().st_size:,} bytes)")
    print(f"Manifest: {manifest_output}")
    print(
        "Parity max absolute differences:",
        payload["parity"]["classifierMaximumAbsoluteDifference"],
        payload["parity"]["regressorMaximumAbsoluteDifference"],
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the frozen Phase D LightGBM models as compact, "
            "browser-evaluable JSON artifacts."
        )
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("models/final_lightgbm_classifier_h50_no_time.joblib"),
    )
    parser.add_argument(
        "--regressor",
        type=Path,
        default=Path("models/final_lightgbm_regressor_h50_no_time.joblib"),
    )
    parser.add_argument(
        "--final-manifest",
        type=Path,
        default=Path("reports/tables/final_evaluation_manifest.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("web/public/models"),
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--purge-events", type=int, default=100)
    parser.add_argument("--parity-rows", type=int, default=5_000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    export_models(
        classifier_path=args.classifier,
        regressor_path=args.regressor,
        final_manifest_path=args.final_manifest,
        output_directory=args.output_directory,
        ticker=args.ticker,
        levels=args.levels,
        purge_events=args.purge_events,
        parity_rows=args.parity_rows,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
