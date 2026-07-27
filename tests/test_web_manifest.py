from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_web_feature_order_matches_frozen_manifest() -> None:
    final_manifest = _json(ROOT / "reports/tables/final_evaluation_manifest.json")
    web_manifest = _json(ROOT / "web/public/models/feature_manifest.json")
    configuration = final_manifest["configuration"]
    assert isinstance(configuration, dict)
    assert web_manifest["featureNames"] == configuration["feature_columns"]
    assert web_manifest["featureCount"] == 31


def test_web_uses_only_aggregate_or_synthetic_data() -> None:
    web_root = ROOT / "web"
    forbidden_suffixes = {".csv", ".parquet", ".joblib", ".pkl", ".itch", ".gz"}
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in web_root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    assert offenders == []


def test_headline_metrics_match_final_results() -> None:
    headline = _json(ROOT / "web/data/headline-results.json")
    metrics = _json(ROOT / "reports/tables/final_holdout_metrics.json")
    configuration = metrics["configuration"]
    assert isinstance(configuration, dict)
    assert headline["configuration"] == configuration
