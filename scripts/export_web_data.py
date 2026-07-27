from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OUTPUT = ROOT / "web" / "data"


def _records(path: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(TABLES / path)
    return json.loads(frame.to_json(orient="records"))


def _read_json(path: str) -> dict[str, Any]:
    return json.loads((TABLES / path).read_text(encoding="utf-8"))


def _write(name: str, payload: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def _synthetic_replay() -> dict[str, Any]:
    events = [
        {
            "index": 0,
            "time": "09:30:00.000001",
            "message": "A",
            "event": "Add bid",
            "orderId": "1001",
            "description": "Add 100 shares at 100.00 on the bid.",
            "bid": [[100.00, 100], [99.99, 180], [99.98, 220]],
            "ask": [[100.01, 120], [100.02, 190], [100.03, 240]],
        },
        {
            "index": 1,
            "time": "09:30:00.000002",
            "message": "F",
            "event": "Add ask with attribution",
            "orderId": "2001",
            "description": "Add 120 displayed shares at 100.01 on the ask.",
            "bid": [[100.00, 100], [99.99, 180], [99.98, 220]],
            "ask": [[100.01, 120], [100.02, 190], [100.03, 240]],
        },
        {
            "index": 2,
            "time": "09:30:00.000003",
            "message": "X",
            "event": "Partial cancel",
            "orderId": "1001",
            "description": "Cancel 20 shares from order 1001; 80 remain.",
            "bid": [[100.00, 80], [99.99, 180], [99.98, 220]],
            "ask": [[100.01, 120], [100.02, 190], [100.03, 240]],
        },
        {
            "index": 3,
            "time": "09:30:00.000004",
            "message": "E",
            "event": "Visible execution",
            "orderId": "2001",
            "description": "Execute 40 shares against the resting ask; 80 remain.",
            "bid": [[100.00, 80], [99.99, 180], [99.98, 220]],
            "ask": [[100.01, 80], [100.02, 190], [100.03, 240]],
        },
        {
            "index": 4,
            "time": "09:30:00.000005",
            "message": "U",
            "event": "Cancel-replace",
            "orderId": "1002",
            "description": "Replace bid order 1001 with 60 shares at 99.99.",
            "bid": [[99.99, 240], [99.98, 220], [99.97, 260]],
            "ask": [[100.01, 80], [100.02, 190], [100.03, 240]],
        },
        {
            "index": 5,
            "time": "09:30:00.000006",
            "message": "A",
            "event": "Add bid",
            "orderId": "1003",
            "description": "Add a new 70-share bid at 100.00.",
            "bid": [[100.00, 70], [99.99, 240], [99.98, 220]],
            "ask": [[100.01, 80], [100.02, 190], [100.03, 240]],
        },
        {
            "index": 6,
            "time": "09:30:00.000007",
            "message": "D",
            "event": "Delete ask",
            "orderId": "2001",
            "description": "Delete the remaining ask order 2001.",
            "bid": [[100.00, 70], [99.99, 240], [99.98, 220]],
            "ask": [[100.02, 190], [100.03, 240], [100.04, 260]],
        },
        {
            "index": 7,
            "time": "09:30:00.000008",
            "message": "A",
            "event": "Add ask",
            "orderId": "2002",
            "description": "Add 50 shares at 100.02 on the ask.",
            "bid": [[100.00, 70], [99.99, 240], [99.98, 220]],
            "ask": [[100.02, 240], [100.03, 240], [100.04, 260]],
        },
        {
            "index": 8,
            "time": "09:30:00.000009",
            "message": "C",
            "event": "Execution with price",
            "orderId": "2002",
            "description": "Execute 10 shares with a reported execution price.",
            "bid": [[100.00, 70], [99.99, 240], [99.98, 220]],
            "ask": [[100.02, 230], [100.03, 240], [100.04, 260]],
        },
    ]
    return {
        "source": "Synthetic lifecycle fixture generated from the repository's ITCH test protocol.",
        "events": events,
    }


def main() -> None:
    final_metrics = _read_json("final_holdout_metrics.json")
    itch = _read_json("phase_e_itch_AAPL_reconstruction_metrics.json")

    _write(
        "headline-results.json",
        {
            "metrics": [
                {"label": "ITCH messages", "value": "368.4M", "detail": "market-wide records"},
                {
                    "label": "AAPL transitions",
                    "value": "1.66M",
                    "detail": "independently reconstructed",
                },
                {
                    "label": "Integrity failures",
                    "value": "0",
                    "detail": "tracked reconstruction invariants",
                },
                {"label": "Holdout rank IC", "value": "0.363", "detail": "50-event LightGBM"},
                {
                    "label": "MAE improvement",
                    "value": "6.67%",
                    "detail": "versus zero-return baseline",
                },
                {
                    "label": "Gross vs cost",
                    "value": "0.289 / 1.854",
                    "detail": "bps per active signal",
                },
            ],
            "conclusion": (
                "The model captured statistically meaningful short-horizon price formation, "
                "but the observed gross edge was insufficient to cover aggressive execution costs."
            ),
            "configuration": final_metrics["configuration"],
        },
    )

    _write(
        "validation.json",
        {
            "walkForward": _records("walk_forward_summary.csv"),
            "walkForwardFolds": {
                "10": _records("walk_forward_h10_folds.csv"),
                "50": _records("walk_forward_h50_folds.csv"),
                "100": _records("walk_forward_h100_folds.csv"),
            },
            "bootstrap": _records("bootstrap_summary.csv"),
            "modelComparison": _records("phase_c_model_summary.csv"),
            "ablation": _records("phase_c_ablation_summary.csv"),
            "featureImportance": _records("phase_c_feature_importance_summary.csv"),
            "holdout": {
                "classification": _records("final_holdout_classification.csv"),
                "regression": _records("final_holdout_regression.csv"),
            },
        },
    )

    _write(
        "execution.json",
        {
            "sensitivity": _records("phase_b_sensitivity_summary.csv"),
            "regimes": _records("phase_b_regime_summary.csv"),
            "spreadConfidence": _records("phase_b_spread_confidence_summary.csv"),
            "holdout": _records("final_holdout_economics.csv"),
        },
    )

    message_counts = _records("phase_e_itch_AAPL_message_counts.csv")
    _write(
        "engineering.json",
        {
            "benchmark": itch.get("benchmark", {}),
            "binaryFile": itch.get("binary_file", {}),
            "reconstruction": itch.get("reconstruction", {}),
            "integrity": itch.get("final_integrity", {}),
            "messageCounts": message_counts,
            "headline": {
                "records": 368_366_634,
                "payloadGb": 10.51,
                "targetEvents": 1_656_597,
                "maxOrders": 42_774,
                "throughput": 40_470,
                "peakMb": 303.6,
            },
        },
    )

    _write("synthetic-replay.json", _synthetic_replay())

    print(f"Wrote website data to {OUTPUT}")


if __name__ == "__main__":
    main()
