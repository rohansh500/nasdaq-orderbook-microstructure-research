from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
MODELS = WEB / "public" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    final_manifest = json.loads(
        (ROOT / "reports" / "tables" / "final_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    web_manifest = json.loads((MODELS / "feature_manifest.json").read_text(encoding="utf-8"))
    expected = final_manifest["configuration"]["feature_columns"]
    if web_manifest["featureNames"] != expected:
        raise SystemExit("Web feature order differs from the frozen manifest.")
    if web_manifest["featureCount"] != len(expected):
        raise SystemExit("Web feature count is inconsistent.")

    required_data = {
        "headline-results.json",
        "validation.json",
        "execution.json",
        "engineering.json",
        "synthetic-replay.json",
    }
    missing_data = sorted(name for name in required_data if not (WEB / "data" / name).exists())
    if missing_data:
        raise SystemExit(f"Missing generated website data: {missing_data}")

    if not web_manifest.get("modelsReady"):
        print("Website data and feature schema are valid; model export is pending.")
        return

    for key in ("classifier", "regressor"):
        item = web_manifest[key]
        path = MODELS / Path(item["path"]).name
        if not path.exists():
            raise SystemExit(f"Missing {key} model: {path}")
        if _sha256(path) != item["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {path}")
        if path.stat().st_size != item["bytes"]:
            raise SystemExit(f"Byte-size mismatch for {path}")
        if path.stat().st_size > 25 * 1024 * 1024:
            raise SystemExit(f"Browser model is unexpectedly large: {path}")
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("format") != "lightgbm-compact-v1":
            raise SystemExit(f"Unexpected browser model format: {path}")

    parity = web_manifest.get("parity", {})
    if not parity.get("classLabelsExact"):
        raise SystemExit("Classifier label parity is not verified.")
    print("Website data, model hashes, feature order and parity are valid.")


if __name__ == "__main__":
    main()
