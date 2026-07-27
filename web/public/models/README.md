# Browser model artifacts

`feature_manifest.json` is committed in placeholder mode so the website can
build before model export. Run `scripts/setup_web_export.ps1` from the
repository root to generate and parity-check:

- `classifier_h50_no_time.json`
- `regressor_h50_no_time.json`
- the completed `feature_manifest.json`

The source joblib files remain local under `models/` and are never published.
The compact JSON models preserve LightGBM split thresholds as JavaScript
double-precision numbers and avoid ONNX tree-ensemble float conversion.
