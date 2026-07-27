"use client";

import { useEffect, useMemo, useState } from "react";
import { calculateFeatures, FEATURE_NAMES, vectorInOrder } from "@/lib/market/feature-engine";
import { generateScenario, presets, type ScenarioControls } from "@/lib/market/scenario-generator";
import { loadManifest, predictWithFrozenModels } from "@/lib/model/inference";
import type { ModelManifest, ModelPrediction } from "@/lib/model/types";

const defaultPreset = "Balanced market";
const labels: Record<number, string> = { [-1]: "Down", 0: "Flat", 1: "Up" };

function Slider({
  label,
  value,
  minimum,
  maximum,
  step,
  onChange,
  format = (item) => item.toFixed(2),
}: {
  label: string;
  value: number;
  minimum: number;
  maximum: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <div className="control">
      <div className="control-row"><label>{label}</label><output>{format(value)}</output></div>
      <input type="range" min={minimum} max={maximum} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

export function PredictionLab() {
  const [controls, setControls] = useState<ScenarioControls>(presets[defaultPreset]);
  const [preset, setPreset] = useState(defaultPreset);
  const [manifest, setManifest] = useState<ModelManifest | null>(null);
  const [prediction, setPrediction] = useState<ModelPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadManifest().then(setManifest).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Could not load model manifest."));
  }, []);

  const rows = useMemo(() => generateScenario(controls), [controls]);
  const features = useMemo(() => calculateFeatures(rows), [rows]);
  const featureNames = useMemo(
    () => manifest?.featureNames ?? [...FEATURE_NAMES],
    [manifest],
  );

  const vector = useMemo(
    () => vectorInOrder(features, featureNames),
    [features, featureNames],
  );
  const rangeRows = featureNames.map((name, index) => {
    const range = manifest?.featureRanges?.[name];
    const value = vector[index];
    const out = range ? value < range.p01 || value > range.p99 : false;
    return { name, value, range, out };
  });
  const outOfRange = rangeRows.filter((item) => item.out).length;

  const update = (key: keyof ScenarioControls, value: number) => {
    setControls((current) => ({ ...current, [key]: value }));
    setPrediction(null);
    setPreset("Custom scenario");
  };

  const selectPreset = (name: string) => {
    if (!(name in presets)) return;
    setPreset(name);
    setControls(presets[name]);
    setPrediction(null);
    setError(null);
  };

  const runPrediction = async () => {
    if (!manifest?.modelsReady) {
      setError("The browser model artifacts have not been exported. Run the repository export command first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setPrediction(await predictWithFrozenModels(manifest, vector));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Inference failed.");
    } finally {
      setLoading(false);
    }
  };

  const confidence = prediction ? Math.max(...prediction.probabilities) : null;
  const crossingCost = features.spread_bps;
  const netAfterCost = prediction ? Math.abs(prediction.predictedReturnBps) - crossingCost : null;

  return (
    <>
      <div className="grid two">
        <section className="card">
          <div className="control-row"><h2>Market mechanics</h2><span className="eyebrow">150-event scenario</span></div>
          <div className="control" style={{ marginBottom: 14 }}>
            <label htmlFor="preset">Scenario preset</label>
            <select id="preset" value={preset} onChange={(event) => selectPreset(event.target.value)}>
              {Object.keys(presets).map((name) => <option key={name}>{name}</option>)}
              {preset === "Custom scenario" ? <option>Custom scenario</option> : null}
            </select>
          </div>
          <div className="control-grid">
            <Slider label="Quoted spread" value={controls.spreadBps} minimum={0.7} maximum={6} step={0.1} onChange={(value) => update("spreadBps", value)} format={(value) => `${value.toFixed(1)} bps`} />
            <Slider label="Displayed liquidity" value={controls.liquidity} minimum={0.05} maximum={1} step={0.01} onChange={(value) => update("liquidity", value)} />
            <Slider label="Queue skew" value={controls.queueSkew} minimum={-0.9} maximum={0.9} step={0.02} onChange={(value) => update("queueSkew", value)} />
            <Slider label="Trade pressure" value={controls.tradePressure} minimum={-1} maximum={1} step={0.02} onChange={(value) => update("tradePressure", value)} />
            <Slider label="Cancel pressure" value={controls.cancelPressure} minimum={-1} maximum={1} step={0.02} onChange={(value) => update("cancelPressure", value)} />
            <Slider label="Volatility" value={controls.volatility} minimum={0.02} maximum={1} step={0.01} onChange={(value) => update("volatility", value)} />
            <Slider label="Event activity" value={controls.activity} minimum={0.05} maximum={1} step={0.01} onChange={(value) => update("activity", value)} />
            <Slider label="Deterministic seed" value={controls.seed} minimum={1} maximum={99} step={1} onChange={(value) => update("seed", value)} format={(value) => value.toFixed(0)} />
          </div>
          <div className="actions">
            <button className="button primary" onClick={runPrediction} disabled={loading}>{loading ? "Running in browser..." : "Run frozen models"}</button>
            <button className="button" onClick={() => selectPreset(defaultPreset)}>Reset</button>
          </div>
          {!manifest?.modelsReady ? (
            <div className="status warn" style={{ marginTop: 16 }}>
              Model assets are not present yet. Run <span className="mono">python scripts/export_web_models.py</span>, then commit the generated compact model files and manifest.
            </div>
          ) : null}
          {error ? <div className="status bad" style={{ marginTop: 16 }}>{error}</div> : null}
        </section>

        <section className="card">
          <div className="eyebrow">Frozen model response</div>
          <h2>{prediction ? labels[prediction.predictedLabel] : "Awaiting inference"}</h2>
          <p className="muted">
            The generator creates 150 coherent snapshots and events. The first 100 warm the rolling windows; the final state produces all 31 features in the frozen training order.
          </p>
          <div className="grid two">
            <div className="metric"><div className="metric-value">{confidence === null ? "-" : `${(confidence * 100).toFixed(1)}%`}</div><div className="metric-label">Direction confidence</div></div>
            <div className="metric"><div className="metric-value">{prediction ? `${prediction.predictedReturnBps >= 0 ? "+" : ""}${prediction.predictedReturnBps.toFixed(3)}` : "-"}</div><div className="metric-label">Predicted return, bps</div></div>
            <div className="metric"><div className="metric-value">{crossingCost.toFixed(3)}</div><div className="metric-label">Illustrative full-spread cost, bps</div></div>
            <div className="metric"><div className="metric-value">{netAfterCost === null ? "-" : `${netAfterCost >= 0 ? "+" : ""}${netAfterCost.toFixed(3)}`}</div><div className="metric-label">Absolute move less cost, bps</div></div>
          </div>
          {prediction ? (
            <div className={`status ${netAfterCost !== null && netAfterCost > 0 ? "good" : "warn"}`} style={{ marginTop: 16 }}>
              {netAfterCost !== null && netAfterCost > 0
                ? "The synthetic predicted move exceeds the current full-spread assumption. This is still not a fill or profitability claim."
                : "The predicted move does not cover the current full-spread assumption, matching the project's central economic finding."}
            </div>
          ) : null}
          <div className={`status ${outOfRange === 0 ? "good" : "warn"}`} style={{ marginTop: 12 }}>
            {manifest?.modelsReady
              ? outOfRange === 0
                ? "All generated features lie inside the development p01-p99 ranges."
                : `${outOfRange} generated features are outside development p01-p99 ranges; treat the response as extrapolative.`
              : "Development ranges will appear after the verified export step."}
          </div>
          <p className="muted" style={{ marginTop: 16 }}>
            Offline educational model trained on one historical AAPL stock-day. This is not a live AAPL forecast, order recommendation, or investment advice.
          </p>
        </section>
      </div>

      <section className="section">
        <div className="section-head"><div><div className="eyebrow">Transparency</div><h2>All derived model inputs</h2></div><p className="muted">No hidden averages are inserted. Every feature is computed from the generated event and book history.</p></div>
        <div className="feature-list">
          <div className="feature-row"><strong>Feature</strong><strong>Value</strong><strong>Development p01-p99</strong><strong>Status</strong></div>
          {rangeRows.map(({ name, value, range, out }) => (
            <div className="feature-row" key={name}>
              <span className="mono">{name}</span>
              <span className="mono">{value.toPrecision(6)}</span>
              <span className="mono">{range ? `${range.p01.toPrecision(4)} to ${range.p99.toPrecision(4)}` : "export pending"}</span>
              <span style={{ color: out ? "var(--warning)" : "var(--positive)" }}>{range ? (out ? "Outside" : "Inside") : "Pending"}</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
