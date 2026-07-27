"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { validation } from "@/lib/data";

const percentage = (value: number) => `${(value * 100).toFixed(1)}%`;

export function ValidationExplorer() {
  const [horizon, setHorizon] = useState(50);
  const modelRows = validation.modelComparison as Array<Record<string, number>>;
  const ablationRows = validation.ablation as Array<Record<string, number | string>>;
  const foldRows = (validation.walkForwardFolds as Record<string, Array<Record<string, number>>>)[String(horizon)];
  const selected = modelRows.find((row) => Number(row.horizon) === horizon)!;
  const holdoutClassification = validation.holdout.classification as Array<Record<string, number | string>>;
  const holdoutRegression = validation.holdout.regression as Array<Record<string, number | string>>;

  const comparison = useMemo(() => modelRows.map((row) => ({
    horizon: `${row.horizon} events`,
    logistic: Number(row.logistic_balanced_accuracy_mean),
    lightgbm: Number(row.lightgbm_balanced_accuracy_mean),
    ridgeIc: Number(row.ridge_rank_ic_mean),
    lightgbmIc: Number(row.lightgbm_rank_ic_mean),
  })), [modelRows]);

  const ablation = ablationRows.map((row) => ({
    name: String(row.feature_set).replaceAll("_", " "),
    rankIc: Number(row.rank_ic_mean),
    mae: Number(row.mae_improvement_pct_mean),
  }));

  return (
    <>
      <div className="tabs">
        {[10, 50, 100].map((value) => (
          <button className={`tab ${horizon === value ? "active" : ""}`} key={value} onClick={() => setHorizon(value)}>{value}-event horizon</button>
        ))}
      </div>

      <div className="grid metrics">
        <div className="metric"><div className="metric-value">{percentage(Number(selected.lightgbm_balanced_accuracy_mean))}</div><div className="metric-label">LightGBM balanced accuracy</div></div>
        <div className="metric"><div className="metric-value">{Number(selected.lightgbm_rank_ic_mean).toFixed(3)}</div><div className="metric-label">LightGBM walk-forward rank IC</div></div>
        <div className="metric"><div className="metric-value">{Number(selected.lightgbm_mae_improvement_pct_mean).toFixed(2)}%</div><div className="metric-label">MAE improvement versus zero</div></div>
      </div>

      <section className="section grid two">
        <article className="card chart-card">
          <h2>Classifier comparison</h2>
          <p className="muted">Mean balanced accuracy across five purged expanding-window folds.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparison}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                <XAxis dataKey="horizon" stroke="#9bb1c0" />
                <YAxis stroke="#9bb1c0" tickFormatter={(value) => percentage(value)} domain={[0.3, 0.5]} />
                <Tooltip formatter={(value) => percentage(Number(value))} />
                <Legend />
                <Bar dataKey="logistic" name="Balanced logistic" fill="#7aa8ff" radius={[6, 6, 0, 0]} />
                <Bar dataKey="lightgbm" name="LightGBM" fill="#66e3d2" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card chart-card">
          <h2>Return ranking</h2>
          <p className="muted">LightGBM improved rank IC over Ridge in every fold at all three horizons.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={comparison}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                <XAxis dataKey="horizon" stroke="#9bb1c0" />
                <YAxis stroke="#9bb1c0" domain={[0, 0.32]} />
                <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                <Legend />
                <Line type="monotone" dataKey="ridgeIc" name="Ridge rank IC" stroke="#7aa8ff" strokeWidth={3} />
                <Line type="monotone" dataKey="lightgbmIc" name="LightGBM rank IC" stroke="#66e3d2" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="section">
        <div className="section-head"><div><div className="eyebrow">Feature-family ablation</div><h2>Event flow carried the signal</h2></div><p className="muted">Removing a feature family and retraining provides stronger evidence than reading gain importance alone.</p></div>
        <article className="card chart-card">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ablation} layout="vertical" margin={{ left: 28 }}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" horizontal={false} />
                <XAxis type="number" stroke="#9bb1c0" domain={[0, 0.3]} />
                <YAxis type="category" dataKey="name" stroke="#9bb1c0" width={150} />
                <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                <Bar dataKey="rankIc" name="50-event rank IC" fill="#66e3d2" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="section grid two">
        <article className="card">
          <div className="eyebrow">Fold-level evidence</div>
          <h2>{horizon}-event walk-forward folds</h2>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Fold</th><th>Balanced accuracy</th><th>Rank IC</th><th>Gross bps</th><th>Net bps</th></tr></thead>
              <tbody>
                {foldRows.map((row, index) => (
                  <tr key={index}>
                    <td>{Number(row.fold)}</td>
                    <td>{percentage(Number(row.balanced_accuracy))}</td>
                    <td>{Number(row.ridge_rank_ic).toFixed(3)}</td>
                    <td>{Number(row.mean_gross_return_active_bps).toFixed(3)}</td>
                    <td>{Number(row.mean_net_return_active_bps).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card">
          <div className="eyebrow">Frozen holdout</div>
          <h2>Configuration-level final evaluation</h2>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Model</th><th>Metric</th><th>Result</th></tr></thead>
              <tbody>
                {holdoutClassification.filter((row) => row.model === "lightgbm_classifier").map((row) => (
                  <tr key="classification"><td>LightGBM classifier</td><td>Balanced accuracy</td><td>{percentage(Number(row.balanced_accuracy))}</td></tr>
                ))}
                {holdoutRegression.filter((row) => row.model === "lightgbm_regressor").flatMap((row) => [
                  <tr key="ic"><td>LightGBM regressor</td><td>Rank IC</td><td>{Number(row.rank_ic).toFixed(3)}</td></tr>,
                  <tr key="mae"><td>LightGBM regressor</td><td>MAE improvement</td><td>{Number(row.mae_improvement_pct_vs_zero).toFixed(2)}%</td></tr>,
                  <tr key="direction"><td>LightGBM regressor</td><td>Non-zero direction</td><td>{percentage(Number(row.nonzero_directional_accuracy))}</td></tr>,
                ])}
              </tbody>
            </table>
          </div>
          <div className="status warn" style={{ marginTop: 16 }}>The holdout is one late-session block from one AAPL day, not cross-day proof.</div>
        </article>
      </section>
    </>
  );
}
