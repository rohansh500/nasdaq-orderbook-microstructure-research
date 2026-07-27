"use client";

import { useState } from "react";
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
import { execution } from "@/lib/data";

const thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
const percentage = (value: number) => `${(value * 100).toFixed(1)}%`;

export function ExecutionExplorer() {
  const [horizon, setHorizon] = useState(50);
  const [threshold, setThreshold] = useState(0.1);
  const [costFraction, setCostFraction] = useState(1);
  const sensitivity = execution.sensitivity as Array<Record<string, number>>;
  const regimes = execution.regimes as Array<Record<string, number | string>>;

  const baseRows = sensitivity.filter((row) => Number(row.horizon) === horizon && Number(row.cost_fraction) === 1);
  const current = baseRows.find((row) => Number(row.confidence_threshold) === threshold) ?? baseRows[0];
  if (!current) throw new Error("Execution sensitivity data is missing for the selected horizon.");
  const gross = Number(current.mean_gross_return_active_bps_mean);
  const fullCost = Number(current.mean_full_estimated_cost_active_bps_mean);
  const appliedCost = fullCost * costFraction;
  const net = gross - appliedCost;

  const thresholdChart = baseRows.map((row) => ({
    threshold: Number(row.confidence_threshold).toFixed(2),
    gross: Number(row.mean_gross_return_active_bps_mean),
    breakEven: Number(row.break_even_cost_fraction_mean),
    active: Number(row.active_signal_fraction_mean),
  }));

  const costChart = Array.from({ length: 21 }, (_, index) => {
    const fraction = index / 20;
    return {
      fraction,
      gross,
      cost: fullCost * fraction,
      net: gross - fullCost * fraction,
    };
  });

  const regimeRows = regimes.filter((row) => Number(row.horizon) === 50 && (row.regime_type === "depth" || row.regime_type === "volatility"));
  const regimeChart = regimeRows.map((row) => ({
    name: `${String(row.regime_type)}: ${String(row.regime_label)}`,
    gross: Number(row.mean_gross_return_active_bps_mean),
    breakEven: Number(row.break_even_cost_fraction_mean),
  }));

  return (
    <>
      <div className="grid two">
        <section className="card">
          <h2>Execution assumptions</h2>
          <div className="control-grid">
            <div className="control">
              <div className="control-row"><label>Horizon</label><output>{horizon} events</output></div>
              <div className="tabs" style={{ marginBottom: 0 }}>
                {[10, 50, 100].map((value) => <button key={value} className={`tab ${horizon === value ? "active" : ""}`} onClick={() => setHorizon(value)}>{value}</button>)}
              </div>
            </div>
            <div className="control">
              <label htmlFor="threshold">Confidence threshold</label>
              <select id="threshold" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))}>
                {thresholds.map((value) => <option key={value} value={value}>{value.toFixed(2)}</option>)}
              </select>
            </div>
            <div className="control" style={{ gridColumn: "1 / -1" }}>
              <div className="control-row"><label htmlFor="cost">Applied fraction of full estimated cost</label><output>{percentage(costFraction)}</output></div>
              <input id="cost" type="range" min={0} max={1} step={0.01} value={costFraction} onChange={(event) => setCostFraction(Number(event.target.value))} />
            </div>
          </div>
        </section>

        <section className="card">
          <div className="grid two">
            <div className="metric"><div className="metric-value">{percentage(Number(current.active_signal_fraction_mean))}</div><div className="metric-label">Active signal fraction</div></div>
            <div className="metric"><div className="metric-value">{percentage(Number(current.break_even_cost_fraction_mean))}</div><div className="metric-label">Break-even cost fraction</div></div>
            <div className="metric"><div className="metric-value">{gross.toFixed(3)}</div><div className="metric-label">Gross bps per signal</div></div>
            <div className="metric"><div className="metric-value">{net.toFixed(3)}</div><div className="metric-label">Net bps per signal</div></div>
          </div>
          <div className={`status ${net >= 0 ? "good" : "bad"}`} style={{ marginTop: 16 }}>
            {net >= 0 ? "Positive under this reduced-cost assumption." : "Negative after the selected execution-cost assumption."}
          </div>
        </section>
      </div>

      <section className="section grid two">
        <article className="card chart-card">
          <h2>Threshold selectivity</h2>
          <p className="muted">Higher confidence raised average gross edge but sharply reduced activity.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={thresholdChart}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                <XAxis dataKey="threshold" stroke="#9bb1c0" />
                <YAxis yAxisId="edge" stroke="#9bb1c0" />
                <YAxis yAxisId="activity" orientation="right" stroke="#9bb1c0" tickFormatter={(value) => percentage(value)} />
                <Tooltip />
                <Legend />
                <Line yAxisId="edge" type="monotone" dataKey="gross" name="Gross bps" stroke="#66e3d2" strokeWidth={3} />
                <Line yAxisId="activity" type="monotone" dataKey="active" name="Active fraction" stroke="#7aa8ff" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card chart-card">
          <h2>Cost sensitivity</h2>
          <p className="muted">The selected signal becomes negative well before the full quoted-spread assumption.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={costChart}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                <XAxis dataKey="fraction" stroke="#9bb1c0" tickFormatter={(value) => percentage(value)} />
                <YAxis stroke="#9bb1c0" />
                <Tooltip labelFormatter={(value) => `${percentage(Number(value))} of full cost`} formatter={(value) => `${Number(value).toFixed(3)} bps`} />
                <Legend />
                <Line type="monotone" dataKey="gross" name="Gross edge" stroke="#66e3d2" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="cost" name="Applied cost" stroke="#f4cf75" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="net" name="Net edge" stroke="#ff8c8c" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="section">
        <div className="section-head"><div><div className="eyebrow">Regime evidence</div><h2>Where the 50-event signal was strongest</h2></div><p className="muted">Low volatility and low displayed depth improved ranking and gross edge, but did not create a full-cost aggressive strategy.</p></div>
        <article className="card chart-card">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={regimeChart} margin={{ bottom: 55 }}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                <XAxis dataKey="name" stroke="#9bb1c0" angle={-25} textAnchor="end" interval={0} />
                <YAxis stroke="#9bb1c0" />
                <Tooltip />
                <Bar dataKey="gross" name="Gross bps" fill="#66e3d2" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>
    </>
  );
}
