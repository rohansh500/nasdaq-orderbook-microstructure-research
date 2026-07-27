"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { engineering } from "@/lib/data";

export function EngineeringExplorer() {
  const headline = engineering.headline;
  const reconstruction = engineering.reconstruction as Record<string, number>;
  const integrity = engineering.integrity as Record<string, boolean | number>;
  const messageCounts = (engineering.messageCounts as Array<{ message_type: string; count: number }>)
    .sort((left, right) => right.count - left.count)
    .slice(0, 8)
    .map((row) => ({ type: row.message_type, countMillions: row.count / 1_000_000 }));

  const invariants = [
    ["Duplicate order references", reconstruction.duplicate_order_references],
    ["Missing target references", reconstruction.missing_target_order_references],
    ["Share underflows", reconstruction.share_underflows],
    ["Timestamp reversals", reconstruction.timestamp_monotonicity_violations],
    ["Crossed snapshots", reconstruction.crossed_book_rows],
    ["Locked snapshots", reconstruction.locked_book_rows],
    ["Final open orders", reconstruction.final_open_orders],
  ] as const;

  return (
    <>
      <div className="grid metrics">
        <div className="metric"><div className="metric-value">{(headline.records / 1_000_000).toFixed(1)}M</div><div className="metric-label">Market-wide records</div></div>
        <div className="metric"><div className="metric-value">{(headline.targetEvents / 1_000_000).toFixed(2)}M</div><div className="metric-label">AAPL book transitions</div></div>
        <div className="metric"><div className="metric-value">{headline.throughput.toLocaleString()}</div><div className="metric-label">Records per second</div></div>
        <div className="metric"><div className="metric-value">{headline.payloadGb.toFixed(2)} GB</div><div className="metric-label">Binary payload processed</div></div>
        <div className="metric"><div className="metric-value">{headline.maxOrders.toLocaleString()}</div><div className="metric-label">Maximum open AAPL orders</div></div>
        <div className="metric"><div className="metric-value">{headline.peakMb.toFixed(1)} MB</div><div className="metric-label">Python-tracked peak allocation</div></div>
      </div>

      <section className="section grid two">
        <article className="card">
          <div className="eyebrow">Streaming pipeline</div>
          <h2>Stateful reconstruction, not snapshot loading</h2>
          <div className="pipeline" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
            {[
              ["BinaryFILE", "Read two-byte lengths and stream gzip payloads"],
              ["Message decoder", "Parse big-endian ITCH 5.0 message fields"],
              ["Order-ID state", "Track add, execute, cancel, delete and replace"],
              ["Price aggregation", "Maintain independently sorted bid and ask levels"],
              ["Top-10 snapshots", "Export book state after every target event"],
              ["Invariant audit", "Reconcile orders, levels, quantities and timestamps"],
            ].map(([title, detail]) => <div className="pipeline-step" key={title}><strong>{title}</strong><span>{detail}</span></div>)}
          </div>
        </article>

        <article className="card chart-card">
          <h2>Largest message classes</h2>
          <p className="muted">Market-wide ITCH message volume in millions.</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={messageCounts}>
                <CartesianGrid stroke="rgba(255,255,255,.08)" vertical={false} />
                <XAxis dataKey="type" stroke="#9bb1c0" />
                <YAxis stroke="#9bb1c0" />
                <Tooltip formatter={(value) => `${Number(value).toFixed(2)}M`} />
                <Bar dataKey="countMillions" name="Messages" fill="#66e3d2" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="section grid two">
        <article className="card">
          <div className="eyebrow">Lifecycle counts</div>
          <h2>1,656,597 AAPL book events</h2>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Lifecycle action</th><th>Count</th></tr></thead>
              <tbody>
                {[
                  ["Adds", reconstruction.adds],
                  ["Executions", reconstruction.executions],
                  ["Partial cancels", reconstruction.cancels],
                  ["Deletes", reconstruction.deletes],
                  ["Replaces", reconstruction.replaces],
                ].map(([label, value]) => <tr key={String(label)}><td>{label}</td><td>{Number(value).toLocaleString()}</td></tr>)}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card">
          <div className="eyebrow">Integrity</div>
          <h2>All tracked checks passed</h2>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Check</th><th>Result</th></tr></thead>
              <tbody>
                {invariants.map(([label, value]) => <tr key={label}><td>{label}</td><td style={{ color: Number(value) === 0 ? "var(--positive)" : "var(--danger)" }}>{Number(value).toLocaleString()}</td></tr>)}
                <tr><td>Bid aggregation matches orders</td><td style={{ color: integrity.bid_aggregate_matches_orders ? "var(--positive)" : "var(--danger)" }}>{String(integrity.bid_aggregate_matches_orders)}</td></tr>
                <tr><td>Ask aggregation matches orders</td><td style={{ color: integrity.ask_aggregate_matches_orders ? "var(--positive)" : "var(--danger)" }}>{String(integrity.ask_aggregate_matches_orders)}</td></tr>
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="section">
        <div className="status good">
          The complete compressed sample was consumed to clean gzip EOF with no payload-length mismatches. This sample did not expose a zero-length BinaryFILE terminator.
        </div>
      </section>
    </>
  );
}
