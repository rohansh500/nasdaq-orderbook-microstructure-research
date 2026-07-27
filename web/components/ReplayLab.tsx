"use client";

import { useEffect, useMemo, useState } from "react";
import { replay } from "@/lib/data";

type Level = [number, number];
type ReplayEvent = {
  index: number;
  time: string;
  message: string;
  event: string;
  orderId: string;
  description: string;
  bid: Level[];
  ask: Level[];
};

function BookSide({ title, rows, side }: { title: string; rows: Level[]; side: "bid" | "ask" }) {
  return (
    <div className={`book-side ${side}`}>
      <div className="book-head"><span>{title} price</span><span>Displayed size</span></div>
      {rows.map(([price, size], index) => (
        <div className="book-row" key={`${price}-${index}`}>
          <strong>{price.toFixed(2)}</strong><span>{size.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

export function ReplayLab() {
  const events = replay.events as ReplayEvent[];
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const current = events[index];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setIndex((value) => (value + 1) % events.length);
    }, 1300);
    return () => window.clearInterval(timer);
  }, [events.length, playing]);

  const metrics = useMemo(() => {
    const bestBid = current.bid[0];
    const bestAsk = current.ask[0];
    const mid = (bestBid[0] + bestAsk[0]) / 2;
    const spreadBps = ((bestAsk[0] - bestBid[0]) / mid) * 10_000;
    const queue = (bestBid[1] - bestAsk[1]) / (bestBid[1] + bestAsk[1]);
    return { mid, spreadBps, queue };
  }, [current]);

  return (
    <div className="grid two">
      <section className="card">
        <div className="control-row">
          <div>
            <div className="eyebrow">Message {current.message}</div>
            <h2>{current.event}</h2>
          </div>
          <span className="mono">{current.time}</span>
        </div>
        <p className="lead">{current.description}</p>
        <div className="grid three">
          <div className="metric"><div className="metric-value">{metrics.mid.toFixed(3)}</div><div className="metric-label">Mid price</div></div>
          <div className="metric"><div className="metric-value">{metrics.spreadBps.toFixed(2)}</div><div className="metric-label">Spread bps</div></div>
          <div className="metric"><div className="metric-value">{metrics.queue.toFixed(3)}</div><div className="metric-label">L1 queue imbalance</div></div>
        </div>
        <div className="actions">
          <button className="button primary" onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play lifecycle"}</button>
          <button className="button" onClick={() => setIndex((value) => (value - 1 + events.length) % events.length)}>Previous</button>
          <button className="button" onClick={() => setIndex((value) => (value + 1) % events.length)}>Next</button>
        </div>
        <div className="control" style={{ marginTop: 16 }}>
          <div className="control-row"><label htmlFor="replay-index">Event position</label><output>{index + 1} / {events.length}</output></div>
          <input id="replay-index" type="range" min={0} max={events.length - 1} value={index} onChange={(event) => { setPlaying(false); setIndex(Number(event.target.value)); }} />
        </div>
      </section>

      <section className="card">
        <div className="control-row"><h2>Displayed top of book</h2><span className="mono">order {current.orderId}</span></div>
        <div className="book">
          <BookSide title="Bid" rows={current.bid} side="bid" />
          <BookSide title="Ask" rows={current.ask} side="ask" />
        </div>
        <p className="muted" style={{ marginTop: 18 }}>
          This replay uses a synthetic lifecycle fixture. The production parser maintained order-ID state and aggregated full displayed depth before exporting top-level snapshots.
        </p>
      </section>
    </div>
  );
}
