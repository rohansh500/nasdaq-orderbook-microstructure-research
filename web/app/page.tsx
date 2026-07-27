import Link from "next/link";
import { MetricCard } from "@/components/MetricCard";
import { headline } from "@/lib/data";
import { repositoryUrl } from "@/lib/site";

export default function HomePage() {
  return (
    <>
      <section className="hero">
        <div>
          <div className="eyebrow">Quant research + market-data engineering</div>
          <h1>Can order flow predict the next move - and survive the spread?</h1>
          <p className="lead">
            An interactive view of a leakage-controlled AAPL microstructure study and an
            independent Nasdaq TotalView-ITCH 5.0 reconstruction engine. The result is
            statistically meaningful, economically constrained, and deliberately honest.
          </p>
          <div className="actions">
            <Link className="button primary" href="/prediction">Open prediction lab</Link>
            <Link className="button" href="/execution">Test execution costs</Link>
            <a className="button" href={repositoryUrl} target="_blank" rel="noreferrer">GitHub repository</a>
          </div>
        </div>
        <aside className="hero-card">
          <div className="eyebrow">Final conclusion</div>
          <h2>Predictability is not the same as executable alpha.</h2>
          <p className="muted">{headline.conclusion}</p>
          <div className="status warn">
            Frozen LightGBM: 0.289 bps gross edge versus 1.854 bps estimated aggressive cost.
          </div>
        </aside>
      </section>

      <section className="section">
        <div className="grid metrics">
          {headline.metrics.map((metric) => <MetricCard key={metric.label} {...metric} />)}
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <div className="eyebrow">Research architecture</div>
            <h2>From binary feed to frozen evaluation</h2>
          </div>
          <p className="muted">
            The prediction study and the raw ITCH reconstruction are separate evidence paths:
            one tests signal quality, the other demonstrates low-level market-data engineering.
          </p>
        </div>
        <div className="pipeline">
          {[
            ["1", "Binary ITCH", "Length-prefixed messages streamed from gzip"],
            ["2", "Book state", "Order-ID lifecycle and full-depth aggregation"],
            ["3", "Features", "35 state, flow, intensity and volatility features"],
            ["4", "Validation", "Purged walk-forward folds and block bootstrap"],
            ["5", "Models", "Linear baselines, LightGBM and family ablation"],
            ["6", "Economics", "Frozen holdout and quoted-spread cost stress"],
          ].map(([number, title, detail]) => (
            <article className="pipeline-step" key={title}>
              <span className="eyebrow">{number}</span>
              <strong>{title}</strong>
              <span>{detail}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <div className="eyebrow">Explore the evidence</div>
            <h2>Five research-friendly views</h2>
          </div>
        </div>
        <div className="grid three">
          {[
            ["Order-book replay", "Watch add, cancel, execute, delete and replace events change the displayed book.", "/replay"],
            ["Prediction lab", "Generate a coherent synthetic market history and run the frozen 31-feature model in-browser.", "/prediction"],
            ["Validation explorer", "Compare horizons, folds, bootstrap intervals and feature-family ablations.", "/validation"],
            ["Execution-cost lab", "Stress the gross edge under confidence and crossing-cost assumptions.", "/execution"],
            ["ITCH engineering", "Inspect throughput, lifecycle counters, invariants and message composition.", "/engineering"],
          ].map(([title, detail, href]) => (
            <article className="card" key={title}>
              <h3>{title}</h3>
              <p className="muted">{detail}</p>
              <Link className="button" href={href}>Explore</Link>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="grid two">
          <article className="card">
            <div className="eyebrow">What held up</div>
            <h2>Event flow mattered.</h2>
            <p className="muted">
              Removing event-flow features reduced 50-event LightGBM rank IC from 0.266 to
              0.048 and almost eliminated the gross signal. Removing explicit time features
              improved generalisation.
            </p>
          </article>
          <article className="card">
            <div className="eyebrow">What failed</div>
            <h2>Crossing the spread dominated the edge.</h2>
            <p className="muted">
              The frozen model improved holdout error and ranking, but full aggressive cost
              produced materially negative net results. The project reports that failure
              instead of hiding it behind classification accuracy.
            </p>
          </article>
        </div>
      </section>
    </>
  );
}
