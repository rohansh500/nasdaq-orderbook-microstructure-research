type Props = { label: string; value: string; detail?: string };

export function MetricCard({ label, value, detail }: Props) {
  return (
    <article className="metric">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {detail ? <div className="metric-detail">{detail}</div> : null}
    </article>
  );
}
