type Props = { eyebrow: string; title: string; description: string };

export function PageIntro({ eyebrow, title, description }: Props) {
  return (
    <section className="page-intro">
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p className="lead">{description}</p>
    </section>
  );
}
