import Link from "next/link";

const links = [
  ["Replay", "/replay"],
  ["Prediction", "/prediction"],
  ["Validation", "/validation"],
  ["Execution", "/execution"],
  ["ITCH", "/engineering"],
] as const;

export function SiteHeader() {
  return (
    <header className="header">
      <div className="container header-row">
        <Link className="brand" href="/">
          <span className="brand-mark">OB</span>
          <span>Order Book Microstructure Lab</span>
        </Link>
        <nav className="nav" aria-label="Primary navigation">
          {links.map(([label, href]) => (
            <Link href={href} key={href}>{label}</Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
