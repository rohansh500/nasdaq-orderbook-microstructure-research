import { repositoryUrl } from "@/lib/site";

export function SiteFooter() {
  return (
    <footer className="footer">
      <div className="container footer-row">
        <span>Research demonstration - not a live forecast or trading recommendation.</span>
        <a href={repositoryUrl} target="_blank" rel="noreferrer">
          View source on GitHub
        </a>
      </div>
    </footer>
  );
}
