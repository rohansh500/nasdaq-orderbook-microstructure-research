const fallbackSiteUrl = "http://localhost:3000";

function normalizeUrl(value: string): string {
  const withProtocol = /^https?:\/\//i.test(value) ? value : `https://${value}`;
  return withProtocol.replace(/\/$/, "");
}

export function getSiteUrl(): string {
  const configured = process.env.NEXT_PUBLIC_SITE_URL?.trim();
  if (configured) return normalizeUrl(configured);

  const vercelProduction = process.env.VERCEL_PROJECT_PRODUCTION_URL?.trim();
  if (vercelProduction) return normalizeUrl(vercelProduction);

  const vercelDeployment = process.env.VERCEL_URL?.trim();
  if (vercelDeployment) return normalizeUrl(vercelDeployment);

  return fallbackSiteUrl;
}

export const repositoryUrl =
  "https://github.com/rohansh500/nasdaq-orderbook-microstructure-research";
