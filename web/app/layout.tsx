import type { Metadata } from "next";

import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { getSiteUrl } from "@/lib/site";

import "./globals.css";

const siteUrl = getSiteUrl();

export const metadata: Metadata = {
  title: {
    default: "Order Book Microstructure Lab",
    template: "%s | Order Book Lab",
  },
  description:
    "Interactive Nasdaq ITCH reconstruction, microstructure validation and execution-cost research.",
  metadataBase: new URL(siteUrl),
  openGraph: {
    title: "Order Book Microstructure Lab",
    description:
      "368M ITCH messages, 1.66M reconstructed AAPL transitions, leakage-controlled modelling and cost-aware conclusions.",
    type: "website",
    url: siteUrl,
  },
  twitter: {
    card: "summary_large_image",
    title: "Order Book Microstructure Lab",
    description:
      "Interactive Nasdaq ITCH reconstruction and cost-aware microstructure research.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div className="site-shell">
          <SiteHeader />
          <main className="main">
            <div className="container">{children}</div>
          </main>
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}
