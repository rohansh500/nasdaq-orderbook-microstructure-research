import type { Metadata } from "next";
import { EngineeringExplorer } from "@/components/EngineeringExplorer";
import { PageIntro } from "@/components/PageIntro";

export const metadata: Metadata = { title: "ITCH engineering" };

export default function EngineeringPage() {
  return (
    <>
      <PageIntro
        eyebrow="Nasdaq TotalView-ITCH 5.0"
        title="368 million messages into a verified order book"
        description="Inspect the streaming parser, order-ID lifecycle, price-level aggregation, throughput and invariant checks behind the independent AAPL reconstruction."
      />
      <EngineeringExplorer />
    </>
  );
}
