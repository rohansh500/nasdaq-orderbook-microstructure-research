import type { Metadata } from "next";
import { PageIntro } from "@/components/PageIntro";
import { ValidationExplorer } from "@/components/ValidationExplorer";

export const metadata: Metadata = { title: "Validation explorer" };

export default function ValidationPage() {
  return (
    <>
      <PageIntro
        eyebrow="Research discipline"
        title="Inspect the evidence before the headline metric"
        description="Explore purged walk-forward folds, model comparisons, feature-family ablation and the frozen holdout without hiding unstable periods or negative economic outcomes."
      />
      <ValidationExplorer />
    </>
  );
}
