import type { Metadata } from "next";
import { PageIntro } from "@/components/PageIntro";
import { PredictionLab } from "@/components/PredictionLab";

export const metadata: Metadata = { title: "Prediction lab" };

export default function PredictionPage() {
  return (
    <>
      <PageIntro
        eyebrow="Browser model inference"
        title="Build a market scenario, then inspect the model response"
        description="The controls generate a 150-event synthetic order-book history. All 31 frozen features are derived coherently and evaluated locally in the browser through parity-verified LightGBM tree models."
      />
      <PredictionLab />
    </>
  );
}
