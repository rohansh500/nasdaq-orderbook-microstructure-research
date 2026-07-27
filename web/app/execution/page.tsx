import type { Metadata } from "next";
import { ExecutionExplorer } from "@/components/ExecutionExplorer";
import { PageIntro } from "@/components/PageIntro";

export const metadata: Metadata = { title: "Execution-cost lab" };

export default function ExecutionPage() {
  return (
    <>
      <PageIntro
        eyebrow="Economic reality"
        title="Stress the edge before calling it alpha"
        description="Adjust the confidence threshold and execution-cost assumption to see why statistical predictability did not survive aggressive spread crossing."
      />
      <ExecutionExplorer />
    </>
  );
}
