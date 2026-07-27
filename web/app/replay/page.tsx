import type { Metadata } from "next";
import { PageIntro } from "@/components/PageIntro";
import { ReplayLab } from "@/components/ReplayLab";

export const metadata: Metadata = { title: "Order-book replay" };

export default function ReplayPage() {
  return (
    <>
      <PageIntro
        eyebrow="Market-data engineering"
        title="Replay the displayed order lifecycle"
        description="Step through add, execute, partial-cancel, delete and replace events and watch the reconstructed top of book respond without exposing licensed market rows."
      />
      <ReplayLab />
    </>
  );
}
