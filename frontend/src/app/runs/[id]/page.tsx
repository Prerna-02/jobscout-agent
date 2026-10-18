import { Suspense } from "react";

import { RunView } from "./RunView";

// The page shell is prerendered; the part that needs the run id streams in inside
// <Suspense>, as Cache Components requires for request-time data such as `params`.
export default function RunPage(props: PageProps<"/runs/[id]">) {
  return (
    <Suspense fallback={<p className="text-muted">Loading your run…</p>}>
      <Run params={props.params} />
    </Suspense>
  );
}

async function Run({ params }: { params: PageProps<"/runs/[id]">["params"] }) {
  const { id } = await params;
  return <RunView runId={id} />;
}
