/**
 * Parse a Server-Sent Events stream that arrives in arbitrary chunks.
 *
 * The browser's EventSource only supports GET, and resuming a run is a POST, so the
 * response body is read with fetch and split into events here. A chunk can end in
 * the middle of an event; the unfinished part is kept until the rest arrives.
 */
export type SSEEvent = { event: string; data: unknown };

export function createSSEParser(onEvent: (event: SSEEvent) => void) {
  let buffer = "";
  return {
    push(chunk: string) {
      buffer += chunk.replace(/\r\n/g, "\n");
      let end = buffer.indexOf("\n\n");
      while (end !== -1) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const parsed = parseBlock(block);
        if (parsed) onEvent(parsed);
        end = buffer.indexOf("\n\n");
      }
    },
  };
}

function parseBlock(block: string): SSEEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null; // a malformed event is skipped rather than breaking the whole stream
  }
}
