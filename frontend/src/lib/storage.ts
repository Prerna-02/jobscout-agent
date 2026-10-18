/**
 * What this browser remembers: its anonymous id, each run's secret key, and the
 * invite code. Storage can be blocked (private windows, strict settings), so every
 * access is guarded and the app keeps working, just without remembering.
 */
const CLIENT_ID = "jobscout.clientId";
const INVITE = "jobscout.inviteCode";
const RUN_KEY = (runId: string) => `jobscout.run.${runId}`;

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // storage unavailable: nothing to do, the value just isn't remembered
  }
}

let memoryClientId: string | null = null;

/** A random id for this browser (daily limits are counted per browser, not per person). */
export function clientId(): string {
  const saved = read(CLIENT_ID);
  if (saved) return saved;
  const fresh = memoryClientId ?? crypto.randomUUID().replace(/-/g, "");
  memoryClientId = fresh;
  write(CLIENT_ID, fresh);
  return fresh;
}

export function saveRunKey(runId: string, key: string): void {
  write(RUN_KEY(runId), key);
}

export function runKey(runId: string): string | null {
  return read(RUN_KEY(runId));
}

export function inviteCode(): string {
  return read(INVITE) ?? "";
}

export function saveInviteCode(code: string): void {
  write(INVITE, code.trim());
}
