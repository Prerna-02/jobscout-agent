/**
 * The only module that talks to the JobScout API. Types come from the API's own
 * OpenAPI schema (`npm run types`), so a change on either side shows up as a type error.
 */
import type { components } from "./api-types";
import { createSSEParser } from "./sse";
import { clientId, inviteCode, runKey, saveRunKey } from "./storage";

export type StartResponse = components["schemas"]["StartResponse"];
export type ResultResponse = components["schemas"]["ResultResponse"];
export type ResumeRequest = components["schemas"]["ResumeRequest"];
export type Rubric = components["schemas"]["Rubric"];
export type Criterion = components["schemas"]["Criterion"];
export type JobMatch = components["schemas"]["JobMatch"];
export type CandidateProfile = components["schemas"]["CandidateProfile"];
export type TrustReport = components["schemas"]["TrustReport"];
export type RunView = StartResponse | ResultResponse;
export type Progress = { step: string; message: string };

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** A failed request, with a message that can be shown to the user as is. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

function headers(runId?: string): Record<string, string> {
  const result: Record<string, string> = { "X-Client-Id": clientId() };
  const invite = inviteCode();
  if (invite) result["X-Invite-Code"] = invite;
  const key = runId ? runKey(runId) : null;
  if (key) result["X-Run-Token"] = key;
  return result;
}

async function failure(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status}).`;
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") message = detail;
    else if (detail?.message) message = detail.message;
    else if (Array.isArray(detail) && detail[0]?.msg) message = detail[0].msg;
  } catch {
    // not JSON: keep the generic message
  }
  if (response.status === 404) message = "This run can't be found in this browser.";
  return new ApiError(message, response.status);
}

async function request(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError("Can't reach JobScout. Check your connection and try again.", 0);
  }
}

/** Upload a CV. Saves the run's secret key in this browser. */
export async function startRun(file: File): Promise<StartResponse> {
  const form = new FormData();
  form.append("cv", file);
  const response = await request("/runs", { method: "POST", body: form, headers: headers() });
  if (!response.ok) throw await failure(response);
  const body = (await response.json()) as StartResponse;
  if (body.run_token) saveRunKey(body.run_id, body.run_token);
  return body;
}

export async function getRun(runId: string): Promise<RunView> {
  const response = await request(`/runs/${encodeURIComponent(runId)}`, {
    headers: headers(runId),
  });
  if (!response.ok) throw await failure(response);
  return (await response.json()) as RunView;
}

/** Continue a run, calling `onProgress` for each step as it happens. */
export async function resumeRun(
  runId: string,
  choices: ResumeRequest,
  onProgress: (progress: Progress) => void,
): Promise<ResultResponse> {
  const response = await request(`/runs/${encodeURIComponent(runId)}/resume`, {
    method: "POST",
    headers: { ...headers(runId), "Content-Type": "application/json" },
    body: JSON.stringify(choices),
  });
  if (!response.ok || !response.body) throw await failure(response);

  let result: ResultResponse | null = null;
  let error: string | null = null;
  const parser = createSSEParser(({ event, data }) => {
    if (event === "progress") onProgress(data as Progress);
    else if (event === "result") result = data as ResultResponse;
    else if (event === "error") error = (data as { message?: string }).message ?? "Search failed.";
  });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.push(value);
  }
  if (error) throw new ApiError(error, 500);
  if (!result) throw new ApiError("The search ended without a result. Please try again.", 500);
  return result;
}
