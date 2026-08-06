// Thin fetch wrapper around the local classifier API.

import type { Health, Job, JobCreated, JobMode } from './types'

const BASE = 'http://127.0.0.1:8000'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* body not JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

/** GET /health — also used to detect whether the backend is reachable. */
export async function getHealth(signal?: AbortSignal): Promise<Health> {
  return json<Health>(await fetch(`${BASE}/health`, { signal }))
}

/** POST /jobs — start a batch classification over a directory. */
export async function createJob(params: {
  directory: string
  mode: JobMode
  recursive?: boolean
  confidenceThreshold?: number
}): Promise<JobCreated> {
  const res = await fetch(`${BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      directory: params.directory,
      mode: params.mode,
      recursive: params.recursive ?? false,
      confidence_threshold: params.confidenceThreshold ?? 0
    })
  })
  return json<JobCreated>(res)
}

/** GET /jobs/{id} — poll progress + results. */
export async function getJob(jobId: string): Promise<Job> {
  return json<Job>(await fetch(`${BASE}/jobs/${jobId}`))
}

/** DELETE /jobs/{id} — request cancellation. */
export async function cancelJob(jobId: string): Promise<Job> {
  return json<Job>(await fetch(`${BASE}/jobs/${jobId}`, { method: 'DELETE' }))
}
