// Types mirroring the classifier API's JSON responses.

export interface Health {
  status: string
  model_loaded: boolean
  device: string | null
}

export interface Top2Item {
  subgenre: string
  probability: number
}

export type JobMode = 'classify' | 'move' | 'copy'
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface FileResult {
  path: string
  subgenre: string
  confidence: number
  organized_path: string | null
  review: boolean
  second_choice: Top2Item | null
}

export interface Job {
  job_id: string
  status: JobStatus
  mode: JobMode
  directory: string
  total: number
  processed: number
  current_file: string | null
  subgenre_counts: Record<string, number>
  review_count: number
  average_confidence: number | null
  elapsed_seconds: number | null
  eta_seconds: number | null
  results: FileResult[]
  error: string | null
}

export interface JobCreated {
  job_id: string
  status: JobStatus
  total: number
}
