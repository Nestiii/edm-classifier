import { useCallback, useEffect, useRef, useState } from 'react'

import { cancelJob, createJob, getHealth, getJob } from './api/client'
import type { Job } from './api/types'
import Preview from './screens/Preview'
import Progress from './screens/Progress'
import Report from './screens/Report'
import SelectFolder, { type RunOptions } from './screens/SelectFolder'
import Startup from './screens/Startup'

type Phase =
  | 'starting'
  | 'startup-error'
  | 'select'
  | 'previewing' // dry-run (classify) running
  | 'preview' // dry-run done, awaiting confirmation
  | 'organizing' // move running
  | 'report'
  | 'job-error'

const HEALTH_RETRIES = 8
const HEALTH_INTERVAL_MS = 800
const POLL_INTERVAL_MS = 400

function App(): JSX.Element {
  const [phase, setPhase] = useState<Phase>('starting')
  const [job, setJob] = useState<Job | null>(null)
  const [device, setDevice] = useState<string | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)
  const optsRef = useRef<RunOptions | null>(null)
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const checkHealth = useCallback(async () => {
    setPhase('starting')
    for (let attempt = 0; attempt < HEALTH_RETRIES; attempt++) {
      try {
        const health = await getHealth()
        setDevice(health.device)
        if (health.model_loaded) {
          setPhase('select')
          return
        }
      } catch {
        /* backend not up yet */
      }
      await new Promise((r) => setTimeout(r, HEALTH_INTERVAL_MS))
    }
    setPhase('startup-error')
  }, [])

  useEffect(() => {
    checkHealth()
  }, [checkHealth])

  // Poll a running job; call onDone with the terminal job.
  const pollJob = useCallback((jobId: string, onDone: (job: Job) => void) => {
    const tick = async (): Promise<void> => {
      try {
        const current = await getJob(jobId)
        setJob(current)
        if (['completed', 'failed', 'cancelled'].includes(current.status)) {
          onDone(current)
          return
        }
      } catch {
        setPhase('startup-error')
        return
      }
      pollRef.current = setTimeout(tick, POLL_INTERVAL_MS)
    }
    tick()
  }, [])

  useEffect(() => {
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current)
    }
  }, [])

  // Start a job in the given mode and poll it to completion.
  const runJob = useCallback(
    async (mode: 'classify' | 'move', onDone: (job: Job) => void) => {
      const opts = optsRef.current
      if (!opts) return
      try {
        const created = await createJob({
          directory: opts.directory,
          mode,
          recursive: opts.recursive,
          confidenceThreshold: opts.confidenceThreshold
        })
        setJob(null)
        pollJob(created.job_id, onDone)
      } catch (e) {
        setJobError(e instanceof Error ? e.message : String(e))
        setPhase('job-error')
      }
    },
    [pollJob]
  )

  function terminalHandler(next: Phase) {
    return (finished: Job): void => {
      if (finished.status === 'completed') setPhase(next)
      else if (finished.status === 'failed') {
        setJobError(finished.error ?? 'Unknown error')
        setPhase('job-error')
      } else setPhase('select') // cancelled
    }
  }

  function handlePreview(opts: RunOptions): void {
    optsRef.current = opts
    setPhase('previewing')
    runJob('classify', terminalHandler('preview'))
  }

  function handleConfirm(): void {
    setPhase('organizing')
    runJob('move', terminalHandler('report'))
  }

  async function handleCancel(): Promise<void> {
    if (pollRef.current) clearTimeout(pollRef.current)
    if (job) await cancelJob(job.job_id).catch(() => undefined)
    setPhase('select')
  }

  function handleRestart(): void {
    setJob(null)
    setPhase('select')
  }

  function handleOpenFolder(): void {
    if (job) window.api.openPath(job.directory)
  }

  switch (phase) {
    case 'starting':
      return <Startup error={false} device={device} onRetry={checkHealth} />
    case 'startup-error':
      return <Startup error={true} onRetry={checkHealth} />
    case 'job-error':
      return (
        <Startup error={false} jobError={jobError} onRetry={checkHealth} onBack={handleRestart} />
      )
    case 'previewing':
      return job ? (
        <Progress job={job} label="Analyzing" onCancel={handleCancel} />
      ) : (
        <Startup error={false} device={device} onRetry={checkHealth} />
      )
    case 'preview':
      return job ? (
        <Preview job={job} onConfirm={handleConfirm} onBack={handleRestart} />
      ) : (
        <SelectFolder onPreview={handlePreview} />
      )
    case 'organizing':
      return job ? (
        <Progress job={job} label="Organizing" onCancel={handleCancel} />
      ) : (
        <Startup error={false} device={device} onRetry={checkHealth} />
      )
    case 'report':
      return job ? (
        <Report job={job} onOpenFolder={handleOpenFolder} onRestart={handleRestart} />
      ) : (
        <SelectFolder onPreview={handlePreview} />
      )
    default:
      return <SelectFolder onPreview={handlePreview} />
  }
}

export default App
