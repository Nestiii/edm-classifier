import { useCallback, useEffect, useRef, useState } from 'react'

import { cancelJob, createJob, getHealth, getJob } from './api/client'
import type { Job } from './api/types'
import Progress from './screens/Progress'
import Report from './screens/Report'
import SelectFolder from './screens/SelectFolder'
import Startup from './screens/Startup'

type Phase = 'starting' | 'startup-error' | 'select' | 'progress' | 'report'

const HEALTH_RETRIES = 8
const HEALTH_INTERVAL_MS = 800
const POLL_INTERVAL_MS = 400

// Low-confidence tracks below this go to the /Revisar folder.
const CONFIDENCE_THRESHOLD = 0.5

function App(): JSX.Element {
  const [phase, setPhase] = useState<Phase>('starting')
  const [job, setJob] = useState<Job | null>(null)
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Poll /health until the model is loaded, or give up after N retries.
  const checkHealth = useCallback(async () => {
    setPhase('starting')
    for (let attempt = 0; attempt < HEALTH_RETRIES; attempt++) {
      try {
        const health = await getHealth()
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

  // Poll a running job until it reaches a terminal state.
  const pollJob = useCallback((jobId: string) => {
    const tick = async (): Promise<void> => {
      try {
        const current = await getJob(jobId)
        setJob(current)
        if (['completed', 'failed', 'cancelled'].includes(current.status)) {
          if (current.status === 'completed') setPhase('report')
          else setPhase('select')
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

  async function handleStart(directory: string): Promise<void> {
    try {
      const created = await createJob({
        directory,
        mode: 'move',
        confidenceThreshold: CONFIDENCE_THRESHOLD
      })
      setJob(null)
      setPhase('progress')
      pollJob(created.job_id)
    } catch {
      setPhase('startup-error')
    }
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
      return <Startup error={false} onRetry={checkHealth} />
    case 'startup-error':
      return <Startup error={true} onRetry={checkHealth} />
    case 'progress':
      return job ? (
        <Progress job={job} onCancel={handleCancel} />
      ) : (
        <Startup error={false} onRetry={checkHealth} />
      )
    case 'report':
      return job ? (
        <Report job={job} onOpenFolder={handleOpenFolder} onRestart={handleRestart} />
      ) : (
        <SelectFolder onStart={handleStart} />
      )
    default:
      return <SelectFolder onStart={handleStart} />
  }
}

export default App
