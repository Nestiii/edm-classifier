// Screen 02: live progress while the batch job runs.

import type { Job } from '../api/types'
import { basename, pct, seconds } from '../lib/format'

interface Props {
  job: Job
  label: string
  onCancel: () => void
}

function Progress({ job, label, onCancel }: Props): JSX.Element {
  const ratio = job.total > 0 ? job.processed / job.total : 0
  // Most recent results first, capped for a tidy list.
  const recent = [...job.results].reverse().slice(0, 6)

  return (
    <main className="app">
      <p className="eyebrow">{label}</p>
      <h1 className="screen-title">
        {job.processed} of {job.total} files
      </h1>
      <p className="meta">
        {job.eta_seconds != null ? `~${seconds(job.eta_seconds)} left · ` : ''}
        {pct(ratio)}
      </p>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: pct(ratio) }} />
      </div>

      <div className="rows">
        {recent.map((r) => (
          <div className="row" key={r.path}>
            <div className="name">
              {basename(r.path)}
              {r.review && r.second_choice && (
                <div className="sub">
                  2nd choice: {r.second_choice.subgenre} · {pct(r.second_choice.probability)}
                </div>
              )}
            </div>
            <span className={r.review ? 'badge outline' : 'badge solid'}>{r.subgenre}</span>
            <span className="conf">{pct(r.confidence)}</span>
          </div>
        ))}
      </div>

      <div className="actions">
        <button className="btn-link" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </main>
  )
}

export default Progress
