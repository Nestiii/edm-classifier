// Screen 03: final report with per-subgenre distribution and summary stats.

import { useState } from 'react'

import ResultsList from '../components/ResultsList'
import type { Job } from '../api/types'
import { dirnameFor, pct, seconds } from '../lib/format'

interface Props {
  job: Job
  onOpenFolder: () => void
  onRestart: () => void
}

function Report({ job, onOpenFolder, onRestart }: Props): JSX.Element {
  const [showAll, setShowAll] = useState(false)
  const entries = Object.entries(job.subgenre_counts).filter(([, n]) => n > 0)
  const organized = entries.reduce((sum, [, n]) => sum + n, 0)
  const max = Math.max(1, ...entries.map(([, n]) => n))
  const folders = entries.length
  const unreadable = job.total - job.processed

  return (
    <main className="app">
      <p className="eyebrow">Done</p>
      <h1 className="screen-title">{organized} tracks organized</h1>

      <div className="stats">
        {job.average_confidence != null && (
          <div className="stat">
            <div className="label">Avg. confidence</div>
            <div className="value">{pct(job.average_confidence)}</div>
          </div>
        )}
        <div className="stat">
          <div className="label">Total time</div>
          <div className="value">{seconds(job.elapsed_seconds)}</div>
        </div>
      </div>

      <p className="meta mono">
        {job.directory} · {folders} subfolder{folders === 1 ? '' : 's'} created
      </p>

      <div className="dist">
        {entries.map(([subgenre, n]) => (
          <div className="dist-item" key={subgenre}>
            <span className="folder mono">{dirnameFor(subgenre)}/</span>
            <span className="bar">
              <span style={{ width: pct(n / max) }} />
            </span>
            <span className="count">{n}</span>
          </div>
        ))}
      </div>

      {job.review_count > 0 && (
        <p className="meta">
          {job.review_count} track{job.review_count === 1 ? '' : 's'} with low confidence went to
          the Review folder, with their 2nd choice noted.
        </p>
      )}
      {unreadable > 0 && (
        <p className="meta">
          {unreadable} file{unreadable === 1 ? '' : 's'} couldn&apos;t be read and stayed in place.
        </p>
      )}

      <button className="btn-link" onClick={() => setShowAll((v) => !v)}>
        {showAll ? 'Hide details' : `Show all ${job.results.length} files`}
      </button>
      {showAll && <ResultsList results={job.results} />}

      <div className="actions">
        <button className="btn-primary" onClick={onOpenFolder}>
          Open folder
        </button>
        <button className="btn-link" onClick={onRestart}>
          Sort another folder
        </button>
      </div>
    </main>
  )
}

export default Report
