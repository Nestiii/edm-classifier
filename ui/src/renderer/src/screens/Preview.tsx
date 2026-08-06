// Screen 02: preview of the dry-run (classify only). Nothing has moved yet.

import ResultsList from '../components/ResultsList'
import type { Job } from '../api/types'
import { pct } from '../lib/format'

interface Props {
  job: Job
  onConfirm: () => void
  onBack: () => void
}

function Preview({ job, onConfirm, onBack }: Props): JSX.Element {
  const willReview = job.results.filter((r) => r.review).length
  const willOrganize = job.results.length - willReview

  return (
    <main className="app">
      <p className="eyebrow">Preview</p>
      <h1 className="screen-title">Here's how your library would look</h1>
      <p className="screen-sub">
        Nothing has moved yet. If it looks right, we'll organize the files into subgenre folders.
      </p>

      <div className="stats">
        <div className="stat">
          <div className="label">To subfolders</div>
          <div className="value">{willOrganize}</div>
        </div>
        {willReview > 0 && (
          <div className="stat">
            <div className="label">To review</div>
            <div className="value">{willReview}</div>
          </div>
        )}
        {job.average_confidence != null && (
          <div className="stat">
            <div className="label">Avg. confidence</div>
            <div className="value">{pct(job.average_confidence)}</div>
          </div>
        )}
      </div>

      <ResultsList results={job.results} />

      <div className="actions">
        <button className="btn-primary" onClick={onConfirm}>
          Organize
        </button>
        <button className="btn-link" onClick={onBack}>
          Discard
        </button>
      </div>
    </main>
  )
}

export default Preview
