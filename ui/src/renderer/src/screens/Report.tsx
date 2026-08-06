// Screen 03: final report with per-subgenre distribution and summary stats.

import type { Job } from '../api/types'
import { dirnameFor, pct, seconds } from '../lib/format'

interface Props {
  job: Job
  onOpenFolder: () => void
  onRestart: () => void
}

function Report({ job, onOpenFolder, onRestart }: Props): JSX.Element {
  const entries = Object.entries(job.subgenre_counts).filter(([, n]) => n > 0)
  const organized = entries.reduce((sum, [, n]) => sum + n, 0)
  const max = Math.max(1, ...entries.map(([, n]) => n))
  const folders = entries.length
  const unreadable = job.total - job.processed

  return (
    <main className="app">
      <p className="eyebrow">Listo</p>
      <h1 className="screen-title">{organized} tracks organizados</h1>

      <div className="stats">
        {job.average_confidence != null && (
          <div className="stat">
            <div className="label">Confianza media</div>
            <div className="value">{pct(job.average_confidence)}</div>
          </div>
        )}
        <div className="stat">
          <div className="label">Tiempo total</div>
          <div className="value">{seconds(job.elapsed_seconds)}</div>
        </div>
      </div>

      <p className="meta mono">
        {job.directory} · {folders} subcarpeta{folders === 1 ? '' : 's'} creada
        {folders === 1 ? '' : 's'}
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
          {job.review_count} track{job.review_count === 1 ? '' : 's'} con confianza baja quedaron
          en /Revisar, con su 2ª opción anotada.
        </p>
      )}
      {unreadable > 0 && (
        <p className="meta">
          {unreadable} archivo{unreadable === 1 ? '' : 's'} no se pudo leer y quedó en su lugar.
        </p>
      )}

      <div className="actions">
        <button className="btn-primary" onClick={onOpenFolder}>
          Abrir carpeta
        </button>
        <button className="btn-link" onClick={onRestart}>
          Clasificar otra carpeta
        </button>
      </div>
    </main>
  )
}

export default Report
