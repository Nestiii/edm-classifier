// Full per-file results, grouped by subgenre (review tracks last).

import type { FileResult } from '../api/types'
import { basename, pct } from '../lib/format'

interface Props {
  results: FileResult[]
}

function groupKey(r: FileResult): string {
  return r.review ? 'Review' : r.subgenre
}

function ResultsList({ results }: Props): JSX.Element {
  // Group by subgenre (or Revisar), preserving a stable order.
  const groups = new Map<string, FileResult[]>()
  for (const r of results) {
    const key = groupKey(r)
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(r)
  }
  // Review group always last.
  const ordered = [...groups.entries()].sort((a, b) => {
    if (a[0] === 'Review') return 1
    if (b[0] === 'Review') return -1
    return b[1].length - a[1].length
  })

  return (
    <div className="results-scroll">
      {ordered.map(([key, items]) => (
        <div className="results-group" key={key}>
          <div className="results-group-head">
            <span className={key === 'Review' ? 'badge outline' : 'badge solid'}>{key}</span>
            <span className="results-group-count">{items.length}</span>
          </div>
          {items.map((r) => (
            <div className="row" key={r.path}>
              <div className="name">
                {basename(r.path)}
                {r.review && r.second_choice && (
                  <div className="sub">
                    2nd choice: {r.second_choice.subgenre} · {pct(r.second_choice.probability)}
                  </div>
                )}
              </div>
              <span className="conf">{pct(r.confidence)}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

export default ResultsList
