// Screen 01: pick a folder, configure the run, and start a preview.

import { useState } from 'react'

export interface RunOptions {
  directory: string
  recursive: boolean
  confidenceThreshold: number
}

interface Props {
  onPreview: (opts: RunOptions) => void
}

function SelectFolder({ onPreview }: Props): JSX.Element {
  const [directory, setDirectory] = useState<string | null>(null)
  const [counts, setCounts] = useState<{ supported: number; ignored: number } | null>(null)
  const [recursive, setRecursive] = useState(false)
  const [threshold, setThreshold] = useState(0.5)

  async function pick(): Promise<void> {
    const dir = await window.api.selectDirectory()
    if (!dir) return
    setDirectory(dir)
    setCounts(await window.api.countAudio(dir))
  }

  const supported = counts?.supported ?? 0
  const noAudio = directory !== null && supported === 0

  return (
    <main className="app">
      <p className="eyebrow">Subgenre Sorter</p>
      <h1 className="screen-title">Sort your library by subgenre</h1>
      <p className="screen-sub">
        Pick a folder of tracks. We analyze every MP3, AIFF, or WAV and show you a preview before
        touching anything.
      </p>

      <div className="field">
        <span className={directory ? 'path mono' : 'path placeholder mono'}>
          {directory ?? 'No folder selected'}
        </span>
        <button className="btn-secondary" onClick={pick}>
          Choose folder…
        </button>
      </div>

      {directory && !noAudio && (
        <>
          <p className="meta">
            {supported} audio file{supported === 1 ? '' : 's'} found (MP3, AIFF, WAV)
          </p>
          {counts!.ignored > 0 && (
            <p className="meta">
              {counts!.ignored} file{counts!.ignored === 1 ? '' : 's'} skipped (unsupported format)
            </p>
          )}

          <div className="options">
            <label className="opt-toggle">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
              />
              <span>Include subfolders</span>
            </label>

            <div className="opt-slider">
              <div className="opt-slider-head">
                <span>Review threshold</span>
                <span className="mono">{Math.round(threshold * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.95}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
              <p className="opt-hint">
                Tracks below this confidence go to the <code>Review</code> folder.
              </p>
            </div>
          </div>
        </>
      )}

      {noAudio && (
        <p className="meta">
          No supported audio in this folder. Try one with MP3, AIFF, or WAV files.
        </p>
      )}

      <div className="actions">
        <button
          className="btn-primary"
          disabled={!directory || noAudio}
          onClick={() =>
            onPreview({ directory: directory!, recursive, confidenceThreshold: threshold })
          }
        >
          {supported > 0 ? `Analyze ${supported} tracks` : 'Analyze'}
        </button>
      </div>
    </main>
  )
}

export default SelectFolder
