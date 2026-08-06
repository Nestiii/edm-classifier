// Screen 01: pick a folder of tracks. Also covers the "no audio" empty state.

import { useState } from 'react'

interface Props {
  onStart: (directory: string) => void
}

function SelectFolder({ onStart }: Props): JSX.Element {
  const [directory, setDirectory] = useState<string | null>(null)
  const [counts, setCounts] = useState<{ supported: number; ignored: number } | null>(null)

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
      <p className="eyebrow">Paso 1 de 2</p>
      <h1 className="screen-title">Elegí la carpeta con tus tracks</h1>
      <p className="screen-sub">
        Se analizan los archivos MP3, AIFF y WAV del directorio y se mueven a subcarpetas por
        subgénero.
      </p>

      <div className="field">
        <span className={directory ? 'path mono' : 'path placeholder mono'}>
          {directory ?? 'Ninguna carpeta seleccionada'}
        </span>
        <button className="btn-secondary" onClick={pick}>
          Elegir carpeta…
        </button>
      </div>

      {directory && !noAudio && (
        <>
          <p className="meta">
            {supported} archivo{supported === 1 ? '' : 's'} de audio detectado
            {supported === 1 ? '' : 's'} (MP3, AIFF, WAV)
          </p>
          {counts!.ignored > 0 && (
            <p className="meta">
              {counts!.ignored} archivo{counts!.ignored === 1 ? '' : 's'} ignorado
              {counts!.ignored === 1 ? '' : 's'} por formato no soportado
            </p>
          )}
        </>
      )}

      {noAudio && (
        <p className="meta">
          La carpeta no contiene audio compatible. Elegí un directorio con archivos MP3, AIFF o
          WAV. Los subdirectorios no se recorren.
        </p>
      )}

      <div className="actions">
        <button className="btn-primary" disabled={!directory || noAudio} onClick={() => onStart(directory!)}>
          {supported > 0 ? `Clasificar ${supported} archivos` : 'Clasificar'}
        </button>
        {!noAudio && <span className="note">Los originales se mueven, no se copian.</span>}
      </div>
    </main>
  )
}

export default SelectFolder
