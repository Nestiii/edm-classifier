import { useState } from 'react'

/**
 * Minimal scaffold screen: pick a directory of tracks to classify.
 * Progress, results and auto-organization are wired up in later stories.
 */
function App(): JSX.Element {
  const [directory, setDirectory] = useState<string | null>(null)

  async function handleSelectDirectory(): Promise<void> {
    const selected = await window.api.selectDirectory()
    if (selected) setDirectory(selected)
  }

  return (
    <main className="app">
      <h1>EDM Classifier</h1>
      <p className="subtitle">Clasificación automática de subgéneros de música electrónica</p>

      <section className="picker">
        <button onClick={handleSelectDirectory}>Seleccionar carpeta de tracks…</button>
        {directory && (
          <p className="selected">
            Carpeta seleccionada: <code>{directory}</code>
          </p>
        )}
      </section>
    </main>
  )
}

export default App
