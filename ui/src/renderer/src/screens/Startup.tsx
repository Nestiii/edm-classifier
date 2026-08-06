// Screen 04: startup states — loading the model, or backend not reachable.

interface Props {
  error: boolean
  onRetry: () => void
}

function Startup({ error, onRetry }: Props): JSX.Element {
  if (error) {
    return (
      <main className="app">
        <p className="eyebrow">No se pudo iniciar</p>
        <h1 className="screen-title">El módulo de clasificación no responde</h1>
        <p className="screen-sub">
          La interfaz no logró conectarse al servicio local de análisis. Verificá que esté
          instalado y volvé a intentar.
        </p>
        <div className="actions">
          <button className="btn-primary" onClick={onRetry}>
            Reintentar
          </button>
        </div>
      </main>
    )
  }

  return (
    <main className="app">
      <p className="eyebrow">Iniciando</p>
      <h1 className="screen-title">Cargando el modelo de clasificación</h1>
      <div className="progress-track">
        <div className="progress-fill indeterminate" />
      </div>
      <p className="meta">Puede tardar unos segundos la primera vez.</p>
    </main>
  )
}

export default Startup
