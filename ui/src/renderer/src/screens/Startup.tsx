// Screen 04: startup + error states.
//  - loading the model (optionally showing the device)
//  - backend not reachable (retry)
//  - a job that failed (show the error, go back)

interface Props {
  error: boolean
  onRetry: () => void
  device?: string | null
  jobError?: string | null
  onBack?: () => void
}

function Startup({ error, onRetry, device, jobError, onBack }: Props): JSX.Element {
  if (jobError) {
    return (
      <main className="app">
        <p className="eyebrow">Something went wrong</p>
        <h1 className="screen-title">We couldn't finish</h1>
        <p className="screen-sub">{jobError}</p>
        <div className="actions">
          <button className="btn-primary" onClick={onBack ?? onRetry}>
            Back
          </button>
        </div>
      </main>
    )
  }

  if (error) {
    return (
      <main className="app">
        <p className="eyebrow">No connection</p>
        <h1 className="screen-title">Couldn't reach the classifier</h1>
        <p className="screen-sub">
          The analysis service isn't available. Make sure it's running and try again.
        </p>
        <div className="actions">
          <button className="btn-primary" onClick={onRetry}>
            Retry
          </button>
        </div>
      </main>
    )
  }

  return (
    <main className="app">
      <p className="eyebrow">Subgenre Sorter</p>
      <h1 className="screen-title">Getting ready</h1>
      <div className="progress-track">
        <div className="progress-fill indeterminate" />
      </div>
      <p className="meta">
        One moment.{device ? ` Running on ${device.toUpperCase()}.` : ''}
      </p>
    </main>
  )
}

export default Startup
