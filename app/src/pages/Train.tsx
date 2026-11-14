import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { Status } from '../lib/api'

const fmt = (v: number | null | undefined) => (v === null || v === undefined ? 'n/a' : v.toFixed(4))

export default function Train() {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const load = () => api.status().then(setStatus).catch((e) => setError(String(e)))
    void load()
    const timer = setInterval(load, 5000)
    return () => clearInterval(timer)
  }, [])

  if (!status) return <div className="page">{error ? <p className="warn">Trainer not reachable: {error}</p> : <p className="muted">Loading…</p>}</div>
  const m = status.champion?.metrics
  const remaining = Math.max(0, status.retrain_every - status.new_lines)

  return (
    <div className="page">
      <div className="stats">
        <div>
          <b>{status.lines}</b>
          <span>lines collected</span>
        </div>
        <div>
          <b>{status.train}</b>
          <span>used for training</span>
        </div>
        <div>
          <b>{status.holdout}</b>
          <span>held out, never trained on</span>
        </div>
        <div>
          <b>{status.photo}</b>
          <span>words from photos</span>
        </div>
        <div>
          <b>{status.champion?.version ?? 'base'}</b>
          <span>current model</span>
        </div>
      </div>
      <div className="panel">
        <p>
          {status.training
            ? 'Training is running.'
            : remaining === 0
              ? 'Enough new lines for a retrain.'
              : `${status.new_lines} new lines since the last retrain; the next one starts automatically after ${remaining} more.`}
        </p>
        {m && (
          <p className="muted">
            Champion held-out DTW {fmt(m.dtw_dir)}, recognizer CER {fmt(m.cer)} (your real lines {fmt(m.cer_real)}).
          </p>
        )}
        <div className="row">
          <button className="primary" disabled={status.training} onClick={() => void api.train().then(() => api.status()).then(setStatus)}>
            Retrain now
          </button>
          {status.reports
            .slice()
            .reverse()
            .map((r) => (
              <a key={r} href={`/files/reports/${r}`} target="_blank" rel="noreferrer">
                {r.replace('.html', '')} report
              </a>
            ))}
        </div>
        {status.log && <pre className="log">{status.log}</pre>}
        {status.habits.words ? (
          <p className="muted">
            Habits measured from {status.habits.words} traced words on {status.habits.lines} lines: slant{' '}
            {status.habits.slant?.toFixed(0)}°, word gap {status.habits.gap?.toFixed(1)} letter-heights, letters joined{' '}
            {Math.round((status.habits.join ?? 0) * 100)}% of the time. The My letters engine follows these.
          </p>
        ) : null}
      </div>
    </div>
  )
}
