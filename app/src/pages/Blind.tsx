import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { Status } from '../lib/api'
import { drawPaths } from '../lib/ink'
import type { Path2 } from '../lib/ink'

type Trial = Awaited<ReturnType<typeof api.blind>>

function Sample({ strokes, onPick, reveal }: { strokes: Path2[]; onPick: () => void; reveal: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const w = Math.max(...strokes.flat().map((p) => p[0]))
    const h = Math.max(...strokes.flat().map((p) => p[1]))
    const dpr = window.devicePixelRatio || 1
    const scale = ((canvas.clientWidth - 24) / w) * dpr
    canvas.width = canvas.clientWidth * dpr
    canvas.height = h * scale + 24 * dpr
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#fffdf8'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    drawPaths(ctx, strokes, scale, { ox: 12 * dpr, oy: 12 * dpr, width: 0.38 })
  }, [strokes])
  return (
    <button className="sample" onClick={onPick}>
      <canvas ref={ref} />
      {reveal && <span className="pill">{reveal}</span>}
    </button>
  )
}

export default function Blind() {
  const [trial, setTrial] = useState<Trial | null>(null)
  const [picked, setPicked] = useState(-1)
  const [score, setScore] = useState<Status['blind'] | null>(null)
  const [error, setError] = useState('')

  const next = useCallback(async () => {
    setPicked(-1)
    setError('')
    try {
      setTrial(await api.blind())
    } catch (e) {
      setTrial(null)
      setError(String(e).includes('409') ? 'Write the held-out lines on the Capture page first; the test compares against them.' : String(e))
    }
  }, [])

  useEffect(() => {
    void next()
    api.status().then((s) => setScore(s.blind)).catch(() => undefined)
  }, [next])

  const pick = async (i: number) => {
    if (!trial || picked >= 0) return
    setPicked(i)
    setScore(await api.blindResult(trial.options[i].real, trial.text))
  }

  return (
    <div className="page">
      <div className="panel">
        <p>
          One of these two lines is your real held-out handwriting; the other was generated for the same text, which the model never
          trained on. Click the one you believe is real. Better still, hand this screen to someone who knows your writing.
        </p>
        {score && (
          <p className={score.passed ? 'good' : 'muted'}>
            {score.trials} trials, real line identified {score.correct} times
            {score.trials > 0 && ` (${Math.round((100 * score.correct) / score.trials)}%)`}.{' '}
            {score.passed
              ? 'Passed: at 20 or more trials the real line is picked no more than 65% of the time.'
              : 'Pass mark: at least 20 trials with the real line picked no more than 65% of the time.'}
          </p>
        )}
        <div className="row">
          <button onClick={() => void next()}>Next pair</button>
          <button onClick={() => void api.blindReset().then(() => api.status()).then((s) => setScore(s.blind))}>Reset tally</button>
        </div>
        {error && <p className="warn">{error}</p>}
      </div>
      {trial && (
        <div className="samples">
          {trial.options.map((o, i) => (
            <Sample key={`${trial.text}-${i}-${o.strokes.length}`} strokes={o.strokes} onPick={() => void pick(i)} reveal={picked < 0 ? '' : o.real ? 'real' : 'generated'} />
          ))}
        </div>
      )}
    </div>
  )
}
