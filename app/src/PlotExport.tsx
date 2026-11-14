import { useState } from 'react'
import { api } from './lib/api'
import type { PlotPart } from './lib/api'
import type { Path2 } from './lib/ink'

type Props = { paths: Path2[]; page: { width: number; height: number }; title: string }

export default function PlotExport({ paths, page, title }: Props) {
  const [parts, setParts] = useState<PlotPart[]>([])
  const [problems, setProblems] = useState<string[]>([])
  const [dryRun, setDryRun] = useState(true)
  const [busy, setBusy] = useState(false)

  const run = async () => {
    setBusy(true)
    try {
      const out = await api.plot(paths, page, title, dryRun)
      setParts(out.parts)
      setProblems(out.problems)
    } catch (e) {
      setParts([])
      setProblems([String(e)])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel">
      <div className="row">
        <button onClick={() => void run()} disabled={busy || !paths.length}>
          {busy ? 'Preparing…' : 'Make plotter file'}
        </button>
        <label className="check">
          <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} /> Dry run (pen 5 mm above the paper)
        </label>
      </div>
      {problems.map((p) => (
        <p key={p} className="warn">
          {p}
        </p>
      ))}
      {parts.map((part, i) => (
        <p key={part.url} className="small">
          <a href={part.url} download={part.name}>
            {part.name}
          </a>{' '}
          — {part.strokes} strokes, about {part.minutes} min.{' '}
          {parts.length > 1 && `Pass ${i + 1} of ${parts.length}. `}
          {part.turned
            ? 'Turn the paper upside down (bottom edge against the back stop) before this pass.'
            : 'Paper the normal way up, top edge against the back stop.'}
        </p>
      ))}
      {parts.length > 0 && (
        <p className="muted small">Open the file in Bambu Studio and print it with bed levelling, flow calibration and timelapse unticked. Steps are on the Plotter tab.</p>
      )}
    </div>
  )
}
