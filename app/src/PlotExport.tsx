import { useState } from 'react'
import { api } from './lib/api'
import type { PlotPart } from './lib/api'
import type { Path2 } from './lib/ink'

type Props = { pages: Path2[][]; current?: number; page: { width: number; height: number }; title: string }

export default function PlotExport({ pages, current = 0, page, title }: Props) {
  const [parts, setParts] = useState<PlotPart[]>([])
  const [problems, setProblems] = useState<string[]>([])
  const [dryRun, setDryRun] = useState(true)
  const [all, setAll] = useState(true)
  const [busy, setBusy] = useState(false)
  const filled = pages.filter((p) => p.length > 0)
  const many = filled.length > 1
  const chosen = many && all ? pages : [pages[current] ?? []]

  const run = async () => {
    setBusy(true)
    try {
      const out = await api.plot(chosen, page, many && all ? title : `${title}-${current + 1}`, dryRun)
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
        <button onClick={() => void run()} disabled={busy || !chosen.some((p) => p.length)}>
          {busy ? 'Preparing…' : many && all ? `Make plotter file for all ${filled.length} sheets` : 'Make plotter file'}
        </button>
        <label className="check">
          <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} /> Dry run (pen 5 mm above the paper)
        </label>
        {many && (
          <label className="check">
            <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> All sheets in one job
          </label>
        )}
      </div>
      {problems.map((p) => (
        <p key={p} className="warn">
          {p}
        </p>
      ))}
      {parts.map((part) => (
        <div key={part.url}>
          <p className="small">
            <a href={part.url} download={part.name}>
              {part.name}
            </a>{' '}
            — {part.sheets} sheet{part.sheets > 1 ? 's' : ''}, {part.strokes} strokes, about {part.minutes} min of writing.{' '}
            {part.flips === 0 ? 'No sheet needs turning.' : `${part.flips} sheet${part.flips > 1 ? 's need' : ' needs'} turning part-way.`}
          </p>
          <ol className="steps small">
            {part.steps.map((s) => (
              <li key={s}>{s.replace(/^Pause \d+: /, '').replace(/^./, (c) => c.toUpperCase())}</li>
            ))}
          </ol>
        </div>
      ))}
      {parts.length > 0 && (
        <p className="muted small">
          The printer pauses once for each numbered step, in this order, with the head parked at the back and the bed lowered. Open the file in
          Bambu Studio and print it with bed levelling, flow calibration and timelapse unticked. Steps are on the Plotter tab.
        </p>
      )}
    </div>
  )
}
