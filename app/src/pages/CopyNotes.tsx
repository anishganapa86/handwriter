import { useEffect, useRef, useState } from 'react'
import PlotExport from '../PlotExport'
import { api } from '../lib/api'
import type { CopiedNotes } from '../lib/api'
import { download, pathsToSvg } from '../lib/ink'
import type { Path2 } from '../lib/ink'

type Rect = { x0: number; y0: number; x1: number; y1: number }

export default function CopyNotes() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [blank, setBlank] = useState<File | null>(null)
  const [filled, setFilled] = useState<File | null>(null)
  const [blankPage, setBlankPage] = useState(1)
  const [filledPage, setFilledPage] = useState(1)
  const [result, setResult] = useState<CopiedNotes | null>(null)
  const [strokes, setStrokes] = useState<Path2[]>([])
  const [history, setHistory] = useState<Path2[][]>([])
  const [showFilled, setShowFilled] = useState(false)
  const [hover, setHover] = useState(-1)
  const [rect, setRect] = useState<Rect | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const dragStart = useRef<{ x: number; y: number } | null>(null)
  const [reach, setReach] = useState<boolean[]>([])

  useEffect(() => {
    if (!result || !strokes.length) {
      setReach([])
      return
    }
    let live = true
    api
      .reach(strokes, { width: result.page.w, height: result.page.h })
      .then((r) => live && setReach(r.reachable))
      .catch(() => live && setReach([]))
    return () => {
      live = false
    }
  }, [strokes, result])

  const outOfReach = reach.length === strokes.length ? reach.filter((r) => !r).length : 0
  const plotStrokes = reach.length === strokes.length ? strokes.filter((_, k) => reach[k]) : strokes

  const run = async () => {
    if (!blank || !filled) return
    setBusy(true)
    setError('')
    try {
      const out = await api.copyNotes(blank, filled, blankPage - 1, filledPage - 1)
      setResult(out)
      setStrokes(out.strokes)
      setHistory([])
    } catch (e) {
      setResult(null)
      setError(String(e).replace(/^Error: \d+ /, '').replace(/^\{"detail":"(.*)"\}$/, '$1'))
    } finally {
      setBusy(false)
    }
  }

  const change = (next: Path2[]) => {
    setHistory((h) => [...h, strokes])
    setStrokes(next)
    setHover(-1)
  }
  const undo = () => {
    const last = history[history.length - 1]
    if (!last) return
    setStrokes(last)
    setHistory((h) => h.slice(0, -1))
  }

  const toMm = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect()
    const k = result!.page.w / r.width
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k }
  }
  const onDown = (e: React.PointerEvent) => {
    if (!result) return
    svgRef.current!.setPointerCapture(e.pointerId)
    dragStart.current = toMm(e)
    setRect(null)
  }
  const onMove = (e: React.PointerEvent) => {
    const s = dragStart.current
    if (!s) return
    const p = toMm(e)
    setRect({ x0: Math.min(s.x, p.x), y0: Math.min(s.y, p.y), x1: Math.max(s.x, p.x), y1: Math.max(s.y, p.y) })
  }
  const onUp = () => {
    const r = rect
    dragStart.current = null
    setRect(null)
    if (!r || r.x1 - r.x0 < 1 || r.y1 - r.y0 < 1) return
    const inside = (s: Path2) => s.every(([x, y]) => x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1)
    const keep = strokes.filter((s) => !inside(s))
    if (keep.length !== strokes.length) change(keep)
  }

  const points = (s: Path2) => (s.length === 1 ? [s[0], [s[0][0] + 0.2, s[0][1]]] : s).map((q) => q.join(',')).join(' ')

  return (
    <div className="page split">
      <div className="side">
        <p className="muted small">
          Copies the handwriting from a filled-in copy of a handout onto a blank copy of the same handout, stroke for stroke.
          Printed text, highlighter and pen colour are ignored; everything is written with the pen in the holder.
        </p>
        <label className="file">
          {blank ? `Blank: ${blank.name}` : '1. Choose the blank handout (PDF or photo)'}
          <input type="file" accept="image/*,application/pdf" onChange={(e) => setBlank(e.target.files?.[0] ?? null)} />
        </label>
        {blank?.name.toLowerCase().endsWith('.pdf') && (
          <label className="row small">
            Page of the blank PDF
            <input type="number" min={1} value={blankPage} onChange={(e) => setBlankPage(Math.max(1, +e.target.value || 1))} style={{ width: 64 }} />
          </label>
        )}
        <label className="file">
          {filled ? `Filled in: ${filled.name}` : '2. Choose your filled-in copy (photo or PDF)'}
          <input type="file" accept="image/*,application/pdf" onChange={(e) => setFilled(e.target.files?.[0] ?? null)} />
        </label>
        {filled?.name.toLowerCase().endsWith('.pdf') && (
          <label className="row small">
            Page of the filled-in PDF
            <input type="number" min={1} value={filledPage} onChange={(e) => setFilledPage(Math.max(1, +e.target.value || 1))} style={{ width: 64 }} />
          </label>
        )}
        <div className="row">
          <button className="primary" disabled={!blank || !filled || busy} onClick={() => void run()}>
            {busy ? 'Copying…' : '3. Copy my notes'}
          </button>
        </div>
        {error && <p className="warn">{error}</p>}
        {result && (
          <>
            <p className="muted small">
              Lined up from {result.matches} matching points. {strokes.length} strokes. Check the page: click a stray mark to remove
              it, or drag a box around several (only marks wholly inside the box go).
            </p>
            <div className="row">
              <button disabled={!history.length} onClick={undo}>
                Undo
              </button>
              <button disabled={!history.length} onClick={() => change(result.strokes)}>
                Restore all
              </button>
              <label className="check">
                <input type="checkbox" checked={showFilled} onChange={(e) => setShowFilled(e.target.checked)} /> Show my filled copy
              </label>
            </div>
            <div className="row">
              <button
                disabled={!strokes.length}
                onClick={() => download('copied-notes.svg', new Blob([pathsToSvg(strokes, result.page.w, result.page.h)], { type: 'image/svg+xml' }))}
              >
                Download ink SVG
              </button>
            </div>
            {outOfReach > 0 && (
              <p className="warn small">
                {outOfReach} {outOfReach === 1 ? 'stroke is' : 'strokes are'} out of the pen's reach (shown in red) and will be left out
                of the plotter file. Write {outOfReach === 1 ? 'it' : 'them'} by hand afterwards, or move the paper's back stop on the
                Plotter tab so the pen can reach more of the sheet.
              </p>
            )}
            {plotStrokes.length > 0 && (
              <PlotExport pages={[plotStrokes]} page={{ width: result.page.w, height: result.page.h }} title="copied-notes" />
            )}
          </>
        )}
      </div>
      {result ? (
        <svg
          ref={svgRef}
          className="sheet placing"
          viewBox={`0 0 ${result.page.w} ${result.page.h}`}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
        >
          <image href={result.image} width={result.page.w} height={result.page.h} />
          {showFilled && <image href={result.filled} width={result.page.w} height={result.page.h} opacity={0.45} />}
          {strokes.map((s, k) => (
            <polyline key={k} className={k === hover ? 'ink hot' : reach.length === strokes.length && !reach[k] ? 'ink out' : 'ink'} points={points(s)} />
          ))}
          {strokes.map((s, k) => (
            <polyline
              key={`hit${k}`}
              className="hit"
              points={points(s)}
              onPointerEnter={() => setHover(k)}
              onPointerLeave={() => setHover(-1)}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={() => change(strokes.filter((_, i) => i !== k))}
            />
          ))}
          {rect && <rect className="erase" x={rect.x0} y={rect.y0} width={rect.x1 - rect.x0} height={rect.y1 - rect.y0} />}
        </svg>
      ) : (
        <div className="sheet empty">The blank handout with your copied notes appears here.</div>
      )}
    </div>
  )
}
