import { useRef, useState } from 'react'
import EnginePicker from '../EnginePicker'
import PlotExport from '../PlotExport'
import SliderPanel from '../SliderPanel'
import { api, defaultSliders } from '../lib/api'
import type { Box, Engine, Sliders } from '../lib/api'
import { download, pathsToSvg } from '../lib/ink'
import type { Path2 } from '../lib/ink'

type Sheet = { id: string; image: string; page: { w: number; h: number }; deskew: string; pages?: number }
type Drag = { index: number; mode: 'move' | 'resize'; startX: number; startY: number; origin: Box }
type Props = { mode?: 'worksheet' | 'copy' }

export default function Worksheet({ mode = 'worksheet' }: Props) {
  const copy = mode === 'copy'
  const svgRef = useRef<SVGSVGElement>(null)
  const fileRef = useRef<File | null>(null)
  const [sheet, setSheet] = useState<Sheet | null>(null)
  const [boxes, setBoxes] = useState<Box[]>([])
  const [answers, setAnswers] = useState<Path2[][]>([])
  const [diagram, setDiagram] = useState<Path2[]>([])
  const [showDiagram, setShowDiagram] = useState(true)
  const [showSource, setShowSource] = useState(true)
  const [pageIndex, setPageIndex] = useState(0)
  const [selected, setSelected] = useState(-1)
  const [sliders, setSliders] = useState<Sliders>(defaultSliders)
  const [busy, setBusy] = useState(false)
  const [engine, setEngine] = useState<Engine>('model')
  const [error, setError] = useState('')
  const drag = useRef<Drag | null>(null)

  const upload = async (file?: File | null, page = 0) => {
    if (!file) return
    fileRef.current = file
    setBusy(true)
    setError('')
    try {
      if (copy) {
        const out = await api.copyPage(file, page)
        setSheet(out)
        setBoxes(out.boxes)
        setDiagram(out.diagram)
        setPageIndex(page)
      } else {
        const out = await api.worksheet(file)
        setSheet(out)
        setBoxes(out.boxes.map((b) => ({ ...b, text: '' })))
      }
      setAnswers([])
      setSelected(-1)
      setShowSource(true)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const toMm = (e: React.PointerEvent) => {
    const rect = svgRef.current!.getBoundingClientRect()
    const k = sheet!.page.w / rect.width
    return { x: (e.clientX - rect.left) * k, y: (e.clientY - rect.top) * k }
  }

  const startDrag = (e: React.PointerEvent, index: number, kind: Drag['mode']) => {
    e.stopPropagation()
    svgRef.current!.setPointerCapture(e.pointerId)
    const p = toMm(e)
    drag.current = { index, mode: kind, startX: p.x, startY: p.y, origin: boxes[index] }
    setSelected(index)
  }

  const onMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const p = toMm(e)
    const dx = p.x - d.startX
    const dy = p.y - d.startY
    setBoxes((all) =>
      all.map((b, i) =>
        i !== d.index
          ? b
          : d.mode === 'move'
            ? { ...b, x: +(d.origin.x + dx).toFixed(2), y: +(d.origin.y + dy).toFixed(2) }
            : { ...b, w: +Math.max(4, d.origin.w + dx).toFixed(2), h: +Math.max(2.5, d.origin.h + dy).toFixed(2), size: undefined },
      ),
    )
  }

  const addBox = (e: React.MouseEvent) => {
    if (!sheet) return
    const rect = svgRef.current!.getBoundingClientRect()
    const k = sheet.page.w / rect.width
    const box: Box = { x: +((e.clientX - rect.left) * k).toFixed(2), y: +((e.clientY - rect.top) * k - 3.5).toFixed(2), w: 40, h: 7, kind: 'line', text: '' }
    setBoxes((all) => [...all, box])
    setSelected(boxes.length)
  }

  const fill = async () => {
    if (!sheet) return
    setBusy(true)
    setError('')
    try {
      const out = await api.fill(boxes, sliders, Math.floor(Math.random() * 1e9), engine)
      setAnswers(out.answers.map((a) => a.strokes))
      if (copy) {
        setDiagram((await api.copyDiagram(sheet.id, boxes)).diagram)
        setShowSource(false)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const setText = (index: number, text: string) => setBoxes((all) => all.map((b, i) => (i === index ? { ...b, text } : b)))
  const remove = (index: number) => {
    setBoxes((all) => all.filter((_, i) => i !== index))
    setAnswers((all) => all.filter((_, i) => i !== index))
    setSelected(-1)
  }

  const ink = [...(copy && showDiagram ? diagram : []), ...answers.flat()]

  return (
    <div className="page split">
      <div className="side">
        <label className="file">
          {busy ? 'Working…' : copy ? 'Choose an image or PDF to copy' : 'Photograph or choose a worksheet'}
          <input
            type="file"
            accept={copy ? 'image/*,application/pdf' : 'image/*'}
            onChange={(e) => void upload(e.target.files?.[0])}
          />
        </label>
        {sheet && !copy && (
          <p className="muted small">
            Deskewed by {sheet.deskew}. {boxes.length} answer blanks suggested. Drag a box to move it, drag its corner to resize,
            double-click the sheet to add one.
          </p>
        )}
        {sheet && copy && (
          <p className="muted small">
            Found {boxes.length} pieces of text and traced the drawing around them. The text I read is a first guess: correct it in
            the list, delete boxes that are not text, and double-click the page to add a box over text I missed. Anything not inside
            a box is traced as drawing.
          </p>
        )}
        {sheet && copy && (sheet.pages ?? 1) > 1 && (
          <div className="row">
            <span className="muted small">Page</span>
            {Array.from({ length: sheet.pages ?? 1 }, (_, i) => (
              <button key={i} className={i === pageIndex ? 'primary tiny' : 'tiny'} disabled={busy} onClick={() => void upload(fileRef.current, i)}>
                {i + 1}
              </button>
            ))}
          </div>
        )}
        <EnginePicker engine={engine} onChange={setEngine} />
        <SliderPanel sliders={sliders} onChange={setSliders} />
        {copy && sheet && (
          <div className="row">
            <label className="check">
              <input type="checkbox" checked={showSource} onChange={(e) => setShowSource(e.target.checked)} /> Show original
            </label>
            <label className="check">
              <input type="checkbox" checked={showDiagram} onChange={(e) => setShowDiagram(e.target.checked)} /> Include drawing
            </label>
          </div>
        )}
        <div className="answers">
          {boxes.map((b, i) => (
            <div key={i} className={i === selected ? 'answer active' : 'answer'} onFocus={() => setSelected(i)}>
              <span className="num">{i + 1}</span>
              <input value={b.text ?? ''} placeholder={copy ? 'text' : 'answer'} onChange={(e) => setText(i, e.target.value)} />
              <button className="tiny" onClick={() => remove(i)}>
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="row">
          <button className="primary" disabled={!sheet || busy || !boxes.some((b) => b.text?.trim())} onClick={() => void fill()}>
            {copy ? 'Write this page' : 'Write answers'}
          </button>
          <button
            disabled={!ink.length || !sheet}
            onClick={() => sheet && download(copy ? 'copied-page.svg' : 'worksheet-answers.svg', new Blob([pathsToSvg(ink, sheet.page.w, sheet.page.h)], { type: 'image/svg+xml' }))}
          >
            Download ink SVG
          </button>
        </div>
        {error && <p className="warn">{error}</p>}
        {sheet && ink.length > 0 && <PlotExport paths={ink} page={{ width: sheet.page.w, height: sheet.page.h }} title={copy ? 'copied-page' : 'worksheet'} />}
      </div>
      {sheet ? (
        <svg
          ref={svgRef}
          className="sheet"
          viewBox={`0 0 ${sheet.page.w} ${sheet.page.h}`}
          onPointerMove={onMove}
          onPointerUp={() => (drag.current = null)}
          onDoubleClick={addBox}
        >
          {showSource && <image href={sheet.image} width={sheet.page.w} height={sheet.page.h} opacity={copy && answers.length ? 0.35 : 1} />}
          {ink.map((p, k) => (
            <polyline key={k} className="ink" points={p.map((q) => q.join(',')).join(' ')} />
          ))}
          {(showSource || !copy) &&
            boxes.map((b, i) => (
              <g key={i}>
                <rect
                  className={i === selected ? 'box active' : 'box'}
                  x={b.x}
                  y={b.y}
                  width={b.w}
                  height={b.h}
                  onPointerDown={(e) => startDrag(e, i, 'move')}
                />
                <text x={b.x + 0.6} y={b.y - 0.8} className="box-num">
                  {i + 1}
                </text>
                <rect className="handle" x={b.x + b.w - 1.6} y={b.y + b.h - 1.6} width={3.2} height={3.2} onPointerDown={(e) => startDrag(e, i, 'resize')} />
              </g>
            ))}
        </svg>
      ) : (
        <div className="sheet empty">{copy ? 'The page to copy appears here.' : 'The deskewed worksheet appears here.'}</div>
      )}
    </div>
  )
}
