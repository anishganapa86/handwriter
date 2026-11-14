import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

type Pt = [number, number]

const GROUPS = [
  { name: 'Lowercase', chars: 'abcdefghijklmnopqrstuvwxyz' },
  { name: 'Capitals', chars: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' },
  { name: 'Digits', chars: '0123456789' },
  { name: 'Punctuation', chars: '.,;:!?\'"()-/&@#%$' },
  { name: 'Maths', chars: '+=*<>[]{}|~_^×÷±' },
  { name: 'Extra maths (optional)', chars: 'πθ√∞≤≥≠≈→∈∑∫Δαβλμσ°' },
]
const ALL = GROUPS.flatMap((g) => [...g.chars])
const CELLS = 4
const SIZES = [
  { label: 'Small (3 mm letters)', x: 3, cell: 16, height: 26, baseline: 17 },
  { label: 'Medium (5 mm letters)', x: 5, cell: 24, height: 38, baseline: 25 },
  { label: 'Large (8 mm letters)', x: 8, cell: 36, height: 58, baseline: 38 },
]

const storedPxPerMm = () => Number(localStorage.getItem('hw.pxPerMm')) || 96 / 25.4

export default function Glyphs() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pxPerMm = storedPxPerMm()
  const [sizeIndex, setSizeIndex] = useState(() => Number(localStorage.getItem('hw.glyphSize') ?? 1))
  const size = SIZES[sizeIndex] ?? SIZES[1]
  const [chars, setChars] = useState<string[]>(ALL)
  const [index, setIndex] = useState(0)
  const [counts, setCounts] = useState<Record<string, number>>({})
  const [strokes, setStrokes] = useState<Pt[][]>([])
  const [extra, setExtra] = useState('')
  const [note, setNote] = useState('')
  const [sheetNote, setSheetNote] = useState('')
  const [importing, setImporting] = useState(false)
  const live = useRef<{ stroke: Pt[] | null; penSeen: boolean }>({ stroke: null, penSeen: false })
  const char = chars[index]

  useEffect(() => {
    api
      .status()
      .then((s) => {
        setCounts(s.glyphs)
        const more = Object.keys(s.glyphs).filter((c) => !ALL.includes(c))
        if (more.length) setChars([...ALL, ...more])
        const next = ALL.findIndex((c) => !s.glyphs[c])
        setIndex(next < 0 ? 0 : next)
      })
      .catch(() => setNote('Server not reachable.'))
  }, [])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const s = pxPerMm * dpr
    canvas.width = Math.round(size.cell * CELLS * s)
    canvas.height = Math.round(size.height * s)
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = dpr
    ctx.strokeStyle = '#d9dee7'
    for (let c = 1; c < CELLS; c++) {
      ctx.beginPath()
      ctx.moveTo(c * size.cell * s, 0)
      ctx.lineTo(c * size.cell * s, canvas.height)
      ctx.stroke()
    }
    ctx.strokeStyle = '#9db4d6'
    ctx.beginPath()
    ctx.moveTo(0, size.baseline * s)
    ctx.lineTo(canvas.width, size.baseline * s)
    ctx.stroke()
    ctx.strokeStyle = '#e1e8f2'
    ctx.setLineDash([6 * dpr, 6 * dpr])
    ctx.beginPath()
    ctx.moveTo(0, (size.baseline - size.x) * s)
    ctx.lineTo(canvas.width, (size.baseline - size.x) * s)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = '#14213d'
    ctx.fillStyle = '#14213d'
    ctx.lineWidth = Math.max(0.3, size.x * 0.09) * s
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    const all = live.current.stroke ? [...strokes, live.current.stroke] : strokes
    for (const pts of all) {
      ctx.beginPath()
      if (pts.length === 1) {
        ctx.arc(pts[0][0] * s, pts[0][1] * s, ctx.lineWidth / 1.5, 0, Math.PI * 2)
        ctx.fill()
        continue
      }
      ctx.moveTo(pts[0][0] * s, pts[0][1] * s)
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * s, pts[i][1] * s)
      ctx.stroke()
    }
  }, [strokes, pxPerMm, size])

  useEffect(redraw, [redraw])

  const point = (e: PointerEvent, rect: DOMRect): Pt => [
    +((e.clientX - rect.left) / pxPerMm).toFixed(3),
    +((e.clientY - rect.top) / pxPerMm).toFixed(3),
  ]

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'pen') live.current.penSeen = true
    if (live.current.penSeen && e.pointerType !== 'pen') return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    live.current.stroke = [point(e.nativeEvent, e.currentTarget.getBoundingClientRect())]
    redraw()
  }

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const stroke = live.current.stroke
    if (!stroke) return
    e.preventDefault()
    const rect = e.currentTarget.getBoundingClientRect()
    const native = e.nativeEvent
    const events = typeof native.getCoalescedEvents === 'function' ? native.getCoalescedEvents() : []
    for (const ev of events.length ? events : [native]) stroke.push(point(ev, rect))
    redraw()
  }

  const onUp = () => {
    const stroke = live.current.stroke
    if (!stroke) return
    live.current.stroke = null
    setStrokes((prev) => [...prev, stroke])
  }

  const go = useCallback(
    (to: number) => {
      setStrokes([])
      setIndex(Math.max(0, Math.min(chars.length - 1, to)))
    },
    [chars.length],
  )

  const save = useCallback(async () => {
    if (!char || !strokes.length) return go(index + 1)
    const variants: Pt[][][] = Array.from({ length: CELLS }, () => [])
    for (const stroke of strokes) {
      const centre = stroke.reduce((sum, p) => sum + p[0], 0) / stroke.length
      variants[Math.max(0, Math.min(CELLS - 1, Math.floor(centre / size.cell)))].push(stroke)
    }
    try {
      const res = await api.saveGlyph(char, variants.filter((v) => v.length), size.baseline)
      setCounts(res.glyphs)
      setNote('')
      go(index + 1)
    } catch (e) {
      setNote(`Could not save: ${String(e)}`)
    }
  }, [char, strokes, size, index, go])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return
      if (e.key === 'Enter') void save()
      if (e.key === 'Backspace') setStrokes((s) => s.slice(0, -1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [save])

  const addExtra = () => {
    const fresh = [...extra].filter((c) => c.trim() && !chars.includes(c))
    if (!fresh.length) return
    setChars((c) => [...c, ...fresh])
    setStrokes([])
    setIndex(chars.length)
    setExtra('')
  }

  const importSheets = async (files: FileList | null) => {
    setImporting(true)
    const lines: string[] = []
    for (const file of Array.from(files ?? [])) {
      try {
        const out = await api.importTemplate(file)
        setCounts(out.glyphs)
        for (const r of out.results)
          lines.push(
            r.page === null
              ? `${file.name}: could not find the four corner squares. Retake the photo with the whole sheet in view.`
              : `${file.name}: sheet ${r.page}, ${r.variants} samples of ${r.units} characters read${r.empty?.length ? `; left blank: ${r.empty.join(' ')}` : ''}.`,
          )
      } catch (e) {
        lines.push(`${file.name}: ${String(e)}`)
      }
    }
    setSheetNote(lines.join(' '))
    setImporting(false)
  }

  const have = chars.filter((c) => counts[c]).length

  return (
    <div className="page">
      <div className="row between">
        <div>
          <span className="pill">
            {have} of {chars.length} characters
          </span>
          <span className="pill muted">{counts[char] ? `${counts[char]} saved for this one` : 'not written yet'}</span>
        </div>
        <select
          value={sizeIndex}
          onChange={(e) => {
            setSizeIndex(+e.target.value)
            localStorage.setItem('hw.glyphSize', e.target.value)
            setStrokes([])
          }}
        >
          {SIZES.map((s, i) => (
            <option key={s.label} value={i}>
              {s.label}
            </option>
          ))}
        </select>
      </div>
      <div className="panel">
        <p>
          <b>Fastest way: print and fill in the letter sheets.</b> Use a fine dark pen, write at your normal speed and a little larger
          than usual, and keep each one inside its box. Then photograph each sheet flat with all four corner squares visible and add
          the photos here. Each character you fill in replaces what was stored for it before.
        </p>
        <div className="row">
          <a className="file" href="/api/template.pdf" target="_blank" rel="noreferrer">
            Open the letter sheets (PDF)
          </a>
          <label className="file">
            {importing ? 'Reading…' : 'Add photos of filled sheets'}
            <input type="file" accept="image/*,application/pdf" multiple onChange={(e) => void importSheets(e.target.files)} />
          </label>
        </div>
        {sheetNote && <p className="muted small">{sheetNote}</p>}
      </div>
      <div className="glyph-stage">
        <div className="glyph-char">{char}</div>
        <div>
          <p className="muted small">
            Write this character once in each box, sitting on the solid line, the way you normally write it. Small letters reach the
            dashed line. Two or three boxes are enough.
          </p>
          <canvas
            ref={canvasRef}
            className="pad glyph-pad"
            style={{ width: size.cell * CELLS * pxPerMm, height: size.height * pxPerMm }}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={onUp}
          />
        </div>
      </div>
      <div className="row">
        <button onClick={() => go(index - 1)} disabled={index === 0}>
          Back
        </button>
        <button onClick={() => setStrokes((s) => s.slice(0, -1))} disabled={!strokes.length}>
          Undo stroke
        </button>
        <button onClick={() => setStrokes([])} disabled={!strokes.length}>
          Clear
        </button>
        <button onClick={() => go(index + 1)}>Skip</button>
        <button className="primary" onClick={() => void save()} disabled={!strokes.length}>
          {counts[char] ? 'Replace and next' : 'Save and next'}
        </button>
        <span className="muted small">Enter saves, Backspace undoes a stroke.</span>
        {note && <span className="warn">{note}</span>}
      </div>
      <div className="panel">
        {GROUPS.map((g) => (
          <div key={g.name} className="glyph-group">
            <span className="muted small">{g.name}</span>
            <div className="glyph-grid">
              {[...g.chars].map((c) => (
                <button key={c} className={`glyph-key${counts[c] ? ' done' : ''}${c === char ? ' on' : ''}`} onClick={() => go(chars.indexOf(c))}>
                  {c}
                </button>
              ))}
            </div>
          </div>
        ))}
        {chars.length > ALL.length && (
          <div className="glyph-group">
            <span className="muted small">Your additions</span>
            <div className="glyph-grid">
              {chars.slice(ALL.length).map((c) => (
                <button key={c} className={`glyph-key${counts[c] ? ' done' : ''}${c === char ? ' on' : ''}`} onClick={() => go(chars.indexOf(c))}>
                  {c}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="row">
          <input className="grow" placeholder="Add other symbols you use, e.g. ∂ ∪ ∩ ⇒" value={extra} onChange={(e) => setExtra(e.target.value)} />
          <button onClick={addExtra} disabled={!extra.trim()}>
            Add symbols
          </button>
        </div>
      </div>
    </div>
  )
}
