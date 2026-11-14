import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import script from '../script.json'
import { api } from '../lib/api'
import { download, lineToIamXml, zip } from '../lib/ink'
import type { CaptureLine, CaptureSession, Point, Stroke } from '../lib/ink'

type Prompt = { id: string; text: string; holdout: boolean }

const CARD_MM = 85.6
const PAD_MM = { height: 46, baseline: 30, xHeight: 4.5 }

const storedPxPerMm = () => Number(localStorage.getItem('hw.pxPerMm')) || 96 / 25.4

function newSessionId() {
  return `s${new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)}`
}

export default function Capture() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [pxPerMm, setPxPerMm] = useState(storedPxPerMm)
  const [calibrating, setCalibrating] = useState(false)
  const [prompts, setPrompts] = useState<Prompt[]>(script.lines)
  const [index, setIndex] = useState(0)
  const [lines, setLines] = useState<Record<string, CaptureLine>>({})
  const [strokes, setStrokes] = useState<Stroke[]>([])
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('hw.sessionId') || newSessionId())
  const [note, setNote] = useState('')
  const [custom, setCustom] = useState('')
  const [customHoldout, setCustomHoldout] = useState(false)
  const live = useRef<{ stroke: Point[] | null; t0: number | null; penSeen: boolean; pointerType: string }>({
    stroke: null,
    t0: null,
    penSeen: false,
    pointerType: 'unknown',
  })

  const prompt = prompts[index]
  const done = Object.keys(lines).length

  useEffect(() => {
    localStorage.setItem('hw.sessionId', sessionId)
    api
      .loadSession(sessionId)
      .then((raw) => {
        const session = raw as CaptureSession
        const restored: Record<string, CaptureLine> = {}
        const extra: Prompt[] = []
        for (const line of session.lines) {
          restored[line.promptId] = line
          if (!script.lines.some((p) => p.id === line.promptId)) extra.push({ id: line.promptId, text: line.text, holdout: line.holdout })
        }
        setLines(restored)
        if (extra.length) setPrompts([...script.lines, ...extra])
        const next = script.lines.findIndex((p) => !restored[p.id])
        setIndex(next < 0 ? 0 : next)
        setNote(`Resumed session ${sessionId} with ${session.lines.length} lines.`)
      })
      .catch(() => undefined)
  }, [sessionId])

  useEffect(() => {
    setStrokes(lines[prompt?.id]?.strokes ?? [])
    live.current.t0 = null
  }, [index, prompt?.id])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
      canvas.width = Math.round(rect.width * dpr)
      canvas.height = Math.round(rect.height * dpr)
    }
    const ctx = canvas.getContext('2d')!
    const s = pxPerMm * dpr
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = dpr
    ctx.strokeStyle = '#9db4d6'
    ctx.beginPath()
    ctx.moveTo(0, PAD_MM.baseline * s)
    ctx.lineTo(canvas.width, PAD_MM.baseline * s)
    ctx.stroke()
    ctx.strokeStyle = '#e1e8f2'
    ctx.setLineDash([6 * dpr, 6 * dpr])
    ctx.beginPath()
    ctx.moveTo(0, (PAD_MM.baseline - PAD_MM.xHeight) * s)
    ctx.lineTo(canvas.width, (PAD_MM.baseline - PAD_MM.xHeight) * s)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = '#14213d'
    ctx.fillStyle = '#14213d'
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    const all = live.current.stroke ? [...strokes, { points: live.current.stroke }] : strokes
    for (const stroke of all) {
      const pts = stroke.points
      if (!pts.length) continue
      if (pts.length === 1) {
        ctx.beginPath()
        ctx.arc(pts[0][0] * s, pts[0][1] * s, 0.3 * s, 0, Math.PI * 2)
        ctx.fill()
        continue
      }
      for (let i = 1; i < pts.length; i++) {
        ctx.lineWidth = (0.25 + 0.35 * (pts[i][3] || 0.5)) * s
        ctx.beginPath()
        ctx.moveTo(pts[i - 1][0] * s, pts[i - 1][1] * s)
        ctx.lineTo(pts[i][0] * s, pts[i][1] * s)
        ctx.stroke()
      }
    }
  }, [strokes, pxPerMm])

  useEffect(() => {
    redraw()
    window.addEventListener('resize', redraw)
    return () => window.removeEventListener('resize', redraw)
  }, [redraw])

  const toPoint = (e: PointerEvent, rect: DOMRect): Point => {
    const state = live.current
    if (state.t0 === null) state.t0 = e.timeStamp
    const pressure = e.pointerType === 'mouse' ? 0.5 : e.pressure
    return [
      +((e.clientX - rect.left) / pxPerMm).toFixed(3),
      +((e.clientY - rect.top) / pxPerMm).toFixed(3),
      +(e.timeStamp - state.t0).toFixed(1),
      +pressure.toFixed(3),
      e.tiltX || 0,
      e.tiltY || 0,
    ]
  }

  const accepts = (e: React.PointerEvent) => {
    const state = live.current
    if (e.pointerType === 'pen') state.penSeen = true
    return e.pointerType === 'pen' || (!state.penSeen && e.pointerType === 'mouse') || (!state.penSeen && e.pointerType === 'touch')
  }

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!accepts(e)) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    live.current.pointerType = e.pointerType
    live.current.stroke = [toPoint(e.nativeEvent, e.currentTarget.getBoundingClientRect())]
    redraw()
  }

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const state = live.current
    if (!state.stroke) return
    e.preventDefault()
    const rect = e.currentTarget.getBoundingClientRect()
    const native = e.nativeEvent
    const events = typeof native.getCoalescedEvents === 'function' ? native.getCoalescedEvents() : []
    for (const ev of events.length ? events : [native]) state.stroke.push(toPoint(ev, rect))
    redraw()
  }

  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const state = live.current
    if (!state.stroke) return
    e.preventDefault()
    const finished = state.stroke
    state.stroke = null
    setStrokes((prev) => [...prev, { points: finished }])
  }

  const session = useCallback(
    (all: Record<string, CaptureLine>): CaptureSession => ({
      schema: 'hw-capture/1',
      sessionId,
      createdAt: new Date().toISOString(),
      device: {
        userAgent: navigator.userAgent,
        pointerType: live.current.pointerType,
        pxPerMm,
        devicePixelRatio: window.devicePixelRatio || 1,
      },
      lines: prompts.filter((p) => all[p.id]).map((p) => all[p.id]),
    }),
    [sessionId, pxPerMm, prompts],
  )

  const push = async (all: Record<string, CaptureLine>, final: boolean) => {
    try {
      const res = await api.saveSession(session(all), final)
      setNote(final ? `Saved ${res.saved} lines.${res.retraining ? ' Retraining started.' : ''}` : `Autosaved ${res.saved} lines.`)
    } catch {
      setNote('Trainer not reachable; lines are kept in this tab. Use the export buttons before closing it.')
    }
  }

  const commit = (advance: number) => {
    let all = lines
    if (prompt && strokes.length) {
      const line: CaptureLine = {
        id: prompt.id,
        promptId: prompt.id,
        text: prompt.text,
        holdout: prompt.holdout,
        strokes,
        guide: { baselineMm: PAD_MM.baseline, xHeightMm: PAD_MM.xHeight },
      }
      all = { ...lines, [prompt.id]: line }
      setLines(all)
      void push(all, false)
    }
    setIndex((i) => Math.min(prompts.length - 1, Math.max(0, i + advance)))
  }

  const addCustom = () => {
    const text = custom.trim().replace(/\s+/g, ' ')
    if (!text || text.length >= 75) return
    const id = `C${Date.now().toString(36)}`
    setPrompts((p) => [...p, { id, text, holdout: customHoldout }])
    setIndex(prompts.length)
    setCustom('')
  }

  const exportFiles = useMemo(
    () => ({
      json: () => download(`${sessionId}.json`, new Blob([JSON.stringify(session(lines))], { type: 'application/json' })),
      xml: () =>
        download(
          `${sessionId}-iam-ondb.zip`,
          zip([
            ...session(lines).lines.map((l) => ({ name: `lineStrokes/${sessionId}/${l.id}.xml`, text: lineToIamXml(l) })),
            { name: `ascii/${sessionId}.txt`, text: session(lines).lines.map((l) => `${l.id} ${l.text}`).join('\n') + '\n' },
          ]),
        ),
    }),
    [lines, session, sessionId],
  )

  return (
    <div className="page">
      <div className="row between">
        <div>
          <span className="pill">
            Line {index + 1} of {prompts.length}
          </span>
          <span className="pill muted">{done} written</span>
          <span className="pill muted">session {sessionId}</span>
        </div>
        <div className="row">
          <button onClick={() => setCalibrating((c) => !c)}>Calibrate size</button>
          <button
            onClick={() => {
              setLines({})
              setIndex(0)
              setPrompts(script.lines)
              setSessionId(newSessionId())
              setNote('Started a new session.')
            }}
          >
            New session
          </button>
        </div>
      </div>

      {calibrating && (
        <div className="panel">
          <p>
            Hold a bank card against the screen and drag the slider until the blue bar is exactly as wide as the card (85.6 mm). This
            makes recorded coordinates true millimetres.
          </p>
          <div className="card-bar" style={{ width: CARD_MM * pxPerMm }} />
          <input
            type="range"
            min={2}
            max={12}
            step={0.005}
            value={pxPerMm}
            onChange={(e) => {
              setPxPerMm(+e.target.value)
              localStorage.setItem('hw.pxPerMm', e.target.value)
            }}
          />
          <span className="muted">{pxPerMm.toFixed(3)} px/mm</span>
        </div>
      )}

      <p className="prompt">{prompt?.text}</p>
      <canvas
        ref={canvasRef}
        className="pad"
        style={{ height: PAD_MM.height * pxPerMm }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      />
      <div className="row between">
        <div className="row">
          <button onClick={() => commit(-1)} disabled={index === 0}>
            Back
          </button>
          <button onClick={() => setStrokes((s) => s.slice(0, -1))} disabled={!strokes.length}>
            Undo stroke
          </button>
          <button onClick={() => setStrokes([])} disabled={!strokes.length}>
            Clear
          </button>
        </div>
        <div className="row">
          <button onClick={() => setIndex((i) => Math.min(prompts.length - 1, i + 1))}>Skip</button>
          <button className="primary" onClick={() => commit(1)} disabled={!strokes.length}>
            {lines[prompt?.id] ? 'Replace and next' : 'Save and next'}
          </button>
        </div>
      </div>

      <div className="panel">
        <div className="row">
          <input
            className="grow"
            placeholder="Add your own line (under 75 characters), e.g. symbols you use that the script lacks"
            value={custom}
            maxLength={74}
            onChange={(e) => setCustom(e.target.value)}
          />
          <label className="check">
            <input type="checkbox" checked={customHoldout} onChange={(e) => setCustomHoldout(e.target.checked)} /> hold out
          </label>
          <button onClick={addCustom} disabled={!custom.trim()}>
            Add line
          </button>
        </div>
        <div className="row">
          <button className="primary" onClick={() => void push(lines, true)} disabled={!done}>
            Finish and send to trainer
          </button>
          <button onClick={exportFiles.json} disabled={!done}>
            Export JSON
          </button>
          <button onClick={exportFiles.xml} disabled={!done}>
            Export IAM-OnDB XML
          </button>
          <span className="muted">{note}</span>
        </div>
      </div>
    </div>
  )
}
