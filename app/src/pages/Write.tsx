import { useCallback, useEffect, useRef, useState } from 'react'
import EnginePicker from '../EnginePicker'
import PlotExport from '../PlotExport'
import SliderPanel from '../SliderPanel'
import { api, defaultSliders } from '../lib/api'
import type { Engine, PageResult, Sliders } from '../lib/api'
import { download, drawPaths, pathsToSvg } from '../lib/ink'

type Label = { text: string; x: number; y: number; size: number | null }

const SAMPLE =
  'Dear Sam,\nThank you for the notes from last week. I read them twice on the train and they helped a lot.\nThe area is x^2 + \\frac{1}{2} of the old one.'

export default function Write() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [text, setText] = useState(SAMPLE)
  const [sliders, setSliders] = useState<Sliders>(defaultSliders)
  const [seed, setSeed] = useState(1)
  const [result, setResult] = useState<PageResult | null>(null)
  const [pageIndex, setPageIndex] = useState(0)
  const [labels, setLabels] = useState<Label[]>([])
  const [placing, setPlacing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [speed, setSpeed] = useState(600)
  const [engine, setEngine] = useState<Engine>('model')
  const [onePass, setOnePass] = useState(true)
  const anim = useRef({ drawn: 0, raf: 0, last: 0, playing: false })

  const paint = useCallback(
    (budget: number) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const page = result?.page ?? { width: 215.9, height: 279.4 }
      const dpr = window.devicePixelRatio || 1
      const cssWidth = canvas.clientWidth
      const scale = (cssWidth / page.width) * dpr
      canvas.width = Math.round(cssWidth * dpr)
      canvas.height = Math.round(page.height * scale)
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = '#fffdf8'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      if (result) drawPaths(ctx, result.pages[pageIndex] ?? [], scale, { budget })
      ctx.fillStyle = '#b3541e'
      ctx.font = `${12 * dpr}px system-ui`
      for (const l of labels) if (!result) ctx.fillText(l.text, l.x * scale, l.y * scale)
    },
    [result, pageIndex, labels],
  )

  const play = useCallback(() => {
    const state = anim.current
    cancelAnimationFrame(state.raf)
    const total = (result?.pages[pageIndex] ?? []).reduce((n, p) => n + p.length, 0)
    state.drawn = 0
    state.last = performance.now()
    state.playing = true
    const tick = (now: number) => {
      state.drawn += ((now - state.last) / 1000) * speed
      state.last = now
      paint(state.drawn)
      if (state.drawn < total && state.playing) state.raf = requestAnimationFrame(tick)
    }
    state.raf = requestAnimationFrame(tick)
  }, [result, pageIndex, paint, speed])

  useEffect(() => {
    if (result) play()
    else paint(0)
    return () => cancelAnimationFrame(anim.current.raf)
  }, [result, pageIndex, play, paint])

  const generate = async (nextSeed = seed) => {
    setBusy(true)
    setError('')
    try {
      const out = await api.generate(text, sliders, nextSeed, labels, engine, onePass)
      setPageIndex(0)
      setResult(out)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!placing) return
    const rect = e.currentTarget.getBoundingClientRect()
    const page = result?.page ?? { width: 215.9, height: 279.4 }
    const label = window.prompt('Label text')
    setPlacing(false)
    if (!label) return
    const mm = page.width / rect.width
    setLabels((l) => [...l, { text: label, x: +((e.clientX - rect.left) * mm).toFixed(1), y: +((e.clientY - rect.top) * mm).toFixed(1), size: null }])
  }

  return (
    <div className="page split">
      <div className="side">
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={10} spellCheck={false} />
        <p className="muted small">
          Layout markup: <code>x^2</code> or <code>x^{'{10}'}</code> for exponents, <code>a_{'{n}'}</code> for subscripts,{' '}
          <code>\frac{'{a}{b}'}</code> for fractions. These are placed by the layout layer, not the model.
        </p>
        <EnginePicker engine={engine} onChange={setEngine} />
        <SliderPanel sliders={sliders} onChange={setSliders} defaultSize={result?.default_size} />
        <label className="check" title="Keeps the writing on each sheet inside what the pencil can reach in one go, so no sheet has to be turned. Longer text runs on to more sheets.">
          <input type="checkbox" checked={onePass} onChange={(e) => setOnePass(e.target.checked)} /> Fit each sheet to one plotter pass (no turning)
        </label>
        <div className="row">
          <button className="primary" onClick={() => void generate()} disabled={busy}>
            {busy ? 'Writing…' : 'Write it'}
          </button>
          <button
            onClick={() => {
              const next = Math.floor(Math.random() * 1e9)
              setSeed(next)
              void generate(next)
            }}
            disabled={busy}
          >
            New variation
          </button>
          <button onClick={play} disabled={!result}>
            Replay
          </button>
        </div>
        <label className="slider">
          <span>Preview speed</span>
          <input type="range" min={100} max={6000} step={100} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
          <output>{speed} pts/s</output>
        </label>
        <div className="row">
          <button onClick={() => setPlacing(true)} className={placing ? 'primary' : ''}>
            {placing ? 'Click the page…' : 'Add diagram label'}
          </button>
          {labels.length > 0 && <button onClick={() => setLabels([])}>Clear {labels.length} labels</button>}
          <button
            disabled={!result}
            onClick={() =>
              result &&
              download(
                `page-${pageIndex + 1}.svg`,
                new Blob([pathsToSvg(result.pages[pageIndex], result.page.width, result.page.height)], { type: 'image/svg+xml' }),
              )
            }
          >
            Download SVG
          </button>
        </div>
        {result && (
          <p className="muted small">
            Written with: {result.checkpoint}
            {result.one_pass && ` · ${result.one_pass.lines} lines per sheet, ${result.pages.length} sheet${result.pages.length > 1 ? 's' : ''}`}
            {result.pages.length > 1 && (
              <>
                {' · '}
                {result.pages.map((_, i) => (
                  <button key={i} className={i === pageIndex ? 'primary tiny' : 'tiny'} onClick={() => setPageIndex(i)}>
                    {i + 1}
                  </button>
                ))}
              </>
            )}
          </p>
        )}
        {result && result.unsupported.length > 0 && (
          <p className="warn">
            Skipped because you have not written them yet: {result.unsupported.join(' ')}. Add them on the My letters tab.
            {engine === 'model' && ' For the model, write capture lines containing them and retrain.'}
          </p>
        )}
        {error && <p className="warn">{error}</p>}
        {result && <PlotExport pages={result.pages} current={pageIndex} page={result.page} title="pages" />}
      </div>
      <canvas ref={canvasRef} className={placing ? 'sheet placing' : 'sheet'} onClick={onCanvasClick} />
    </div>
  )
}
