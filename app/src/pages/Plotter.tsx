import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { PenProfile, PlotterInfo } from '../lib/api'

const FIELDS: { key: keyof PenProfile; label: string; hint: string; step: number }[] = [
  { key: 'offset_x', label: 'Pen offset X (mm)', hint: 'Pen tip position minus nozzle position, left to right. Measured by the calibration file.', step: 0.1 },
  { key: 'offset_y', label: 'Pen offset Y (mm)', hint: 'Pen tip minus nozzle, front to back. Negative when the pen sits in front of the nozzle.', step: 0.1 },
  { key: 'holder_front', label: 'Holder reach in front of nozzle (mm)', hint: 'How far the holder sticks out towards the door. The head stays this far plus 5 mm from the front.', step: 1 },
  { key: 'z_draw', label: 'Draw height (mm)', hint: 'Nozzle height while writing. The pen tip is set 6 mm below the nozzle, so 5 presses a sprung pen 1 mm.', step: 0.1 },
  { key: 'z_lift', label: 'Lift height (mm)', hint: 'Nozzle height while moving between strokes. At least 7.', step: 0.1 },
  { key: 'draw_speed', label: 'Writing speed (mm/min)', hint: 'Slower is cleaner. 1500 is a safe start.', step: 100 },
  { key: 'travel_speed', label: 'Travel speed (mm/min)', hint: 'Speed between strokes with the pen up.', step: 100 },
  { key: 'z_speed', label: 'Lift speed (mm/min)', hint: 'How fast the bed moves for pen up and down.', step: 50 },
  { key: 'accel', label: 'Acceleration (mm/s²)', hint: 'Lower values shake the pen less.', step: 100 },
  { key: 'dot_dwell_ms', label: 'Dot dwell (ms)', hint: 'How long the pen rests for a full stop or the dot of an i.', step: 10 },
  { key: 'back_stop_y', label: 'Paper back edge at Y (mm)', hint: 'Where the top edge of the paper sits, measured from the front of the plate. 254 is 2 mm in from the back edge.', step: 1 },
]

export default function Plotter() {
  const [info, setInfo] = useState<PlotterInfo | null>(null)
  const [pen, setPen] = useState<PenProfile | null>(null)
  const [note, setNote] = useState('')

  useEffect(() => {
    api
      .plotter()
      .then((out) => {
        setInfo(out)
        setPen(out.pen)
      })
      .catch((e) => setNote(String(e)))
  }, [])

  const save = async () => {
    if (!pen) return
    try {
      const out = await api.savePlotter(pen)
      setInfo(out)
      setPen(out.pen)
      setNote('Saved.')
    } catch (e) {
      setNote(String(e))
    }
  }

  if (!pen || !info) return <div className="page">{note ? <p className="warn">{note}</p> : <p className="muted">Loading…</p>}</div>

  return (
    <div className="page">
      <div className="panel">
        <p>
          <b>Bambu Lab P1S as a pen plotter.</b> The app makes a <code>.gcode.3mf</code> file that you open in Bambu Studio and print.
          Nothing is heated and nothing is extruded. The hotend stays on; the pen sits in a holder beside it.
        </p>
        <ol className="steps">
          <li>Fit a sprung or free-sliding pen holder to the toolhead, with no pen in it. Remove any filament from the nozzle tip.</li>
          <li>In Bambu Studio open the file, press Print, and untick bed levelling, flow dynamics calibration and timelapse.</li>
          <li>The printer homes, then brings the head to the front and pauses.</li>
          <li>Tape the paper to the plate with its top edge against your back stop. Stand a 10 mm block under the holder, drop the pen in so its tip rests on the block, tighten it, and take the block away.</li>
          <li>Press Resume. Stay by the printer for the first runs and press Stop if anything looks wrong.</li>
        </ol>
        <p className="warn">
          Do every new setup as a dry run first: the pen then writes 5 mm above the paper so you can watch a whole job safely.
        </p>
      </div>
      <div className="panel">
        <p>
          <b>Reach with these settings.</b> {info.summary}
        </p>
        {info.problems.map((p) => (
          <p key={p} className="warn">
            {p}
          </p>
        ))}
        <div className="fields">
          {FIELDS.map((f) => (
            <label key={f.key} title={f.hint}>
              <span>{f.label}</span>
              <input type="number" step={f.step} value={pen[f.key]} onChange={(e) => setPen({ ...pen, [f.key]: +e.target.value })} />
              <small className="muted">{f.hint}</small>
            </label>
          ))}
        </div>
        <div className="row">
          <button className="primary" onClick={() => void save()}>
            Save settings
          </button>
          <a className="file" href="/api/plot/calibration" download="pen-calibration.gcode.3mf">
            Download calibration file
          </a>
          <span className="muted">{note}</span>
        </div>
        <p className="muted small">
          Calibration: the file draws a cross with a circle where the pen thinks the middle of the plate is, then parks the nozzle over
          the true middle and pauses. Measure from the nozzle tip to the centre of the cross, left-right and front-back, add those
          amounts to the two offsets above, save, and run it again until the cross sits under the nozzle.
        </p>
      </div>
    </div>
  )
}
