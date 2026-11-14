import type { Sliders } from './lib/api'

type Props = { sliders: Sliders; onChange: (s: Sliders) => void; defaultSize?: number }

export default function SliderPanel({ sliders, onChange, defaultSize }: Props) {
  const size = sliders.size ?? defaultSize ?? 2.4
  const row = (label: string, value: number, min: number, max: number, step: number, set: (v: number) => void, unit: string) => (
    <label className="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => set(+e.target.value)} />
      <output>
        {value.toFixed(step < 0.1 ? 2 : 1)}
        {unit}
      </output>
    </label>
  )
  return (
    <div className="sliders">
      {row('Neatness', sliders.neatness, 0, 1, 0.05, (v) => onChange({ ...sliders, neatness: v }), '')}
      {row('Size (x-height)', size, 1.2, 5, 0.1, (v) => onChange({ ...sliders, size: v }), ' mm')}
      {row('Slant', sliders.slant, -20, 25, 1, (v) => onChange({ ...sliders, slant: v }), '°')}
      {row('Spacing', sliders.spacing, 0.6, 1.8, 0.05, (v) => onChange({ ...sliders, spacing: v }), '×')}
      {row('Baseline jitter', sliders.jitter, 0, 1, 0.05, (v) => onChange({ ...sliders, jitter: v }), '')}
      {row('Joined-up', sliders.joins, 0, 1.5, 0.05, (v) => onChange({ ...sliders, joins: v }), '×')}
      <label className="check">
        <input type="checkbox" checked={sliders.real_words} onChange={(e) => onChange({ ...sliders, real_words: e.target.checked })} />
        Experimental: my real joined hand from traced pieces (can misspell)
      </label>
    </div>
  )
}
