import { useEffect, useState } from 'react'
import { api } from './lib/api'
import type { Engine } from './lib/api'

type Props = { engine: Engine; onChange: (e: Engine) => void }

export default function EnginePicker({ engine, onChange }: Props) {
  const [count, setCount] = useState(0)
  const [model, setModel] = useState('base')
  const [real, setReal] = useState(0)

  useEffect(() => {
    api
      .status()
      .then((s) => {
        const n = Object.keys(s.glyphs).length
        setCount(n)
        setReal(s.real_words)
        setModel(s.champion?.version ?? 'base')
        if (n >= 26) onChange('glyphs')
      })
      .catch(() => undefined)
  }, [onChange])

  return (
    <div className="engine">
      <button className={engine === 'glyphs' ? 'primary' : ''} disabled={!count} onClick={() => onChange('glyphs')}>
        My letters ({count}){real > 0 && ` + ${real} real words`}
      </button>
      <button className={engine === 'model' ? 'primary' : ''} onClick={() => onChange('model')}>
        Model ({model === 'base' ? 'not yet trained on you' : model})
      </button>
    </div>
  )
}
