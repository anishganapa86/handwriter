import { useEffect, useState } from 'react'
import Blind from './pages/Blind'
import Capture from './pages/Capture'
import CopyNotes from './pages/CopyNotes'
import Glyphs from './pages/Glyphs'
import Plotter from './pages/Plotter'
import Review from './pages/Review'
import Train from './pages/Train'
import Worksheet from './pages/Worksheet'
import Write from './pages/Write'

function CopyPage() {
  return <Worksheet mode="copy" />
}

// The everyday tools. Training tools sit behind one button so the bar stays short.
const main = [
  { id: 'write', label: 'Write', view: Write },
  { id: 'copy', label: 'From a photo', view: CopyPage },
  { id: 'worksheet', label: 'Worksheet', view: Worksheet },
  { id: 'notes', label: 'Copy my notes', view: CopyNotes },
  { id: 'plotter', label: 'Printer', view: Plotter },
]

const training = [
  { id: 'glyphs', label: 'My letters', view: Glyphs },
  { id: 'capture', label: 'Line capture', view: Capture },
  { id: 'review', label: 'Photo review', view: Review },
  { id: 'blind', label: 'Blind test', view: Blind },
  { id: 'train', label: 'Training', view: Train },
]

const all = [...main, ...training]
const current = () => all.find((t) => t.id === window.location.hash.slice(1)) ?? main[0]
const isTraining = (id: string) => training.some((t) => t.id === id)

export default function App() {
  const [tab, setTab] = useState(current)
  const [showTraining, setShowTraining] = useState(() => isTraining(current().id))
  useEffect(() => {
    const onHash = () => {
      const next = current()
      setTab(next)
      if (isTraining(next.id)) setShowTraining(true)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const View = tab.view
  return (
    <>
      <nav>
        <strong>Handwriter</strong>
        {main.map((t) => (
          <a key={t.id} href={`#${t.id}`} className={t.id === tab.id ? 'on' : ''}>
            {t.label}
          </a>
        ))}
        <button
          className={showTraining || isTraining(tab.id) ? 'navtoggle on' : 'navtoggle'}
          aria-expanded={showTraining}
          onClick={() => setShowTraining((v) => !v)}
        >
          Training {showTraining ? '▴' : '▾'}
        </button>
      </nav>
      {showTraining && (
        <nav className="subnav">
          {training.map((t) => (
            <a key={t.id} href={`#${t.id}`} className={t.id === tab.id ? 'on' : ''}>
              {t.label}
            </a>
          ))}
        </nav>
      )}
      <View />
    </>
  )
}
