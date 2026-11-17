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

const tabs = [
  { id: 'write', label: 'Write', view: Write },
  { id: 'glyphs', label: 'My letters', view: Glyphs },
  { id: 'capture', label: 'Line capture', view: Capture },
  { id: 'review', label: 'Photo review', view: Review },
  { id: 'worksheet', label: 'Worksheet', view: Worksheet },
  { id: 'copy', label: 'Copy a page', view: CopyPage },
  { id: 'notes', label: 'Copy my notes', view: CopyNotes },
  { id: 'plotter', label: 'Plotter', view: Plotter },
  { id: 'blind', label: 'Blind test', view: Blind },
  { id: 'train', label: 'Training', view: Train },
]

const current = () => tabs.find((t) => t.id === window.location.hash.slice(1)) ?? tabs[0]

export default function App() {
  const [tab, setTab] = useState(current)
  useEffect(() => {
    const onHash = () => setTab(current())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const View = tab.view
  return (
    <>
      <nav>
        <strong>Handwriter</strong>
        {tabs.map((t) => (
          <a key={t.id} href={`#${t.id}`} className={t.id === tab.id ? 'on' : ''}>
            {t.label}
          </a>
        ))}
      </nav>
      <View />
    </>
  )
}
