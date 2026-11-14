import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { PhotoWord } from '../lib/api'

type Jobs = Awaited<ReturnType<typeof api.photoJobs>>

function WordCard({ word, onDone }: { word: PhotoWord; onDone: () => void }) {
  const [text, setText] = useState(word.text)
  const [strokes, setStrokes] = useState(word.strokes)
  const [w, h] = word.size
  const edited = text !== word.text || strokes.length !== word.strokes.length

  const send = async (status: string) => {
    await api.updateWord(word.id, { status, text: text.trim(), strokes })
    onDone()
  }

  return (
    <div className="word">
      <div className="word-views">
        <img src={word.crop} alt="" style={{ aspectRatio: `${w} / ${h}` }} />
        <svg viewBox={`0 0 ${w} ${h}`}>
          {strokes.map((s, i) => (
            <polyline
              key={i}
              points={s.map((p) => p.join(',')).join(' ')}
              onClick={() => setStrokes((all) => all.filter((_, k) => k !== i))}
            />
          ))}
        </svg>
      </div>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && text.trim()) void send('accepted')
          if (e.key === 'Escape') void send('rejected')
        }}
      />
      <div className="row">
        <button className="primary" disabled={!text.trim() || !strokes.length} onClick={() => void send('accepted')}>
          {edited ? 'Accept fixed' : 'Accept'}
        </button>
        <button onClick={() => void send('rejected')}>Reject</button>
        {strokes.length !== word.strokes.length && <button onClick={() => setStrokes(word.strokes)}>Restore strokes</button>}
      </div>
    </div>
  )
}

export default function Review() {
  const [words, setWords] = useState<PhotoWord[]>([])
  const [counts, setCounts] = useState<Record<string, number>>({})
  const [jobs, setJobs] = useState<Jobs>({})
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const [list, running] = await Promise.all([api.words(), api.photoJobs()])
      setWords(list.words)
      setCounts(list.counts)
      setJobs(running)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), 4000)
    return () => clearInterval(timer)
  }, [refresh])

  const lines = useMemo(() => {
    const map = new Map<string, PhotoWord[]>()
    for (const w of words) {
      const key = `${w.photo ?? 'photo'}:${(w.line ?? 0) + 1}`
      map.set(key, [...(map.get(key) ?? []), w])
    }
    return [...map.entries()]
  }, [words])

  const acceptLine = async (group: PhotoWord[]) => {
    for (const w of group) if (w.text.trim() && w.strokes.length) await api.updateWord(w.id, { status: 'accepted', text: w.text.trim(), strokes: w.strokes })
    void refresh()
  }

  const onFiles = async (files: FileList | null) => {
    for (const file of Array.from(files ?? [])) await api.photo(file).catch((e) => setError(String(e)))
    void refresh()
  }

  return (
    <div className="page">
      <div className="panel">
        <div className="row">
          <label className="file">
            Add photos of old notes
            <input type="file" accept="image/*" multiple onChange={(e) => void onFiles(e.target.files)} />
          </label>
          <span className="muted">
            {counts.pending ?? 0} to review · {counts.accepted ?? 0} accepted · {counts.rejected ?? 0} rejected
          </span>
        </div>
        {Object.entries(jobs).map(([id, job]) => (
          <p key={id} className={job.state === 'error' ? 'warn' : 'muted small'}>
            {id}: {job.state === 'running' ? `converting ${job.total || ''} words to ink…` : job.state === 'done' ? `${job.words} words ready` : job.error}
          </p>
        ))}
        <p className="muted small">
          Each word was traced by InkSight. Fix the text under a word if it is wrong, click any stray stroke to delete it, and reject
          words that are scribbles or badly traced. Then accept word by word (Enter accepts, Escape rejects) or accept the rest of a
          line at once. Every accepted word is used as-is when that exact word appears in your text, and is also cut into letter pieces that build words you never wrote, so a wrong transcript here produces wrong letters later.
        </p>
        {error && <p className="warn">{error}</p>}
      </div>
      {lines.map(([key, group]) => (
        <section key={key} className="line-group">
          <div className="row between">
            <span className="muted small">
              {key.replace(':', ', line ')}: <b>{group.map((w) => w.text).join(' ')}</b>
            </span>
            <button onClick={() => void acceptLine(group)}>Accept this line as shown</button>
          </div>
          <div className="words">
            {group.map((w) => (
              <WordCard key={w.id} word={w} onDone={() => setWords((all) => all.filter((x) => x.id !== w.id))} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
