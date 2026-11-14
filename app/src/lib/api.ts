import type { Path2 } from './ink'

export type Engine = 'glyphs' | 'model'
export type Sliders = { neatness: number; size: number | null; slant: number; spacing: number; jitter: number; real_words: boolean; joins: number }
export const defaultSliders: Sliders = { neatness: 0.6, size: null, slant: 0, spacing: 1, jitter: 0.3, real_words: false, joins: 1 }

export type Status = {
  lines: number
  train: number
  holdout: number
  photo: number
  new_lines: number
  retrain_every: number
  champion: { version: string; metrics: Record<string, number | null> } | null
  versions: string[]
  training: boolean
  log: string
  reports: string[]
  blind: { trials: number; correct: number; passed: boolean }
  glyphs: Record<string, number>
  real_words: number
  habits: Record<string, number | null>
}

export type PageResult = {
  page: { width: number; height: number }
  pages: Path2[][]
  unsupported: string[]
  checkpoint: string
  default_size: number
}

export type PenProfile = {
  offset_x: number
  offset_y: number
  holder_front: number
  z_draw: number
  z_lift: number
  draw_speed: number
  travel_speed: number
  z_speed: number
  accel: number
  dot_dwell_ms: number
  back_stop_y: number
  calibrated: number
}
export type PlotterInfo = { pen: PenProfile; reach: number[]; paper_left: number; summary: string; problems: string[] }
export type PlotPart = { name: string; url: string; turned: boolean; strokes: number; minutes: number }

export type Box = { x: number; y: number; w: number; h: number; kind: string; text?: string; size?: number }

export type PhotoWord = {
  id: string
  text: string
  strokes: Path2[]
  size: [number, number]
  crop: string
  status: string
  photo?: string
  line?: number
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

const post = <T,>(url: string, body: unknown) =>
  json<T>(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

const upload = <T,>(url: string, file: File) => {
  const form = new FormData()
  form.append('file', file)
  return json<T>(url, { method: 'POST', body: form })
}

export const api = {
  status: () => json<Status>('/api/status'),
  saveSession: (session: unknown, final: boolean) =>
    post<{ saved: number; retraining: boolean }>(`/api/sessions?final=${final ? 1 : 0}`, session),
  loadSession: (id: string) => json<unknown>(`/files/sessions/${id}.json`),
  train: () => post<{ started: boolean }>('/api/train', {}),
  generate: (text: string, sliders: Sliders, seed: number, labels: unknown[], engine: Engine) =>
    post<PageResult>('/api/generate', { text, sliders, seed, labels, engine }),
  plotter: () => json<PlotterInfo>('/api/plotter'),
  savePlotter: (pen: PenProfile) => post<PlotterInfo>('/api/plotter', pen),
  plot: (paths: Path2[], page: { width: number; height: number }, title: string, dry_run: boolean) =>
    post<{ parts: PlotPart[]; problems: string[] }>('/api/plot', { paths, page, title, dry_run }),
  importTemplate: (file: File) =>
    upload<{ results: { page: number | null; units?: number; variants?: number; empty?: string[] }[]; glyphs: Record<string, number> }>(
      '/api/template/import',
      file,
    ),
  saveGlyph: (char: string, variants: number[][][][], baseline: number) =>
    post<{ glyphs: Record<string, number> }>('/api/glyphs', { char, variants, baseline }),
  worksheet: (file: File) =>
    upload<{ id: string; image: string; page: { w: number; h: number }; boxes: Box[]; deskew: string }>('/api/worksheet', file),
  copyPage: (file: File, page: number) =>
    upload<{ id: string; image: string; page: { w: number; h: number }; pages: number; boxes: Box[]; deskew: string; diagram: Path2[] }>(
      `/api/copy?page=${page}`,
      file,
    ),
  copyDiagram: (id: string, boxes: Box[]) => post<{ diagram: Path2[] }>('/api/copy/diagram', { id, boxes }),
  fill: (boxes: Box[], sliders: Sliders, seed: number, engine: Engine) =>
    post<{ answers: { strokes: Path2[]; size: number }[] }>('/api/worksheet/fill', { boxes, sliders, seed, engine }),
  photo: (file: File) => upload<{ id: string }>('/api/photos', file),
  photoJobs: () => json<Record<string, { state: string; total: number; words?: number; error?: string }>>('/api/photos/jobs'),
  words: () => json<{ words: PhotoWord[]; counts: Record<string, number> }>('/api/photos/words?status=pending'),
  updateWord: (id: string, body: { status: string; text?: string; strokes?: Path2[] }) =>
    post<{ retraining: boolean }>(`/api/photos/words/${id}`, body),
  blind: () => json<{ text: string; options: { real: boolean; strokes: Path2[] }[]; checkpoint: string }>('/api/blind'),
  blindResult: (correct: boolean, text: string) => post<Status['blind']>('/api/blind', { correct, text }),
  blindReset: () => json<{ ok: boolean }>('/api/blind', { method: 'DELETE' }),
}
