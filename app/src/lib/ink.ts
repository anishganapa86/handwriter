export type Point = [number, number, number, number, number, number]
export type Stroke = { points: Point[] }
export type CaptureLine = {
  id: string
  promptId: string
  text: string
  holdout: boolean
  strokes: Stroke[]
  guide: { baselineMm: number; xHeightMm: number }
}
export type CaptureSession = {
  schema: 'hw-capture/1'
  sessionId: string
  createdAt: string
  device: { userAgent: string; pointerType: string; pxPerMm: number; devicePixelRatio: number }
  lines: CaptureLine[]
}
export type Path2 = number[][]

const UNITS_PER_MM = 100

const esc = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;')

export function lineToIamXml(line: CaptureLine): string {
  const all = line.strokes.flatMap((s) => s.points)
  const q = (v: number) => Math.round(v * UNITS_PER_MM)
  const xs = all.map((p) => p[0])
  const ys = all.map((p) => p[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const strokes = line.strokes
    .filter((s) => s.points.length > 0)
    .map((s) => {
      const t0 = (s.points[0][2] / 1000).toFixed(3)
      const t1 = (s.points[s.points.length - 1][2] / 1000).toFixed(3)
      const pts = s.points.map((p) => `      <Point x="${q(p[0])}" y="${q(p[1])}" time="${(p[2] / 1000).toFixed(3)}"/>`).join('\n')
      return `    <Stroke colour="black" start_time="${t0}" end_time="${t1}">\n${pts}\n    </Stroke>`
    })
    .join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>
<WhiteboardCaptureSession>
  <WhiteboardDescription>
    <SensorLocation corner="top_left"/>
    <DiagonallyOppositeCoords x="${q(x1)}" y="${q(y1)}"/>
    <VerticallyOppositeCoords x="${q(x0)}" y="${q(y1)}"/>
    <HorizontallyOppositeCoords x="${q(x1)}" y="${q(y0)}"/>
  </WhiteboardDescription>
  <Transcription>
    <TextLine id="${esc(line.id)}" text="${esc(line.text)}"/>
  </Transcription>
  <StrokeSet>
${strokes}
  </StrokeSet>
</WhiteboardCaptureSession>
`
}

const crcTable = (() => {
  const table = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c >>> 0
  }
  return table
})()

function crc32(data: Uint8Array): number {
  let c = 0xffffffff
  for (let i = 0; i < data.length; i++) c = crcTable[(c ^ data[i]) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

export function zip(files: { name: string; text: string }[]): Blob {
  const enc = new TextEncoder()
  const parts: Uint8Array[] = []
  const central: Uint8Array[] = []
  let offset = 0
  for (const file of files) {
    const name = enc.encode(file.name)
    const data = enc.encode(file.text)
    const crc = crc32(data)
    const local = new DataView(new ArrayBuffer(30))
    local.setUint32(0, 0x04034b50, true)
    local.setUint16(4, 20, true)
    local.setUint16(6, 0x0800, true)
    local.setUint32(14, crc, true)
    local.setUint32(18, data.length, true)
    local.setUint32(22, data.length, true)
    local.setUint16(26, name.length, true)
    parts.push(new Uint8Array(local.buffer), name, data)
    const dir = new DataView(new ArrayBuffer(46))
    dir.setUint32(0, 0x02014b50, true)
    dir.setUint16(4, 20, true)
    dir.setUint16(6, 20, true)
    dir.setUint16(8, 0x0800, true)
    dir.setUint32(16, crc, true)
    dir.setUint32(20, data.length, true)
    dir.setUint32(24, data.length, true)
    dir.setUint16(28, name.length, true)
    dir.setUint32(42, offset, true)
    central.push(new Uint8Array(dir.buffer), name)
    offset += 30 + name.length + data.length
  }
  const size = central.reduce((n, c) => n + c.length, 0)
  const end = new DataView(new ArrayBuffer(22))
  end.setUint32(0, 0x06054b50, true)
  end.setUint16(8, files.length, true)
  end.setUint16(10, files.length, true)
  end.setUint32(12, size, true)
  end.setUint32(16, offset, true)
  return new Blob([...parts, ...central, new Uint8Array(end.buffer)] as BlobPart[], { type: 'application/zip' })
}

export function download(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

export function drawPaths(
  ctx: CanvasRenderingContext2D,
  paths: Path2[],
  scale: number,
  opts: { ox?: number; oy?: number; budget?: number; width?: number; colour?: string } = {},
) {
  const { ox = 0, oy = 0, budget = Infinity, width = 0.35, colour = '#14213d' } = opts
  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = width * scale
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  let left = budget
  for (const path of paths) {
    if (left <= 0) break
    const n = Math.min(path.length, Math.ceil(left))
    left -= path.length
    if (path.length === 1 || n === 1) {
      ctx.beginPath()
      ctx.arc(ox + path[0][0] * scale, oy + path[0][1] * scale, (width * scale) / 1.6, 0, Math.PI * 2)
      ctx.fill()
      continue
    }
    ctx.beginPath()
    ctx.moveTo(ox + path[0][0] * scale, oy + path[0][1] * scale)
    for (let i = 1; i < n; i++) ctx.lineTo(ox + path[i][0] * scale, oy + path[i][1] * scale)
    ctx.stroke()
  }
}

export function pathsToSvg(paths: Path2[], w: number, h: number): string {
  const body = paths
    .map((p) =>
      p.length === 1
        ? `<circle cx="${p[0][0]}" cy="${p[0][1]}" r="0.15"/>`
        : `<polyline points="${p.map((q) => `${q[0]},${q[1]}`).join(' ')}"/>`,
    )
    .join('\n')
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}mm" height="${h}mm" viewBox="0 0 ${w} ${h}" fill="none" stroke="#14213d" stroke-width="0.35" stroke-linecap="round" stroke-linejoin="round">\n${body}\n</svg>\n`
}
