import xml.etree.ElementTree as ET

import numpy as np

XML_UNITS_PER_MM = 100.0


def clean(strokes):
    out = []
    for s in strokes:
        s = np.asarray(s, dtype=np.float64)
        if s.ndim != 2 or len(s) == 0:
            continue
        keep = np.ones(len(s), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(s[:, :2], axis=0)) > 1e-6, axis=1)
        out.append(s[keep])
    return out


def bounds(strokes):
    pts = np.concatenate([s[:, :2] for s in strokes])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def densify(strokes, step):
    out = []
    for s in strokes:
        if len(s) == 1:
            out.append(s[:, :2])
            continue
        seg = np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1)
        dist = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(dist[-1] / step) + 1)
        grid = np.linspace(0, dist[-1], n)
        out.append(np.stack([np.interp(grid, dist, s[:, 0]), np.interp(grid, dist, s[:, 1])], axis=1))
    return out


def core_metrics(strokes):
    x0, y0, x1, y1 = bounds(strokes)
    span = max(y1 - y0, 0.05 * (x1 - x0), 1e-3)
    pts = np.concatenate(densify(strokes, span / 60))
    slope = 0.0
    if x1 - x0 > 4 * span and len(pts) > 20:
        slope = float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0])
    ys = np.sort(pts[:, 1] - slope * (pts[:, 0] - x0))
    n = len(ys)
    k = max(2, int(0.62 * n))
    widths = ys[k - 1:] - ys[:n - k + 1]
    i = int(widths.argmin())
    return {"baseline": float(ys[i + k - 1]), "xline": float(ys[i]), "core": float(max(widths[i], 1e-6)), "slope": slope}


def resample_time(strokes, dt):
    out = []
    for s in strokes:
        if len(s) < 2 or s.shape[1] < 3:
            out.append(s[:, :2])
            continue
        t = s[:, 2]
        t = np.maximum.accumulate(t)
        n = max(2, int(round((t[-1] - t[0]) / dt)) + 1)
        grid = np.linspace(t[0], t[-1], n)
        out.append(np.stack([np.interp(grid, t, s[:, 0]), np.interp(grid, t, s[:, 1])], axis=1))
    return out


def kinematic(strokes, step):
    out = []
    for s in strokes:
        s = s[:, :2]
        if len(s) < 3:
            out.append(s)
            continue
        for _ in range(2):
            q = 0.75 * s[:-1] + 0.25 * s[1:]
            r = 0.25 * s[:-1] + 0.75 * s[1:]
            s = np.concatenate([s[:1], np.stack([q, r], axis=1).reshape(-1, 2), s[-1:]])
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        dist = np.concatenate([[0], np.cumsum(seg)])
        if dist[-1] < step:
            out.append(s[[0, -1]])
            continue
        fine = np.linspace(0, dist[-1], max(8, int(dist[-1] / (step / 6))))
        pts = np.stack([np.interp(fine, dist, s[:, 0]), np.interp(fine, dist, s[:, 1])], axis=1)
        d1 = np.gradient(pts, fine, axis=0)
        d2 = np.gradient(d1, fine, axis=0)
        curvature = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
        window = np.ones(7) / 7
        curvature = np.convolve(np.pad(curvature, 3, mode="edge"), window, mode="valid")
        speed = np.clip((curvature * step + 0.02) ** (-1.0 / 3.0), 0.45, 2.2)
        ramp = np.minimum(np.minimum(fine, dist[-1] - fine) / (1.5 * step) + 0.45, 1.0)
        speed = speed * ramp
        speed = speed * np.mean(1.0 / speed)
        time = np.concatenate([[0], np.cumsum(np.diff(fine) / ((speed[:-1] + speed[1:]) / 2))])
        n = max(2, int(round(time[-1] / step)) + 1)
        grid = np.linspace(0, time[-1], n)
        out.append(np.stack([np.interp(grid, time, pts[:, 0]), np.interp(grid, time, pts[:, 1])], axis=1))
    return out


def to_offsets(strokes):
    rows = []
    for s in strokes:
        for i, p in enumerate(s):
            rows.append((p[0], p[1], 1.0 if i == len(s) - 1 else 0.0))
    arr = np.array(rows, dtype=np.float32)
    out = arr.copy()
    out[1:, :2] = arr[1:, :2] - arr[:-1, :2]
    out[0, :2] = 0
    return out


def from_offsets(offsets, chars=None):
    offsets = np.asarray(offsets, dtype=np.float64)
    xy = np.cumsum(offsets[:, :2], axis=0)
    strokes, labels, start = [], [], 0
    for i in range(len(offsets)):
        if offsets[i, 2] > 0.5 or i == len(offsets) - 1:
            strokes.append(xy[start:i + 1])
            if chars is not None:
                labels.append(np.asarray(chars[start:i + 1]))
            start = i + 1
    return (strokes, labels) if chars is not None else strokes


def step_median(strokes):
    seg = [np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1) for s in strokes if len(s) > 1]
    return float(np.median(np.concatenate(seg))) if seg else 0.0


def capture_line_to_strokes(line):
    out = []
    for stroke in line["strokes"]:
        pts = np.asarray(stroke["points"], dtype=np.float64)
        if len(pts) == 0:
            continue
        cols = pts[:, :3].copy()
        cols[:, 2] = cols[:, 2] / 1000.0
        out.append(cols)
    return out


def write_iam_xml(strokes, text, path, line_id="line"):
    root = ET.Element("WhiteboardCaptureSession")
    desc = ET.SubElement(root, "WhiteboardDescription")
    x0, y0, x1, y1 = bounds(strokes)
    q = lambda v: str(int(round(v * XML_UNITS_PER_MM)))
    ET.SubElement(desc, "SensorLocation", corner="top_left")
    ET.SubElement(desc, "DiagonallyOppositeCoords", x=q(x1), y=q(y1))
    ET.SubElement(desc, "VerticallyOppositeCoords", x=q(x0), y=q(y1))
    ET.SubElement(desc, "HorizontallyOppositeCoords", x=q(x1), y=q(y0))
    trans = ET.SubElement(root, "Transcription")
    ET.SubElement(trans, "TextLine", id=line_id, text=text)
    sset = ET.SubElement(root, "StrokeSet")
    for s in strokes:
        has_t = s.shape[1] > 2
        t0 = float(s[0, 2]) if has_t else 0.0
        t1 = float(s[-1, 2]) if has_t else 0.0
        el = ET.SubElement(sset, "Stroke", colour="black", start_time=f"{t0:.3f}", end_time=f"{t1:.3f}")
        for p in s:
            ET.SubElement(el, "Point", x=q(p[0]), y=q(p[1]), time=f"{(p[2] if has_t else 0.0):.3f}")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_iam_xml(path):
    root = ET.parse(path).getroot()
    strokes = []
    for el in root.iter("Stroke"):
        pts = [(float(p.get("x")) / XML_UNITS_PER_MM, float(p.get("y")) / XML_UNITS_PER_MM, float(p.get("time", 0)))
               for p in el.iter("Point")]
        strokes.append(np.array(pts))
    line = root.find("./Transcription/TextLine")
    return strokes, (line.get("text") if line is not None else None)
