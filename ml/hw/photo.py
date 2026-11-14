import json
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

SIZE = 224
WORKER = Path(__file__).resolve().parents[1] / "inksight_worker.py"


def bands(flags):
    out, start = [], None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def strip_rules(binary):
    h, w = binary.shape
    thin = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    for y0, y1 in bands((thin > 0).sum(axis=1) >= 0.15 * w):
        if y1 - y0 > 8:
            continue
        y0, y1 = max(0, y0 - 1), min(h - 1, y1 + 1)
        above = binary[max(0, y0 - 2)] > 0
        below = binary[min(h - 1, y1 + 2)] > 0
        binary[y0:y1 + 1, ~(above & below)] = 0
    tall = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
    for x0, x1 in bands((tall > 0).sum(axis=0) >= 0.6 * h):
        if x1 - x0 > 8:
            continue
        x0, x1 = max(0, x0 - 1), min(w - 1, x1 + 1)
        left = binary[:, max(0, x0 - 2)] > 0
        right = binary[:, min(w - 1, x1 + 2)] > 0
        binary[~(left & right), x0:x1 + 1] = 0
    return binary


def ink_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block = max(31, (min(h, w) // 12) | 1)
    block = min(block, 255)
    diff = cv2.medianBlur(gray, block).astype(np.int32) - gray.astype(np.int32)
    level = max(22.0, 0.42 * float(np.percentile(diff, 99.5)))
    binary = (diff > level).astype(np.uint8) * 255
    binary = strip_rules(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    flat = [i for i in range(1, count) if stats[i][2] >= 10 and stats[i][3] <= max(6, 0.08 * stats[i][2] + 4)]
    binary[np.isin(labels, flat)] = 0
    return binary


def segment_words(image):
    binary = ink_mask(image)
    h, w = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        return []
    areas = stats[1:, 4]
    solid = stats[1:][areas >= 0.08 * np.percentile(areas, 95)]
    if len(solid) == 0:
        return []
    letter = float(np.median(solid[:, 3]))
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if (bh > 2.5 * letter and bw < 0.35 * letter) or bw > 0.8 * w:
            binary[labels == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(letter * 0.16)), max(3, int(letter * 0.35))))
    merged = cv2.dilate(binary, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    boxes = []
    for x, y, bw, bh, area in stats[1:]:
        if bh < 0.6 * letter or bw < 0.12 * letter or bh > 5 * letter:
            continue
        pad = int(letter * 0.12)
        boxes.append([max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)])
    boxes.sort(key=lambda b: (round((b[1] + b[3]) / 2 / (letter * 2.5)), b[0]))
    return boxes


def find_rules(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block = min(255, max(31, (min(h, w) // 12) | 1))
    diff = cv2.medianBlur(gray, block).astype(np.float32) - gray.astype(np.float32)
    level = max(22.0, 0.42 * float(np.percentile(diff, 99.5)))
    faint = np.clip(diff, 0, level * 0.6)
    streak = cv2.blur(faint, (max(41, w // 20) | 1, 1))
    ridge = streak - cv2.blur(streak, (1, 15))
    mask = (ridge > max(1.2, 0.25 * float(np.percentile(ridge, 99)))).astype(np.uint8) * 255
    segments = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=w // 6, minLineLength=w // 4, maxLineGap=w // 10)
    if segments is None:
        return []
    found = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).astype(np.float64):
        if abs(x2 - x1) < 10:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.2:
            found.append((y1 + slope * (w / 2 - x1), slope, abs(x2 - x1)))
    found.sort()
    clusters = []
    for mid, slope, length in found:
        if clusters and mid - clusters[-1][-1][0] < 9:
            clusters[-1].append((mid, slope, length))
        else:
            clusters.append([(mid, slope, length)])
    rules = []
    for cluster in clusters:
        weights = np.array([c[2] for c in cluster], dtype=np.float64)
        rules.append((float(np.average([c[0] for c in cluster], weights=weights)),
                      float(np.average([c[1] for c in cluster], weights=weights))))
    gaps = np.diff([r[0] for r in rules])
    if len(gaps) < 3:
        return []
    pitch = float(np.median(gaps))
    rules = [rules[0]] + [r for prev, r in zip(rules, rules[1:]) if r[0] - prev[0] > 0.55 * pitch]
    filled = [rules[0]]
    for rule in rules[1:]:
        missing = int(round((rule[0] - filled[-1][0]) / pitch)) - 1
        start = filled[-1]
        for k in range(1, missing + 1):
            t = k / (missing + 1)
            filled.append((start[0] + t * (rule[0] - start[0]), start[1] + t * (rule[1] - start[1])))
        filled.append(rule)
    return [(mid - slope * w / 2, slope) for mid, slope in filled]


def group_by_rules(boxes, rules):
    arr = np.array(boxes, dtype=np.float64)
    cx = (arr[:, 0] + arr[:, 2]) / 2
    ref = arr[:, 1] + 0.62 * (arr[:, 3] - arr[:, 1])
    levels = np.array([[b + k * x for b, k in rules] for x in cx])
    owner = (ref[:, None] > levels).sum(axis=1)
    lines = [[int(i) for i in np.argsort(arr[:, 0]) if owner[i] == k] for k in range(len(rules) + 1)]
    return [line for line in lines if line]


def group_lines(boxes, image=None):
    if not boxes:
        return []
    rules = find_rules(image) if image is not None else []
    if len(rules) >= 4:
        return group_by_rules(boxes, rules)
    return group_by_peaks(boxes)


def group_by_peaks(boxes):
    arr = np.array(boxes, dtype=np.float64)
    cx, cy = (arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2
    height = float(np.median(arr[:, 3] - arr[:, 1]))
    weight = np.minimum(arr[:, 2] - arr[:, 0], 3 * height)
    step = max(2.0, height / 6)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)

    def profile(slope):
        y = cy - slope * cx
        bins = np.arange(y.min() - 3 * step, y.max() + 4 * step, step)
        hist, edges = np.histogram(y, bins=bins, weights=weight)
        return y, np.convolve(hist, kernel, mode="same"), edges

    slope = max(np.linspace(-0.12, 0.12, 49), key=lambda k: float((profile(k)[1] ** 2).sum()))
    y, hist, edges = profile(slope)
    centres = (edges[:-1] + edges[1:]) / 2
    reach = max(1, int(0.45 * height / step))
    peaks = [centres[i] for i in range(len(hist))
             if hist[i] > 0.08 * hist.max() and hist[i] == hist[max(0, i - reach):i + reach + 1].max()]
    if not peaks:
        return [sorted(range(len(boxes)), key=lambda i: boxes[i][0])]
    owner = np.abs(y[:, None] - np.array(peaks)[None, :]).argmin(axis=1)
    lines = [[i for i in np.argsort(arr[:, 0]) if owner[i] == k] for k in range(len(peaks))]
    return [[int(i) for i in line] for line in lines if line]


def pad_crop(image, box):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    ratio = min(SIZE / crop.shape[1], SIZE / crop.shape[0])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * ratio)), max(1, int(crop.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    corners = np.array([resized[0, 0], resized[0, -1], resized[-1, 0], resized[-1, -1]]).mean(axis=0)
    canvas = np.full((SIZE, SIZE, 3), corners, dtype=np.uint8)
    dx, dy = (SIZE - resized.shape[1]) // 2, (SIZE - resized.shape[0]) // 2
    canvas[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
    return canvas, ratio, dx, dy


def wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def worker_command(in_path, out_path):
    custom = os.environ.get("HW_INKSIGHT_PYTHON")
    if custom:
        return shlex.split(custom) + [str(WORKER), str(in_path), str(out_path)], lambda p: str(p)
    if platform.system() == "Windows":
        inner = f"~/inksight-venv/bin/python {shlex.quote(wsl_path(WORKER))} {shlex.quote(wsl_path(in_path))} {shlex.quote(wsl_path(out_path))}"
        return ["wsl", "-e", "bash", "-lc", inner], wsl_path
    return [str(Path.home() / "inksight-venv" / "bin" / "python"), str(WORKER), str(in_path), str(out_path)], lambda p: str(p)


def derender(image, boxes, crop_dir, prefix):
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=crop_dir) as tmp:
        in_path, out_path = Path(tmp) / "in.json", Path(tmp) / "out.json"
        command, convert = worker_command(in_path, out_path)
        jobs, frames = [], {}
        for i, box in enumerate(boxes):
            canvas, ratio, dx, dy = pad_crop(image, box)
            name = f"{prefix}_{i:04d}"
            cv2.imwrite(str(crop_dir / f"{name}.png"), image[box[1]:box[3], box[0]:box[2]])
            cv2.imwrite(str(Path(tmp) / f"{name}.jpg"), canvas)
            jobs.append({"id": name, "image": convert(Path(tmp) / f"{name}.jpg")})
            frames[name] = (ratio, dx, dy, box)
        in_path.write_text(json.dumps(jobs), encoding="utf-8")
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"InkSight worker failed: {done.stderr[-800:]}")
        results = json.loads(out_path.read_text(encoding="utf-8"))
    words = []
    for res in results:
        ratio, dx, dy, box = frames[res["id"]]
        strokes = [[[(x - dx) / ratio, (y - dy) / ratio] for x, y in s] for s in res["strokes"] if s]
        words.append({"id": res["id"], "text": res["text"], "strokes": strokes, "box": [int(v) for v in box],
                      "size": [int(box[2] - box[0]), int(box[3] - box[1])]})
    return words
