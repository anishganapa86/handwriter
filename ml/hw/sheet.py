import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from .glyphs import load_library

DESCENDERS = "gjpqyQ"
BOTTOM = ".!?&@#%$πθ√σαΔ∑:λ"
LOW = {",": 0.3, ";": 0.3, "β": 0.6, "μ": 0.45, "_": 0.2}
CENTRE = "+=*<>~×÷±-≤≥≠≈→∞∈"
TALL = "[]{}|()/∫"
TOP = "'\"^°"
PARTS = {".": 1, ",": 1, "'": 1, "°": 1, ":": 2, ";": 2, "!": 2, "?": 2, "\"": 2}
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def ink_mask(image, upscale, threshold):
    big = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    diff = cv2.medianBlur(gray, 101).astype(np.int32) - gray.astype(np.int32)
    mask = (diff > threshold).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def neighbours(pixel, pixels):
    y, x = pixel
    return [(y + dy, x + dx) for dy, dx in NEIGHBOURS if (y + dy, x + dx) in pixels]


def skeleton_paths(mask):
    pixels = {(int(y), int(x)) for y, x in zip(*np.nonzero(skeletonize(mask > 0)))}
    degree = {p: len(neighbours(p, pixels)) for p in pixels}
    nodes = {p for p in pixels if degree[p] != 2}
    paths, used = [], set()

    def walk(start, nxt):
        path, prev, cur = [start, nxt], start, nxt
        while cur not in nodes:
            step = [n for n in neighbours(cur, pixels) if n != prev and n not in path[-3:]]
            if not step:
                break
            prev, cur = cur, step[0]
            path.append(cur)
            if cur == start:
                break
        return path

    for node in nodes:
        for nxt in neighbours(node, pixels):
            if (node, nxt) in used:
                continue
            path = walk(node, nxt)
            used.add((node, nxt))
            used.add((path[-1], path[-2]))
            paths.append(path)
    covered = {p for path in paths for p in path}
    for p in pixels:
        if p not in covered and p not in nodes:
            ring = neighbours(p, pixels)
            if ring:
                path = walk(p, ring[0])
                paths.append(path)
                covered.update(path)
    lonely = [[p] for p in nodes if degree[p] == 0]
    return paths + lonely, nodes, degree


def prune_and_merge(paths, nodes, degree, spur):
    def is_spur(path):
        ends = [path[0], path[-1]]
        tips = [e for e in ends if degree.get(e, 0) == 1]
        hubs = [e for e in ends if degree.get(e, 0) >= 3]
        return len(path) < spur and len(tips) == 1 and len(hubs) == 1

    paths = [p for p in paths if not is_spur(p)]
    strokes = [np.array([(x, y) for y, x in p], dtype=np.float64) for p in paths]

    def direction(stroke, at_start):
        pts = stroke[:6] if at_start else stroke[::-1][:6]
        vec = pts[0] - pts[-1]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else np.zeros(2)

    merged = True
    while merged:
        merged = False
        best = None
        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                for si in (True, False):
                    for sj in (True, False):
                        a = strokes[i][0] if si else strokes[i][-1]
                        b = strokes[j][0] if sj else strokes[j][-1]
                        if len(strokes[i]) < 2 or len(strokes[j]) < 2 or np.linalg.norm(a - b) > 3.0:
                            continue
                        score = float(np.dot(direction(strokes[i], si), direction(strokes[j], sj)))
                        if score < -0.45 and (best is None or score < best[0]):
                            best = (score, i, j, si, sj)
        if best:
            _, i, j, si, sj = best
            first = strokes[i][::-1] if si else strokes[i]
            second = strokes[j] if sj else strokes[j][::-1]
            joined = np.concatenate([first, second])
            strokes = [s for k, s in enumerate(strokes) if k not in (i, j)] + [joined]
            merged = True
    return strokes


def smooth(stroke, window=5, step=2):
    if len(stroke) < 4:
        return stroke
    closed = np.linalg.norm(stroke[0] - stroke[-1]) < 2.5 and len(stroke) > 12
    pad = window // 2
    if closed:
        padded = np.concatenate([stroke[-pad - 1:-1], stroke, stroke[1:pad + 1]])
    else:
        padded = np.concatenate([np.repeat(stroke[:1], pad, axis=0), stroke, np.repeat(stroke[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], axis=1)
    keep = list(range(0, len(out), step))
    if keep[-1] != len(out) - 1:
        keep.append(len(out) - 1)
    out = out[keep]
    if closed:
        out = np.concatenate([out, out[:1]])
    return out


def trace(mask, dot_size, spur):
    strokes = []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w <= dot_size and h <= dot_size:
            strokes.append(np.array([centroids[i]], dtype=np.float64))
            continue
        part = (labels == i).astype(np.uint8)
        for stroke in prune_and_merge(*skeleton_paths(part), spur):
            if len(stroke) == 1:
                strokes.append(stroke)
                continue
            if stroke[0][1] > stroke[-1][1] + 4 or (abs(stroke[0][1] - stroke[-1][1]) <= 4 and stroke[0][0] > stroke[-1][0]):
                stroke = stroke[::-1]
            strokes.append(smooth(stroke))
    strokes.sort(key=lambda s: (len(s) == 1, s[:, 0].min()))
    return strokes


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    keep = np.ones(len(xs), dtype=bool)
    coef = np.array([0.0, float(np.median(ys))])
    for _ in range(3):
        if keep.sum() >= 3:
            coef = np.polyfit(xs[keep], ys[keep], 1)
        resid = np.abs(ys - np.polyval(coef, xs))
        keep = resid < max(10.0, 1.5 * np.median(resid[keep]))
    return coef


def finish(found, rows, mm, dot_px, spur_px):
    xheights = [g["box"][3] - g["box"][1] for g in found if g["char"] in "acemnorsuvwxz"]
    core = float(np.median(xheights)) if xheights else None
    plain = lambda unit: unit.isascii() and unit.isalnum()
    low = lambda unit: any(ch in DESCENDERS for ch in unit)
    baselines = {}
    for r in range(rows):
        flat = [g for g in found if g["row"] == r and plain(g["char"]) and not low(g["char"])]
        if len(flat) >= 3:
            baselines[r] = fit_line([(g["box"][0] + g["box"][2]) / 2 for g in flat], [g["box"][3] for g in flat])
        elif flat:
            baselines[r] = np.array([0.0, float(np.median([g["box"][3] for g in flat]))])
    if core is None:
        tall = [g["box"][3] - g["box"][1] for g in found if plain(g["char"])]
        core = 0.55 * float(np.median(tall)) if tall else 20.0
    out = {}
    for g in found:
        ch = g["char"]
        x0, top, x1, bottom = g["box"]
        if plain(ch) and g["row"] in baselines:
            base = float(np.polyval(baselines[g["row"]], (x0 + x1) / 2))
        elif ch in CENTRE:
            base = (top + bottom) / 2 + 0.55 * core
        elif ch in TALL:
            base = (top + bottom) / 2 + 0.75 * core
        elif ch in TOP:
            base = top + 2.2 * core
        elif ch in LOW:
            base = bottom - LOW[ch] * core
        else:
            base = bottom
        strokes = trace(g["mask"] * 255, dot_px, spur_px)
        if plain(ch) and not any(c in "ij" for c in ch):
            strokes = [s for s in strokes if len(s) > 1]
        if not strokes:
            continue
        variant = [np.stack([(s[:, 0] - x0) * mm, (s[:, 1] - base) * mm], axis=1).round(3).tolist() for s in strokes]
        out.setdefault(ch, []).append(variant)
    return out, core * mm


def import_sheet(image_path, spec):
    image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    mask = ink_mask(image, spec["upscale"], spec["threshold"])
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    found = []
    for row in spec["rows"]:
        y0, y1 = row["y"]
        comps = [i for i in range(1, count) if y0 <= centroids[i][1] < y1 and stats[i][3] > 5 and stats[i][4] >= row.get("min_area", 30)]
        for glyph in row["glyphs"]:
            chosen = []
            for i in comps:
                cx = centroids[i][0]
                cy = stats[i][1] + stats[i][3] / 2 - y0
                for sel in glyph["sel"]:
                    if sel[0] <= cx < sel[1] and (len(sel) < 4 or sel[2] <= cy < sel[3]):
                        chosen.append(i)
                        break
            if not chosen:
                print(f"nothing found for {glyph['char']}")
                continue
            part = np.isin(labels, chosen).astype(np.uint8)
            ys, xs = np.nonzero(part)
            found.append({"char": glyph["char"], "row": spec["rows"].index(row), "mask": part,
                          "box": (xs.min(), ys.min(), xs.max(), ys.max())})
    return finish(found, len(spec["rows"]), spec["mm_per_px"], spec["dot_px"], spec["spur_px"])


def ruled_mask(image, upscale):
    from .photo import find_rules
    big = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    diff = cv2.medianBlur(gray, 151).astype(np.int32) - gray.astype(np.int32)
    mask = (diff > max(30.0, 0.45 * float(np.percentile(diff, 99.5)))).astype(np.uint8) * 255
    rules = find_rules(big)
    h, w = mask.shape
    xs = np.arange(w)
    for b, k in rules:
        line = b + k * xs
        for off in range(-3, 4):
            yy = np.clip(np.round(line + off).astype(int), 0, h - 1)
            above = mask[np.clip(yy - 5 - off, 0, h - 1), xs] > 0
            below = mask[np.clip(yy + 5 - off, 0, h - 1), xs] > 0
            gone = ~(above & below)
            mask[yy[gone], xs[gone]] = 0
    mids = sorted(b + k * w / 2 for b, k in rules)
    pitch = float(np.median(np.diff(mids))) if len(mids) > 3 else None
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), pitch


def import_rows(image_path, spec, report=print):
    image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    mask, pitch = ruled_mask(image, spec["upscale"])
    mm = spec["rule_mm"] / pitch if pitch else spec["mm_per_px"]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    found, good = [], True
    for r, row in enumerate(spec["rows"]):
        lo, hi = row.get("x", [0, mask.shape[1]])
        comps = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            if not (lo <= cx < hi) or abs(cy - (row["yc"] + row.get("k", 0.0) * cx)) > row["half"] or area < row.get("min_area", 20):
                continue
            if row.get("drop_flat") and h <= 7 and w >= 14:
                continue
            if "base" in row and y > row["base"][0] + row["base"][1] * cx - row.get("low", 40) and h < row.get("low_h", 75) and area < 900:
                continue
            if any(a <= cx < b and c <= cy < d for a, b, c, d in row.get("erase", [])):
                continue
            comps.append(i)
        comps.sort(key=lambda i: stats[i][0])
        groups = []
        for i in comps:
            x0, x1, cx = stats[i][0], stats[i][0] + stats[i][2], centroids[i][0]
            cut = groups and any(groups[-1]["cx"] < c <= cx for c in row.get("cuts", []))
            if groups and not cut and x0 <= groups[-1]["x1"] + row.get("gap", 20):
                groups[-1]["ids"].append(i)
                groups[-1]["x1"] = max(groups[-1]["x1"], x1)
                groups[-1]["cx"] = max(groups[-1]["cx"], cx)
            else:
                groups.append({"ids": [i], "x0": x0, "x1": x1, "cx": cx})
        if len(groups) != len(row["seq"]):
            good = False
            report(f"row {r}: expected {len(row['seq'])} got {len(groups)}: " + " ".join(f"{g['x0']}-{g['x1']}" for g in groups))
            continue
        for unit, group in zip(row["seq"], groups):
            if not unit:
                continue
            ids = sorted(group["ids"], key=lambda i: -stats[i][4])[:PARTS.get(unit, len(group["ids"]))]
            part = np.isin(labels, ids).astype(np.uint8)
            ys, xs = np.nonzero(part)
            found.append({"char": unit, "row": r, "mask": part, "box": (xs.min(), ys.min(), xs.max(), ys.max())})
    glyphs, core = finish(found, len(spec["rows"]), mm, spec["dot_px"], spec["spur_px"])
    return glyphs, core, good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "glyphs.json"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if "seq" in spec["rows"][0]:
        glyphs, core, good = import_rows(spec_path.parent / spec["image"], spec)
        if not good:
            raise SystemExit("some rows did not split into the expected number of characters; nothing saved")
    else:
        glyphs, core = import_sheet(spec_path.parent / spec["image"], spec)
    library = load_library(args.out)
    for ch, variants in glyphs.items():
        library["glyphs"][ch] = (library["glyphs"].get(ch, []) if args.keep else []) + variants
    Path(args.out).write_text(json.dumps(library, ensure_ascii=False), encoding="utf-8")
    print(f"{len(glyphs)} characters imported, x-height {core:.2f} mm, library now has {len(library['glyphs'])}")


if __name__ == "__main__":
    main()
