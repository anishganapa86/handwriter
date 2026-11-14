from pathlib import Path

import cv2
import numpy as np

from .sheet import trace

PAGE_W, PAGE_H = 215.9, 279.4
PX = 12
MARK, HOLE = 8.0, 4.0
CORNERS = [(12.0, 12.0), (PAGE_W - 12.0, 12.0), (PAGE_W - 12.0, PAGE_H - 12.0), (12.0, PAGE_H - 12.0)]
ORIENT = (20.0, 10.0, 4.0)
BITS = [(27.0, 10.0, 4.0), (34.0, 10.0, 4.0)]
GRID_TOP, GRID_LEFT, GRID_RIGHT, GRID_BOTTOM = 22.0, 12.0, PAGE_W - 12.0, PAGE_H - 22.0
CELL_H, LABEL_H, BASELINE, XHEIGHT = 15.5, 3.2, 11.2, 3.0
JOINED = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of", "ed", "is", "it",
          "al", "ar", "st", "to", "nt", "ng", "ing", "tion", "the", "and"]
FONTS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/seguisym.ttf", "C:/Windows/Fonts/cambria.ttc",
         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def units():
    out = [(ch, 4) for ch in "abcdefghijklmnopqrstuvwxyz"]
    out += [(ch, 3) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    out += [(ch, 3) for ch in "0123456789"]
    out += [(ch, 2) for ch in ".,;:!?'\"()-/&@#%$"]
    out += [(ch, 2) for ch in "+=*<>[]{}|~_^×÷±"]
    out += [(ch, 1) for ch in "πθ√∞≤≥≠≈→∈∑∫Δαβλμσ°"]
    out += [(unit, 2) for unit in JOINED]
    return out


def cell_width(unit):
    return 14.0 if len(unit) == 1 else 20.0 if len(unit) == 2 else 28.0


def layout():
    pages, cells, x, y = [], [], GRID_LEFT, GRID_TOP
    for unit, count in units():
        width = cell_width(unit)
        rows_needed = 1 + int((x + count * width > GRID_RIGHT + 1e-6))
        if y + rows_needed * CELL_H > GRID_BOTTOM + 1e-6:
            pages.append(cells)
            cells, x, y = [], GRID_LEFT, GRID_TOP
        for _ in range(count):
            if x + width > GRID_RIGHT + 1e-6:
                x, y = GRID_LEFT, y + CELL_H
            cells.append({"unit": unit, "x": x, "y": y, "w": width, "h": CELL_H})
            x += width
    pages.append(cells)
    return pages


def make_pdf(path):
    import fitz
    pt = 72 / 25.4
    fonts = [(f"F{i}", p, fitz.Font(fontfile=p)) for i, p in enumerate(FONTS) if Path(p).exists()]
    doc = fitz.open()
    pages = layout()
    for number, cells in enumerate(pages, 1):
        page = doc.new_page(width=PAGE_W * pt, height=PAGE_H * pt)
        box = lambda x, y, w, h: fitz.Rect(x * pt, y * pt, (x + w) * pt, (y + h) * pt)
        for cx, cy in CORNERS:
            page.draw_rect(box(cx - MARK / 2, cy - MARK / 2, MARK, MARK), color=None, fill=(0, 0, 0))
            page.draw_rect(box(cx - HOLE / 2, cy - HOLE / 2, HOLE, HOLE), color=None, fill=(1, 1, 1))
        page.draw_rect(box(*ORIENT[:2], ORIENT[2], ORIENT[2]), color=None, fill=(0, 0, 0))
        for bit, (bx, by, size) in enumerate(BITS):
            if number >> bit & 1:
                page.draw_rect(box(bx, by, size, size), color=None, fill=(0, 0, 0))
        page.insert_text((44 * pt, 13.6 * pt), f"Letter sheet {number} of {len(pages)}: write each one the way you normally do, "
                         "sitting on the solid line. Stay inside the box.", fontsize=7.5, color=(0.35, 0.35, 0.35))
        light, label = (0.8, 0.8, 0.8), (0.45, 0.45, 0.45)
        for cell in cells:
            x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
            page.draw_rect(box(x, y, w, h), color=light, width=0.5)
            page.draw_line((x * pt, (y + BASELINE) * pt), ((x + w) * pt, (y + BASELINE) * pt), color=light, width=0.6)
            page.draw_line((x * pt, (y + BASELINE - XHEIGHT) * pt), ((x + w) * pt, (y + BASELINE - XHEIGHT) * pt),
                           color=(0.88, 0.88, 0.88), width=0.4, dashes="[2 2] 0")
            choice = next(((n, p) for n, p, f in fonts if all(f.has_glyph(ord(ch)) for ch in cell["unit"])), None)
            if choice:
                page.insert_text(((x + 0.8) * pt, (y + 2.6) * pt), cell["unit"], fontsize=7, color=label,
                                 fontname=choice[0], fontfile=choice[1])
            else:
                page.insert_text(((x + 0.8) * pt, (y + 2.6) * pt), f"U+{ord(cell['unit'][0]):04X}", fontsize=5, color=label)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return len(pages)


def find_marks(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51 | 1, 15)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    area_min = gray.size * 0.00012
    found = []
    for i, contour in enumerate(contours):
        child = hierarchy[0][i][2]
        if hierarchy[0][i][3] != -1 or child == -1:
            continue
        area = cv2.contourArea(contour)
        if area < area_min or area > gray.size * 0.02:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(contour)
        if min(w, h) < 1 or max(w, h) / min(w, h) > 1.6 or area / (w * h) < 0.78:
            continue
        holes, k = [], child
        while k != -1:
            holes.append(cv2.contourArea(contours[k]))
            k = hierarchy[0][k][0]
        ratio = max(holes) / area
        if 0.12 < ratio < 0.42:
            found.append((cx, cy, area))
    return found


def flatten(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marks = find_marks(gray)
    if len(marks) < 4:
        return None
    pts = np.array([[m[0], m[1]] for m in sorted(marks, key=lambda m: -m[2])[:8]], dtype=np.float32)
    total, diff = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    quad = np.array([pts[total.argmin()], pts[diff.argmax()], pts[total.argmax()], pts[diff.argmin()]], dtype=np.float32)
    if len({tuple(p) for p in quad}) < 4:
        return None
    target = np.array(CORNERS, dtype=np.float32) * PX
    size = (int(PAGE_W * PX), int(PAGE_H * PX))
    for turn in range(4):
        matrix = cv2.getPerspectiveTransform(np.roll(quad, turn, axis=0), target)
        flat = cv2.warpPerspective(image, matrix, size, borderValue=(255, 255, 255))
        if dark(flat, ORIENT):
            return flat
    return None


def dark(flat, spot):
    x, y, s = spot
    patch = cv2.cvtColor(flat[int((y + 1) * PX):int((y + s - 1) * PX), int((x + 1) * PX):int((x + s - 1) * PX)], cv2.COLOR_BGR2GRAY)
    paper = np.percentile(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), 80)
    return patch.size > 0 and patch.mean() < 0.55 * paper


def read_page(image):
    flat = flatten(image)
    if flat is None:
        return None
    number = sum(1 << bit for bit, spot in enumerate(BITS) if dark(flat, spot))
    pages = layout()
    if not 1 <= number <= len(pages):
        return None
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    diff = cv2.medianBlur(gray, 181).astype(np.int32) - gray.astype(np.int32)
    inked = diff[int(GRID_TOP * PX):int(GRID_BOTTOM * PX), int(GRID_LEFT * PX):int(GRID_RIGHT * PX)]
    level = max(45.0, 0.5 * float(np.percentile(inked, 99.7)))
    mask = (diff > level).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    found, empty = {}, []
    inset = 0.9
    for cell in pages[number - 1]:
        x0, y0 = int((cell["x"] + inset) * PX), int((cell["y"] + LABEL_H) * PX)
        x1, y1 = int((cell["x"] + cell["w"] - inset) * PX), int((cell["y"] + cell["h"] - inset) * PX)
        crop = mask[y0:y1, x0:x1].copy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(crop)
        for i in range(1, count):
            bx, by, bw, bh, area = stats[i]
            if area < 6 or (bw > 0.7 * crop.shape[1] and bh <= 5):
                crop[labels == i] = 0
        if int((crop > 0).sum()) < 25:
            empty.append(cell["unit"])
            continue
        strokes = trace(crop, int(0.7 * PX), int(0.8 * PX))
        if not strokes:
            empty.append(cell["unit"])
            continue
        left = min(s[:, 0].min() for s in strokes)
        base = (cell["y"] + BASELINE) * PX - y0
        variant = [np.stack([(s[:, 0] - left) / PX, (s[:, 1] - base) / PX], axis=1).round(3).tolist() for s in strokes]
        found.setdefault(cell["unit"], []).append(variant)
    return {"page": number, "glyphs": found, "empty": sorted(set(empty) - set(found))}
