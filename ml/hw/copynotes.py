"""Copy the handwriting from a filled-in scan of a handout onto a blank copy of the same handout.

The blank is the page that goes on the plotter (a PDF page or a photo); the filled scan is a photo
or PDF of the same handout with notes written on it. The filled scan is lined up with the blank
(one perspective fit over the whole page, then a smooth local correction because two printings or a
curled photo never match exactly), the printed content is taken away, highlighter and colour are
ignored, and what is left - the pen and pencil ink - is traced as centre lines in millimetres on the
blank page, ready for the plotter. No handwriting recognition is involved: the strokes are copied as
they were written.
"""
import cv2
import numpy as np

from . import worksheet
from .sheet import trace

OUT_W = worksheet.OUT_W          # working width in pixels, the same scale the tracer was tuned for
PDF_DPI = 170
LETTER_W_MM = worksheet.PAGE_W_MM


def load(data, name="", page=0):
    """Image bytes or a PDF page -> (BGR image, page width in mm, number of pages)."""
    if name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        page = max(0, min(page, len(doc) - 1))
        pix = doc[page].get_pixmap(dpi=PDF_DPI, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), pix.width * 25.4 / PDF_DPI, len(doc)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image, None, 1


def blank_page(image, width_mm, is_pdf):
    """The page the notes go onto, at OUT_W pixels wide. A photo is flattened and taken as letter width."""
    if not is_pdf:
        image, _ = worksheet.deskew(image)
        width_mm = LETTER_W_MM
    scale = OUT_W / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC), width_mm


def flat_gray(image):
    """Grayscale with the paper levelled to white, so shadows and uneven light do not read as ink."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(31, (min(gray.shape) // 15) | 1)
    paper = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    paper = cv2.medianBlur(paper.astype(np.uint8), min(k, 255)).astype(np.float32)
    return np.clip(gray / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)


def align(blank, filled):
    """Perspective fit of the filled scan onto the blank from matched printed features."""
    g1 = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    scale = blank.shape[1] / filled.shape[1]
    small = cv2.resize(filled, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    g2 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None, len(good)
    src = np.float32([k2[m.queryIdx].pt for m in good]) / scale
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H, int(inliers.sum()) if inliers is not None else 0


def local_shifts(blank_ink, filled_ink, px_per_mm, tile_mm=50.0, max_mm=8.0):
    """Per-tile shift that lines the filled page's print up with the blank's, smoothed into a field."""
    h, w = blank_ink.shape
    tile = int(tile_mm * px_per_mm)
    step = tile // 2
    ys = list(range(0, max(1, h - tile) + 1, step))
    xs = list(range(0, max(1, w - tile) + 1, step))
    win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    grid = np.full((len(ys), len(xs), 2), np.nan, np.float32)
    a_all = cv2.GaussianBlur(blank_ink.astype(np.float32), (0, 0), 1.5)
    b_all = cv2.GaussianBlur(filled_ink.astype(np.float32), (0, 0), 1.5)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            a = a_all[y:y + tile, x:x + tile]
            b = b_all[y:y + tile, x:x + tile]
            if a.shape != (tile, tile) or a.mean() < 2 or b.mean() < 2:
                continue
            (dx, dy), response = cv2.phaseCorrelate(b, a, win)
            if response > 0.08 and np.hypot(dx, dy) < max_mm * px_per_mm:
                grid[i, j] = (dx, dy)
    valid = ~np.isnan(grid[..., 0])
    if not valid.any():
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), 0
    filled_grid = grid.copy()
    vi, vj = np.nonzero(valid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not valid[i, j]:
                nearest = np.argmin((vi - i) ** 2 + (vj - j) ** 2)
                filled_grid[i, j] = grid[vi[nearest], vj[nearest]]
    for c in range(2):
        filled_grid[..., c] = cv2.medianBlur(filled_grid[..., c].copy(), 3) if min(grid.shape[:2]) >= 3 else filled_grid[..., c]
    # tile centres -> full-resolution field
    dx = cv2.resize(filled_grid[..., 0], (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(filled_grid[..., 1], (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_LINEAR)
    pad = lambda m: cv2.copyMakeBorder(m, step // 2, max(0, h - m.shape[0] - step // 2), step // 2,
                                       max(0, w - m.shape[1] - step // 2), cv2.BORDER_REPLICATE)[:h, :w]
    return pad(dx), pad(dy), int(valid.sum())


def ink(gray, level):
    return ((255 - gray.astype(np.int16)) > level).astype(np.uint8) * 255


def picture_areas(blank, px_per_mm):
    """Parts of the blank that are photos or detailed figures: tinted and textured in a few-millimetre
    window. Flat colour fills (shaded answer areas) are not counted. Differences between two printings
    of a figure show up in these areas."""
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY).astype(np.float32)
    size = 2 * int(3 * px_per_mm) + 1
    tint = cv2.blur(((255 - gray) > 20).astype(np.float32), (size, size))
    mean = cv2.blur(gray, (size, size))
    texture = np.sqrt(np.maximum(cv2.blur(gray * gray, (size, size)) - mean * mean, 0))
    area = ((tint > 0.45) & (texture > 14)).astype(np.uint8) * 255
    area = cv2.morphologyEx(area, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    return cv2.dilate(area, np.ones((size, size), np.uint8))


def paper_area(image):
    """Where the sheet itself is in a photo (bright paper against a darker background)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    h, w = gray.shape
    middle = float(np.median(gray[h // 4:3 * h // 4, w // 4:3 * w // 4]))   # paper brightness in the open
    _, mask = cv2.threshold(gray, 0.5 * middle, 255, cv2.THRESH_BINARY)     # shaded paper stays above half
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.full(gray.shape, 255, np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, 4]))
    sheet = (labels == biggest).astype(np.uint8) * 255
    if stats[biggest, 4] < 0.3 * gray.size:
        return np.full(gray.shape, 255, np.uint8)
    contours, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(sheet)
    cv2.drawContours(hull, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 255, -1)
    return hull


def handwriting(blank_flat, filled_flat, valid, px_per_mm, blank_edges=None, pictures=None, level=95, print_level=60):
    """Dark ink on the filled page that the blank does not have. Light marks (highlighter, pale colour)
    stay below the level. Shapes that are thick all through (hole punches, blacked-out areas) are dropped,
    since no pen line is that wide. A mark lying over fine printed detail on the blank is a printing
    difference (a figure or arrow printed differently on the two copies), not notes: handwriting sits on
    white paper or flat colour and has almost no printed edges under it."""
    printed = ink(blank_flat, print_level)
    printed = cv2.dilate(printed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.7 * px_per_mm) + 1,) * 2))
    hand = ink(filled_flat, level)
    hand[printed > 0] = 0
    hand[valid == 0] = 0
    hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hand)
    thick = cv2.distanceTransform(hand, cv2.DIST_L2, 3)
    H, W = hand.shape
    pad = int(1.5 * px_per_mm)
    ring = np.ones((2 * int(px_per_mm) + 1,) * 2, np.uint8)
    keep = np.zeros(count, bool)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 0.6 * px_per_mm ** 2:                                   # specks
            continue
        own = labels[y:y + h, x:x + w] == i
        if (thick[y:y + h, x:x + w][own] > 0.6 * px_per_mm).mean() > 0.4:  # solid blobs, not lines
            continue
        if blank_edges is not None:
            y0, x0, y1, x1 = max(0, y - pad), max(0, x - pad), min(H, y + h + pad), min(W, x + w + pad)
            near = cv2.dilate((labels[y0:y1, x0:x1] == i).astype(np.uint8), ring)
            detail = (blank_edges[y0:y1, x0:x1][near > 0] > 0).sum() / max(1, int(area))
            in_picture = pictures is not None and (pictures[y:y + h, x:x + w][own] > 0).mean() > 0.5
            if detail > 0.35 or (in_picture and detail > 0.18):
                continue
        keep[i] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def copy_notes(blank_bytes, filled_bytes, blank_name="", filled_name="", blank_page_no=0, filled_page_no=0):
    """Returns the strokes (mm on the blank page), the page size, and images for checking the result."""
    raw, width_mm, pages = load(blank_bytes, blank_name, blank_page_no)
    if raw is None:
        raise ValueError("The blank handout is not an image or PDF.")
    blank, width_mm = blank_page(raw, width_mm, width_mm is not None)
    filled, _, _ = load(filled_bytes, filled_name, filled_page_no)
    if filled is None:
        raise ValueError("The filled-in scan is not an image or PDF.")
    H, inliers = align(blank, filled)
    if H is None or inliers < 25:
        raise ValueError(f"Could not line the scan up with the blank handout ({inliers} matching points). "
                         "Check both are the same handout and the photo shows the whole page.")
    h, w = blank.shape[:2]
    warped = cv2.warpPerspective(filled, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    px_per_mm = w / width_mm
    valid = cv2.warpPerspective(paper_area(filled), H, (w, h), flags=cv2.INTER_NEAREST)
    edge = 2 * int(2.0 * px_per_mm) + 1
    valid = cv2.erode(valid, np.ones((edge, edge), np.uint8))
    blank_flat, filled_flat = flat_gray(blank), flat_gray(warped)
    dx, dy, tiles = local_shifts(ink(blank_flat, 60), ink(filled_flat, 60), px_per_mm)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    filled_flat = cv2.remap(filled_flat, gx - dx, gy - dy, cv2.INTER_LINEAR, borderValue=255)
    warped = cv2.remap(warped, gx - dx, gy - dy, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    valid = cv2.remap(valid, gx - dx, gy - dy, cv2.INTER_NEAREST, borderValue=0)
    edges = cv2.Canny(cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY), 40, 120)
    mask = handwriting(blank_flat, filled_flat, valid, px_per_mm, edges, picture_areas(blank, px_per_mm))
    strokes = [(s / px_per_mm).round(2).tolist() for s in trace(mask, 5, 7) if len(s) >= 1]
    return {
        "strokes": strokes,
        "page": {"w": round(width_mm, 2), "h": round(h / px_per_mm, 2)},
        "matches": inliers,
        "tiles": tiles,
        "pages": pages,
        "blank": blank,
        "aligned": warped,
        "mask": mask,
    }


def preview(result, scale=1.0):
    """The blank page with the copied strokes drawn on it in blue, for checking before plotting."""
    image = result["blank"].copy()
    px = image.shape[1] / result["page"]["w"]
    for s in result["strokes"]:
        pts = (np.asarray(s) * px).round().astype(np.int32)
        if len(pts) == 1:
            cv2.circle(image, tuple(pts[0]), 2, (200, 60, 20), -1)
        else:
            cv2.polylines(image, [pts], False, (200, 60, 20), 2, cv2.LINE_AA)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else image
