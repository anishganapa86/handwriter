import cv2
import numpy as np

PAGE_W_MM = 215.9
OUT_W = 1275


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def find_page(gray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(contour)
        if area < 0.3 * w * h or area > 0.985 * w * h:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return order_corners(approx)
    return None


def text_angle(gray):
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120, minLineLength=gray.shape[1] // 6, maxLineGap=12)
    if lines is None:
        return 0.0
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
    angles = [a for a in angles if abs(a) < 20]
    return float(np.median(angles)) if angles else 0.0


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = find_page(gray)
    if corners is not None:
        top = np.linalg.norm(corners[1] - corners[0])
        side = np.linalg.norm(corners[3] - corners[0])
        out_h = int(round(OUT_W * side / max(top, 1)))
        target = np.array([[0, 0], [OUT_W, 0], [OUT_W, out_h], [0, out_h]], dtype=np.float32)
        warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, target), (OUT_W, out_h),
                                     borderValue=(255, 255, 255))
        return warped, "perspective"
    angle = text_angle(gray)
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    out_h = int(round(OUT_W * h / w))
    return cv2.resize(rotated, (OUT_W, out_h), interpolation=cv2.INTER_AREA), f"rotation {angle:.1f} deg"


def overlap(a, b):
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1, y1 = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1e-6, min(a["w"] * a["h"], b["w"] * b["h"]))


def suggest_blanks(page):
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 12)
    mm = PAGE_W_MM / w
    boxes = []
    rects = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 0.035 * w or bh < 0.012 * h or bw * bh > 0.3 * w * h or bw > 0.97 * w:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or cv2.contourArea(contour) < 0.8 * bw * bh:
            continue
        inset = max(3, int(0.12 * min(bw, bh)))
        inner = binary[y + inset:y + bh - inset, x + inset:x + bw - inset]
        if inner.size and inner.mean() / 255 < 0.02:
            rects.append({"x": x * mm, "y": y * mm, "w": bw * mm, "h": bh * mm, "kind": "box"})
    for r in rects:
        if not any(overlap(r, o) > 0.6 for o in boxes):
            boxes.append(r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 22), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_h = 0.026 * h
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 0.045 * w or bh > 0.006 * h + 6 or bw > 0.97 * w:
            continue
        above = binary[max(0, int(y - line_h)):max(0, y - 2), x:x + bw]
        if above.size == 0 or above.mean() / 255 > 0.03:
            continue
        cand = {"x": x * mm, "y": (y - line_h) * mm, "w": bw * mm, "h": line_h * mm, "kind": "line"}
        edge = y * mm
        on_frame = any(o["kind"] == "box" and min(abs(edge - o["y"]), abs(edge - o["y"] - o["h"])) < 2.0
                       and cand["x"] < o["x"] + o["w"] and o["x"] < cand["x"] + cand["w"] for o in boxes)
        if not on_frame and not any(overlap(cand, o) > 0.3 for o in boxes):
            boxes.append(cand)
    boxes.sort(key=lambda b: (round(b["y"] / 4), b["x"]))
    return [{k: (round(v, 2) if k != "kind" else v) for k, v in b.items()} for b in boxes], {"w": PAGE_W_MM, "h": round(h * mm, 2)}
