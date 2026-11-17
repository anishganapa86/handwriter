"""Copy my notes: a synthetic handout, the same handout photographed with notes, highlighter and a hole
punch on it, and checks that only the notes come back, in the right place on the blank page."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw import copynotes, plotter

W, H = 1275, 1650                       # letter page at the working scale (about 5.9 px per mm)
PX = W / 215.9


def handout():
    page = np.full((H, W, 3), 255, np.uint8)
    lines = ["AP Biology   Chapter 10 Reading Guide", "A. Describe", "*photosynthesis", "*autotroph",
             "B. Write the chemical reaction.", "C. Explain where the oxygen comes from.", "D. Label the diagram."]
    for i, text in enumerate(lines):
        cv2.putText(page, text, (90, 130 + i * 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.rectangle(page, (300, 1420), (520, 1480), (40, 40, 40), 2)
    cv2.rectangle(page, (700, 1420), (920, 1480), (40, 40, 40), 2)
    return page


NOTES = [  # pixel polylines on the blank page: an answer line, a squiggle, an arrow and a label in a box
    np.array([[420, 510], [470, 470], [520, 515], [570, 470], [620, 515], [670, 470]]),
    np.array([[420, 700], [500, 690], [580, 710], [660, 690], [740, 705]]),
    np.array([[150, 950], [600, 950], [570, 930]]),
    np.array([[330, 1455], [360, 1440], [390, 1460], [430, 1445]]),
]


def filled_photo(blank):
    page = blank.copy()
    cv2.rectangle(page, (400, 470), (700, 525), (150, 230, 255), -1)          # yellow highlighter under an answer
    page = np.minimum(page, blank)
    for s in NOTES:
        cv2.polylines(page, [s.astype(np.int32)], False, (170, 60, 20), 4, cv2.LINE_AA)   # blue pen
    cv2.circle(page, (40, 400), 22, (15, 15, 15), -1)                         # hole punch
    # photograph it: shrink, tilt, put it on a darker desk
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[60, 40], [1010, 70], [990, 1330], [40, 1300]])
    photo = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst), (1060, 1380), borderValue=(70, 60, 55))
    return photo


def encode(image):
    return cv2.imencode(".png", image)[1].tobytes()


def test_copies_notes_onto_blank():
    blank = handout()
    result = copynotes.copy_notes(encode(blank), encode(filled_photo(blank)), "blank.png", "filled.png")
    assert result["matches"] >= 25
    strokes = [np.asarray(s) * PX for s in result["strokes"]]
    assert strokes, "no notes found"
    pts = np.concatenate(strokes)
    # every note is found within 2 mm of where it was written
    for note in NOTES:
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(note[:-1], note[1:])])
        near = np.min(np.linalg.norm(dense[:, None, :] - pts[None, :, :], axis=2), axis=1)
        assert np.median(near) < 2 * PX, f"note near {note[0]} missing"
    # nothing copied from the printed text, the hole punch or the desk edge
    printed = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY) < 128
    hits = sum(bool(printed[int(y), int(x)]) for x, y in pts if 0 <= int(x) < W and 0 <= int(y) < H)
    assert hits < 0.03 * len(pts)
    assert not ((pts[:, 0] < 70) & (pts[:, 1] > 370) & (pts[:, 1] < 430)).any(), "hole punch copied"
    assert pts[:, 0].min() > 50 and pts[:, 1].max() < H - 50, "page edge copied"


def test_unrelated_photo_is_refused():
    blank = handout()
    noise = (np.random.default_rng(0).random((600, 500, 3)) * 255).astype(np.uint8)
    try:
        copynotes.copy_notes(encode(blank), encode(noise), "blank.png", "other.png")
    except ValueError as e:
        assert "line the scan up" in str(e)
    else:
        raise AssertionError("an unrelated photo was accepted")


def test_reachable_matches_place_page():
    pen = plotter.Pen()
    strokes = [[[100.0, 120.0], [110.0, 125.0]], [[100.0, 2.0], [110.0, 4.0]]]   # mid-page, and at the very top edge
    flags = plotter.reachable(strokes, 215.9, 279.4, pen)
    assert flags[0] is True
    parts, problems = plotter.place_page([s for s, ok in zip(strokes, flags) if ok], 215.9, 279.4, pen)
    assert parts and not problems
