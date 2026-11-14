import argparse
import json
from pathlib import Path

import numpy as np

from .strokes import densify

DEFAULT = {"gap": 1.15, "gap_spread": 0.25, "baseline": 0.12, "baseline_memory": 0.5, "line_slope": 0.004,
           "slant": None, "slant_spread": 2.0, "size_spread": 0.05, "margin": 0.6, "join": 0.0,
           "join_out": {}, "join_in": {}, "lines": 0, "words": 0}


def slant_of(strokes, step):
    angles = []
    for s in densify(strokes, step):
        if len(s) < 3:
            continue
        d = np.diff(s, axis=0)
        d = d[np.abs(d[:, 1]) > 1.2 * np.abs(d[:, 0])]
        d = np.where(d[:, 1:2] > 0, -d, d)
        angles.append(np.degrees(np.arctan2(d[:, 0], -d[:, 1])))
    angles = np.concatenate(angles) if angles else np.array([])
    return float(np.median(angles)) if len(angles) >= 8 else None


def lag_one(rows):
    a = np.concatenate([r[:-1] for r in rows if len(r) > 2] or [np.zeros(1)])
    b = np.concatenate([r[1:] for r in rows if len(r) > 2] or [np.zeros(1)])
    if len(a) < 8 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.5
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.95))


def fit(data_dir):
    from .fragments import Fragments
    from .glyphs import RULE_MM, load_real_words, photo_rules
    data_dir = Path(data_dir)
    lines = {}
    for path in sorted((data_dir / "photo_words").glob("*.json")):
        word = json.loads(path.read_text(encoding="utf-8"))
        if word.get("status") == "rejected" or "line" not in word or not word.get("strokes"):
            continue
        rules, pitch = photo_rules(data_dir, word.get("photo", ""))
        if not pitch:
            continue
        x0, y0, x1, y1 = word["box"]
        ref, cx = y0 + 0.62 * (y1 - y0), (x0 + x1) / 2
        below = [(b + k * cx, b, k) for b, k in rules if b + k * cx > ref]
        if not below or min(below)[0] - ref > 1.2 * pitch:
            continue
        _, b, k = min(below)
        mm = RULE_MM / pitch
        strokes = [np.asarray(s, dtype=np.float64) + [x0, y0] for s in word["strokes"]]
        strokes = [np.stack([s[:, 0], s[:, 1] - (b + k * s[:, 0])], axis=1) * mm for s in strokes if len(s) > 1]
        if not strokes:
            continue
        pts = np.concatenate(densify(strokes, 0.2))
        lines.setdefault((word["photo"], word["line"]), []).append({
            "order": word["order"], "left": pts[:, 0].min(), "right": pts[:, 0].max(),
            "base": float(np.percentile(pts[:, 1], 85)), "body": float(np.percentile(pts[:, 1], 85) - np.percentile(pts[:, 1], 25)),
            "slant": slant_of(strokes, 0.2), "wide": pts[:, 0].max() - pts[:, 0].min()})
    real = load_real_words(data_dir)
    frags = Fragments(real) if real else None
    xh = frags.xheight if frags else 1.8
    gaps, base_rows, slopes, slant_rows, sizes, margins = [], [], [], [], [], {}
    for (photo, _), words in lines.items():
        words.sort(key=lambda w: w["order"])
        margins.setdefault(photo, []).append(words[0]["left"])
        for a, b in zip(words, words[1:]):
            if b["order"] == a["order"] + 1 and -0.3 * xh < b["left"] - a["right"] < 5 * xh:
                gaps.append(max(b["left"] - a["right"], 0.1 * xh) / xh)
        long = [w for w in words if w["wide"] > 2.0 * xh]
        if len(long) >= 4:
            xs = np.array([(w["left"] + w["right"]) / 2 for w in long])
            ys = np.array([w["base"] for w in long])
            coef = np.polyfit(xs, ys, 1)
            slopes.append(coef[0])
            base_rows.append((ys - np.polyval(coef, xs)) / xh)
            body = np.array([w["body"] for w in long])
            sizes.extend(np.log(body / np.median(body)))
            lean = np.array([w["slant"] for w in long if w["slant"] is not None])
            if len(lean) >= 4:
                slant_rows.append(lean)
    style = dict(DEFAULT)
    if gaps:
        logs = np.log(gaps)
        style["gap"], style["gap_spread"] = float(np.exp(np.median(logs))), float(np.clip(logs.std(), 0.1, 0.7))
    if base_rows:
        flat = np.concatenate(base_rows)
        style["baseline"] = float(np.clip(np.percentile(np.abs(flat), 68), 0.03, 0.5))
        style["baseline_memory"] = lag_one(base_rows)
        style["line_slope"] = float(np.clip(np.std(slopes), 0.0, 0.03))
    if slant_rows:
        flat = np.concatenate(slant_rows)
        style["slant"] = float(np.median(flat))
        style["slant_spread"] = float(np.clip(np.median([np.percentile(np.abs(r - np.median(r)), 68) for r in slant_rows]), 0.5, 8.0))
    if sizes:
        style["size_spread"] = float(np.clip(0.5 * np.percentile(np.abs(sizes), 68), 0.02, 0.15))
    spread = [np.percentile(np.abs(np.array(v) - np.median(v)), 68) for v in margins.values() if len(v) >= 5]
    if spread:
        style["margin"] = float(np.clip(np.median(spread), 0.2, 3.0))
    if frags:
        out, into, total = {}, {}, []
        for sample in frags.samples:
            if not sample.cuts or len(sample.text) < 2:
                continue
            for i in range(1, len(sample.text)):
                joined = not bool(sample.lift[sample.cuts[i]])
                total.append(joined)
                out.setdefault(sample.text[i - 1].lower(), []).append(joined)
                into.setdefault(sample.text[i].lower(), []).append(joined)
        if total:
            overall = float(np.mean(total))
            shrink = lambda v: float((np.sum(v) + 4 * overall) / (len(v) + 4))
            style["join"] = overall
            style["join_out"] = {ch: shrink(v) for ch, v in out.items()}
            style["join_in"] = {ch: shrink(v) for ch, v in into.items()}
    style["lines"], style["words"] = len(lines), sum(len(w) for w in lines.values())
    style["xheight_mm"] = float(xh)
    return style


def load(data_dir):
    path = Path(data_dir) / "style.json"
    if path.exists():
        return {**DEFAULT, **json.loads(path.read_text(encoding="utf-8"))}
    return dict(DEFAULT)


def join_chance(style, a, b):
    overall = style.get("join", 0.0)
    if overall <= 0:
        return 0.0
    out = style["join_out"].get(a.lower(), overall)
    into = style["join_in"].get(b.lower(), overall)
    return float(np.clip(out * into / overall, 0.0, 1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
    args = parser.parse_args()
    found = fit(args.data)
    (Path(args.data) / "style.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in found.items() if not isinstance(v, dict)}))
    print("joins after:", {k: round(v, 2) for k, v in sorted(found["join_out"].items())})
    print("joins before:", {k: round(v, 2) for k, v in sorted(found["join_in"].items())})
