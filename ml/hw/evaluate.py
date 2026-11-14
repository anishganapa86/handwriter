import html
import json

import numpy as np

from .dataset import model_strokes
from .strokes import bounds, densify

EVAL_BIAS = 0.6
_recognizer = None


def dtw(a, b):
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        d[i, j] = cost[i - 1, j - 1] + np.minimum(np.minimum(d[i - 1, j], d[i, j - 1]), d[i - 1, j - 1])
    return float(d[n, m] / (n + m))


def features(strokes, core):
    dense = densify(strokes, core / 5)
    origin = dense[0][0]
    pos, direction = [], []
    for s in dense:
        p = (s - origin) / core
        if len(p) > 1:
            delta = np.diff(p, axis=0)
            unit = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
            unit = np.concatenate([unit[:1], unit])
        else:
            unit = np.zeros((1, 2))
        up = np.zeros((len(p), 1))
        up[0, 0] = 1.0
        pos.append(p)
        direction.append(np.concatenate([unit, up], axis=1))
    return np.concatenate(pos), np.concatenate(direction)


def stroke_distance(real, generated, core):
    rp, rd = features(real, core)
    gp, gd = features(generated, core)
    return {"dtw_dir": dtw(rd, gd), "dtw_pos": dtw(rp, gp)}


def render(strokes, height=96, pad=10):
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = bounds(strokes)
    scale = (height - 2 * pad) / max(y1 - y0, 1e-6)
    width = int((x1 - x0) * scale) + 2 * pad
    img = Image.new("RGB", (max(width, height), height), "white")
    draw = ImageDraw.Draw(img)
    for s in strokes:
        pts = [((p[0] - x0) * scale + pad, (p[1] - y0) * scale + pad) for p in s]
        if len(pts) == 1:
            draw.ellipse([pts[0][0] - 1.5, pts[0][1] - 1.5, pts[0][0] + 1.5, pts[0][1] + 1.5], fill="black")
        else:
            draw.line(pts, fill="black", width=3, joint="curve")
    return img


def load_recognizer(name="microsoft/trocr-small-handwritten"):
    global _recognizer
    if _recognizer is None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from .model import pick_device
            processor = TrOCRProcessor.from_pretrained(name)
            model = VisionEncoderDecoderModel.from_pretrained(name).to(pick_device()).eval()

            def recognise(images):
                with torch.no_grad():
                    pixels = processor(images=images, return_tensors="pt").pixel_values.to(model.device)
                    ids = model.generate(pixels, max_new_tokens=96)
                return processor.batch_decode(ids, skip_special_tokens=True)

            _recognizer = recognise
        except Exception as exc:
            print(f"recognizer unavailable: {exc}")
            _recognizer = False
    return _recognizer or None


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(predictions, references):
    errors = sum(edit_distance(" ".join(p.split()), r) for p, r in zip(predictions, references))
    return errors / max(1, sum(len(r) for r in references))


def evaluate(writer, holdout, profile, stats, samples_per_line=3, seed=1234, use_recognizer=True):
    core = writer.core_units
    lines = [writer.line(s["text"], EVAL_BIAS, seed + 7919 * (i * samples_per_line + j))
             for i, s in enumerate(holdout) for j in range(samples_per_line)]
    rows, images, refs = [], [], []
    for i, sample in enumerate(holdout):
        real = model_strokes(sample, profile, stats, resample=False)
        gens = lines[i * samples_per_line:(i + 1) * samples_per_line]
        dists = [stroke_distance(real, g["strokes"], core) for g in gens]
        best = int(np.argmin([d["dtw_dir"] for d in dists]))
        rows.append({"text": sample["text"], "real": real, "generated": [g["strokes"] for g in gens], "best": best,
                     "dtw_dir": float(np.mean([d["dtw_dir"] for d in dists])),
                     "dtw_pos": float(np.mean([d["dtw_pos"] for d in dists])),
                     "complete": float(np.mean([g["ok"] for g in gens]))})
        for g in gens:
            images.append(render(g["strokes"]))
            refs.append(sample["text"])
    metrics = {"lines": len(holdout), "dtw_dir": float(np.mean([r["dtw_dir"] for r in rows])),
               "dtw_pos": float(np.mean([r["dtw_pos"] for r in rows])),
               "complete": float(np.mean([r["complete"] for r in rows])), "cer": None, "cer_real": None}
    recognise = load_recognizer() if use_recognizer else None
    if recognise:
        preds = []
        for start in range(0, len(images), 8):
            preds += recognise(images[start:start + 8])
        metrics["cer"] = cer(preds, refs)
        real_preds = []
        real_images = [render(r["real"]) for r in rows]
        for start in range(0, len(real_images), 8):
            real_preds += recognise(real_images[start:start + 8])
        metrics["cer_real"] = cer(real_preds, [r["text"] for r in rows])
    return metrics, rows


def better(candidate, champion):
    if champion is None:
        return True, "no previous champion"
    if candidate["complete"] < champion["complete"] - 0.05:
        return False, "more unfinished lines than the champion"
    c_cer, p_cer = candidate.get("cer"), champion.get("cer")
    cer_ok = c_cer is None or p_cer is None or c_cer <= p_cer + 0.02
    if candidate["dtw_dir"] < champion["dtw_dir"] * 0.995 and cer_ok:
        return True, "lower held-out DTW with no CER regression"
    if c_cer is not None and p_cer is not None and c_cer < p_cer - 0.02 and candidate["dtw_dir"] <= champion["dtw_dir"] * 1.01:
        return True, "lower CER with no DTW regression"
    return False, "no improvement on held-out DTW or CER"


def svg(strokes, height=70, colour="#1a2a6c"):
    x0, y0, x1, y1 = bounds(strokes)
    scale = (height - 12) / max(y1 - y0, 1e-6)
    width = (x1 - x0) * scale + 12
    paths = []
    for s in strokes:
        pts = " ".join(f"{(p[0] - x0) * scale + 6:.1f},{(p[1] - y0) * scale + 6:.1f}" for p in s)
        if len(s) == 1:
            x, y = pts.split(",")
            paths.append(f'<circle cx="{x}" cy="{y}" r="1.1" fill="{colour}"/>')
        else:
            paths.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')
    return f'<svg viewBox="0 0 {width:.0f} {height}" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(paths)}</svg>'


def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def write_report(path, version, candidate, rows, champion, champion_rows, promoted, reason, info):
    head = ["metric", f"candidate {version}", f"champion {champion['version']}" if champion else "champion"]
    names = [("dtw_dir", "Held-out DTW (shape)"), ("dtw_pos", "Held-out DTW (position)"),
             ("cer", "Recognizer CER, generated"), ("cer_real", "Recognizer CER, your real lines"),
             ("complete", "Lines finished cleanly")]
    table = "".join(f"<tr><td>{label}</td><td>{fmt(candidate.get(k))}</td><td>{fmt(champion['metrics'].get(k)) if champion else 'n/a'}</td></tr>"
                    for k, label in names)
    blocks = []
    for i, row in enumerate(rows):
        old = ""
        if champion_rows:
            cr = champion_rows[i]
            old = f'<div class="lab">champion</div>{svg(cr["generated"][cr["best"]], colour="#555")}'
        blocks.append(
            f'<section><h3>{html.escape(row["text"])}</h3><div class="lab">real</div>{svg(row["real"], colour="#111")}'
            f'<div class="lab">candidate (DTW {row["dtw_dir"]:.3f})</div>{svg(row["generated"][row["best"]])}{old}</section>')
    verdict = "PROMOTED" if promoted else "NOT PROMOTED"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Eval {version}</title><style>
body{{font:15px/1.5 system-ui,sans-serif;margin:32px auto;max-width:1100px;padding:0 16px;color:#1b1b1b}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}
section{{border-top:1px solid #ddd;margin-top:18px;padding-top:8px}}h3{{font-size:14px;margin:4px 0}}
.lab{{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:.06em;margin-top:6px}}
svg{{display:block;max-width:100%}}.v{{font-weight:700;color:{'#0a7a35' if promoted else '#a12b2b'}}}</style></head><body>
<h1>Checkpoint {version}</h1><p class="v">{verdict}: {html.escape(reason)}</p>
<p>{html.escape(json.dumps(info))}</p>
<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{table}</table>{''.join(blocks)}</body></html>"""
    path.write_text(page, encoding="utf-8")
