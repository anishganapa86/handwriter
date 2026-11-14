import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .strokes import bounds, capture_line_to_strokes, clean, core_metrics, densify, kinematic, resample_time, step_median, to_offsets

PHOTO_WEIGHT = 0.5
LINE_INK_MM = 5.5


def norm_text(text):
    return " ".join(text.split()).lower()


def load_samples(data_dir):
    data_dir = Path(data_dir)
    samples = []
    for path in sorted((data_dir / "sessions").glob("*.json")):
        session = json.loads(path.read_text(encoding="utf-8"))
        for line in session.get("lines", []):
            strokes = clean(capture_line_to_strokes(line))
            if not strokes or not line.get("text", "").strip():
                continue
            samples.append({"id": f"{session['sessionId']}:{line['id']}", "text": " ".join(line["text"].split()),
                            "strokes": strokes, "holdout": bool(line.get("holdout")), "source": "capture",
                            "timed": True, "group": session["sessionId"]})
    runs = {}
    for path in sorted((data_dir / "photo_words").glob("*.json")):
        word = json.loads(path.read_text(encoding="utf-8"))
        if "line" in word:
            runs.setdefault((word.get("photo", "photo"), word["line"]), []).append(word)
            continue
        if word.get("status") != "accepted" or not word.get("text", "").strip():
            continue
        strokes = clean([np.asarray(s, dtype=np.float64) for s in word["strokes"]])
        if strokes:
            samples.append({"id": f"photo:{path.stem}", "text": word["text"].strip(), "strokes": strokes,
                            "holdout": False, "source": "photo", "timed": False, "group": word.get("photo", "photo")})
    for (photo, line), words in sorted(runs.items()):
        words.sort(key=lambda w: w["order"])
        run = []
        for word in words + [None]:
            good = word is not None and word.get("status") == "accepted" and word.get("text", "").strip() and word.get("strokes")
            if good:
                origin = np.asarray(word["box"][:2], dtype=np.float64)
                strokes = clean([np.asarray(s, dtype=np.float64) + origin for s in word["strokes"]])
                if strokes:
                    run.append((word, strokes))
                    continue
            if run:
                key = f"{photo}:{line}:{run[0][0]['order']}"
                held = len(run) >= 3 and zlib.crc32(key.encode()) % 7 == 0
                if len(run) >= 2:
                    samples.append({"id": f"photoline:{key}", "text": " ".join(w["text"].strip() for w, _ in run),
                                    "strokes": [s for _, strokes in run for s in strokes], "holdout": held,
                                    "source": "photo", "timed": False, "group": photo})
                if not held:
                    for w, strokes in run:
                        samples.append({"id": f"photo:{w['id']}", "text": w["text"].strip(), "strokes": strokes,
                                        "holdout": False, "source": "photo", "timed": False, "group": photo})
            run = []
    return samples


def split(samples):
    held = {norm_text(s["text"]) for s in samples if s["holdout"] and s["source"] == "capture"}
    holdout = [s for s in samples if s["holdout"]]
    train = [s for s in samples if not s["holdout"] and (s["source"] != "capture" or norm_text(s["text"]) not in held)]
    return train, holdout


def new_symbols(samples, charset, minimum=2):
    counts = Counter(ch for s in samples for ch in s["text"] if ch not in charset.index and ch != " ")
    return "".join(sorted(ch for ch, n in counts.items() if n >= minimum))


def ink_length(strokes):
    return float(sum(np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1).sum() for s in strokes))


def writer_profile(samples, stats):
    timed = [s for s in samples if s["timed"]]
    groups, heights = {}, {}
    for s in samples:
        if not s["timed"] and len(s["text"]) >= 6:
            x0, y0, x1, y1 = bounds(s["strokes"])
            groups.setdefault(s["group"], []).append(ink_length(s["strokes"]) / len(s["text"].replace(" ", "")))
            if " " in s["text"]:
                ys = np.concatenate([k[:, 1] for k in s["strokes"]])
                heights.setdefault(s["group"], []).append(np.percentile(ys, 97) - np.percentile(ys, 3))
    photo_units = {g: stats["ink"] / float(np.median(c)) for g, c in groups.items()}
    if not timed:
        tall = [float(np.median(h)) * photo_units[g] for g, h in heights.items()] or [3.0 * stats["core"]]
        core_mm = LINE_INK_MM * stats["core"] / float(np.median(tall))
        return {"core_mm": core_mm, "units_per_mm": stats["core"] / core_mm, "dt": 0.01, "photo_units": photo_units}
    cores = [core_metrics(s["strokes"])["core"] for s in timed if len(s["text"]) >= 12] or \
            [core_metrics(s["strokes"])["core"] for s in timed]
    core_mm = float(np.median(cores))
    units = stats["core"] / core_mm
    dt = 0.01
    for _ in range(3):
        steps = [step_median(resample_time(s["strokes"], dt)) * units for s in timed]
        dt = float(np.clip(dt * stats["step"] / max(np.median(steps), 1e-6), 0.002, 0.05))
    return {"core_mm": core_mm, "units_per_mm": units, "dt": dt, "photo_units": photo_units}


def model_strokes(sample, profile, stats, resample=True):
    if sample["timed"]:
        scaled = [np.concatenate([s[:, :2] * profile["units_per_mm"], s[:, 2:3]], axis=1) for s in sample["strokes"]]
        return resample_time(scaled, profile["dt"]) if resample else [s[:, :2] for s in scaled]
    units = profile["photo_units"].get(sample["group"], profile["units_per_mm"])
    scaled = [s[:, :2] * units for s in sample["strokes"]]
    return kinematic(scaled, stats["step"]) if resample else scaled


def encode(sample, profile, stats, mu, std):
    offsets = to_offsets(model_strokes(sample, profile, stats))
    offsets[:, :2] = (offsets[:, :2] - np.asarray(mu[:2], dtype=np.float32)) / np.asarray(std[:2], dtype=np.float32)
    return offsets


def augment(offsets, mu, std, rng):
    out = offsets.copy()
    raw = out[:, :2] * np.asarray(std[:2], dtype=np.float32) + np.asarray(mu[:2], dtype=np.float32)
    sx, sy = rng.uniform(0.94, 1.06, 2)
    sh = rng.uniform(-0.05, 0.05)
    mixed = np.stack([sx * raw[:, 0] + sh * raw[:, 1], sy * raw[:, 1]], axis=1)
    out[:, :2] = (mixed - np.asarray(mu[:2], dtype=np.float32)) / np.asarray(std[:2], dtype=np.float32)
    out[0, :2] = offsets[0, :2]
    return out


def collate(items, charset, device):
    longest = max(len(o) for o, _, _ in items)
    widest = max(len(t) for _, t, _ in items)
    x = torch.zeros(len(items), longest, 3)
    x_mask = torch.zeros(len(items), longest)
    c = torch.zeros(len(items), widest, charset.size)
    c_mask = torch.zeros(len(items), widest)
    for i, (offsets, text, weight) in enumerate(items):
        x[i, : len(offsets)] = torch.from_numpy(offsets)
        x_mask[i, : len(offsets)] = weight
        tokens = charset.encode(text)
        c[i, torch.arange(len(tokens)), torch.tensor(tokens)] = 1.0
        c_mask[i, : len(tokens)] = 1.0
    return x.to(device), x_mask.to(device), c.to(device), c_mask.to(device)
