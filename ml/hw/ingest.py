import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import photo
from .evaluate import edit_distance

SKIP = "#"


def align_lines(counts, lengths):
    n, m = len(counts), len(lengths)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    back = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and cost[i, j] + 1.0 < cost[i + 1, j]:
                cost[i + 1, j] = cost[i, j] + 1.0
                back[(i + 1, j)] = (i, j)
            if j < m and cost[i, j] + 1.0 < cost[i, j + 1]:
                cost[i, j + 1] = cost[i, j] + 1.0
                back[(i, j + 1)] = (i, j)
            if i < n and j < m:
                step = cost[i, j] + min(0.95, abs(counts[i] - lengths[j]) / max(lengths[j], 1))
                if step < cost[i + 1, j + 1]:
                    cost[i + 1, j + 1] = step
                    back[(i + 1, j + 1)] = (i, j)
    pairs, node = {}, (n, m)
    while node != (0, 0):
        prev = back[node]
        if prev[0] == node[0] - 1 and prev[1] == node[1] - 1:
            pairs[prev[0]] = prev[1]
        node = prev
    return pairs


def spread_words(boxes, words):
    if len(boxes) == len(words):
        return [[w] for w in words]
    left, right = boxes[0][0], boxes[-1][2]
    total = sum(len(w) for w in words) + len(words) - 1
    out, cursor = [[] for _ in boxes], 0
    for word in words:
        centre = left + (cursor + len(word) / 2) / max(total, 1) * (right - left)
        cursor += len(word) + 1
        gaps = [0 if b[0] <= centre <= b[2] else min(abs(centre - b[0]), abs(centre - b[2])) for b in boxes]
        out[int(np.argmin(gaps))].append(word)
    return out


def distance(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b), 1)


def match_reads(reads, words):
    n, m = len(reads), len(words)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    back = {}

    def relax(i, j, ni, nj, step, label):
        if cost[i, j] + step < cost[ni, nj]:
            cost[ni, nj] = cost[i, j] + step
            back[(ni, nj)] = (i, j, label)

    for i in range(n + 1):
        for j in range(m + 1):
            if not np.isfinite(cost[i, j]):
                continue
            if i < n:
                relax(i, j, i + 1, j, 0.8, None)
            if j < m:
                relax(i, j, i, j + 1, 0.8, None)
            if i < n and j < m:
                one = 0.45 if words[j] == SKIP else distance(reads[i], words[j])
                relax(i, j, i + 1, j + 1, one, [words[j]])
            if i < n and j + 1 < m and SKIP not in words[j:j + 2]:
                relax(i, j, i + 1, j + 2, distance(reads[i], words[j] + words[j + 1]) + 0.12, words[j:j + 2])
    out, node = [None] * n, (n, m)
    while node != (0, 0):
        i, j, label = back[node]
        if label is not None:
            out[i] = label
        node = (i, j)
    return out


def relabel(spec_path, data_dir):
    spec_path, data_dir = Path(spec_path), Path(data_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for page in spec["pages"]:
        paths = sorted((data_dir / "photo_words").glob(f"{page['id']}_*.json"))
        words = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        order = sorted(range(len(words)), key=lambda k: (words[k]["line"], words[k]["order"]))
        transcript = [w for line in page.get("lines", []) for w in line.split()]
        labels = match_reads([words[k]["read"] for k in order], transcript)
        kept = 0
        for k, label in zip(order, labels):
            word = words[k]
            if word["status"] == "accepted":
                continue
            real = [w for w in (label or []) if w != SKIP]
            word["text"] = " ".join(real) if real else word["read"]
            word["prefilled"] = bool(real)
            word["status"] = "pending" if real and word["strokes"] else "rejected"
            kept += word["status"] == "pending"
            paths[k].write_text(json.dumps(word, ensure_ascii=False), encoding="utf-8")
        print(f"{page['id']}: {kept} of {len(words)} boxes matched to {len(transcript)} transcript words")


def run(spec_path, data_dir):
    spec_path, data_dir = Path(spec_path), Path(data_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for name in ("photos", "photo_words", "photo_crops"):
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    for page in spec["pages"]:
        pid = page["id"]
        image = cv2.cvtColor(np.array(Image.open(spec_path.parent / page["image"]).convert("RGB")), cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = page["roi"]
        scale = page.get("upscale", 2)
        image = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(data_dir / "photos" / f"{pid}.jpg"), image)
        boxes = photo.segment_words(image)
        lines = photo.group_lines(boxes, image)
        transcript = [line.split() for line in page.get("lines", [])]
        pairs = align_lines([len(line) for line in lines], [len(words) for words in transcript])
        ordered, meta = [], []
        for li, line in enumerate(lines):
            guesses = spread_words([boxes[i] for i in line], transcript[pairs[li]]) if li in pairs else [None] * len(line)
            for order, (index, guess) in enumerate(zip(line, guesses)):
                ordered.append(boxes[index])
                meta.append({"line": li, "order": order, "guess": guess, "exact": li in pairs and len(line) == len(transcript[pairs[li]])})
        print(f"{pid}: {len(ordered)} words on {len(lines)} lines, {len(pairs)} lines matched to the transcript, "
              f"{sum(m['exact'] for m in meta)} words matched one to one")
        words = photo.derender(image, ordered, data_dir / "photo_crops", pid)
        for word, info in zip(words, meta):
            guess = info["guess"]
            text = word["text"] if guess is None else " ".join(w for w in guess if w != SKIP)
            scribble = guess is not None and (not guess or all(w == SKIP for w in guess))
            word.update({"photo": pid, "line": info["line"], "order": info["order"], "read": word["text"], "text": text,
                         "prefilled": guess is not None, "exact": info["exact"],
                         "status": "rejected" if scribble or not word["strokes"] else "pending",
                         "crop": f"/files/photo_crops/{word['id']}.png"})
            (data_dir / "photo_words" / f"{word['id']}.json").write_text(json.dumps(word, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()
    if not args.relabel:
        run(args.spec, args.data)
    relabel(args.spec, args.data)
