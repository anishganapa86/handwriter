import json
import os
from pathlib import Path

import numpy as np
import torch

from .model import SIMILAR, load_checkpoint
from .strokes import core_metrics, from_offsets

BASE_DIR = Path(__file__).resolve().parent.parent / "weights" / "base"
CARRIER = "so {} is"


def base_stats():
    path = BASE_DIR / "stats.json"
    return json.loads(path.read_text()) if path.exists() else {"core": 155.0, "step": 18.4, "ink": 530.0}


class Writer:
    def __init__(self, checkpoint=None, device=None):
        self.path = Path(checkpoint) if checkpoint else BASE_DIR
        self.model, self.charset, self.meta = load_checkpoint(self.path, device or os.environ.get("HW_DEVICE", "cpu"))
        self.device = next(self.model.parameters()).device
        self.mu = np.array(self.meta["mu"], dtype=np.float64)
        self.std = np.array(self.meta["std"], dtype=np.float64)
        stats = base_stats()
        self.core_units = float(self.meta.get("core_units", stats["core"]))
        self.size_mm = float(self.meta.get("writer", {}).get("core_mm", 2.4))
        prime = self.meta.get("prime")
        self.prime = None
        if prime:
            self.prime = (torch.tensor(prime["offsets"], dtype=torch.float32), self.charset.encode(prime["text"] + " "))

    def sanitize(self, text):
        out = []
        for ch in text:
            if ch in self.charset.index:
                out.append(ch)
            elif SIMILAR.get(ch) in self.charset.index:
                out.append(SIMILAR[ch])
        return "".join(out)

    def unsupported(self, text):
        return self.charset.missing(text.replace("\n", ""))

    def sample_lines(self, texts, bias=0.0, seed=None, use_prime=True):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) if seed is not None else int(np.random.randint(0, 2 ** 31 - 1)))
        prime = self.prime if use_prime else None
        out = [None] * len(texts)
        todo = list(range(len(texts)))
        for attempt in range(2):
            if not todo:
                break
            tokens = [self.charset.encode(texts[i] + " ") for i in todo]
            results = []
            for start in range(0, len(todo), 32):
                results += self.model.sample(tokens[start:start + 32], bias=bias, prime=prime, generator=gen)
            failed = []
            for i, res in zip(todo, results):
                line = self._decode(res, texts[i])
                if line["ok"] or attempt == 1:
                    out[i] = line
                else:
                    failed.append(i)
            todo = failed
        return out

    def _decode(self, res, text):
        offsets = res["offsets"].numpy().astype(np.float64)
        offsets[:, :2] = offsets[:, :2] * self.std[:2] + self.mu[:2]
        strokes, chars = from_offsets(offsets, res["chars"].numpy())
        keep = [i for i, c in enumerate(chars) if np.mean(c < 0) < 0.5]
        if keep:
            strokes, chars = [strokes[i] for i in keep], [np.maximum(chars[i], 0) for i in keep]
        else:
            chars = [np.maximum(c, 0) for c in chars]
        words = split_words(strokes, chars, text)
        expected = len(text.split())
        width = max(s[:, 0].max() for s in strokes) - min(s[:, 0].min() for s in strokes)
        sane = width < self.core_units * (2.2 * len(text) + 6)
        ok = res["finished"] and len(words) == expected and all(w["strokes"] for w in words) and sane
        return {"text": text, "strokes": strokes, "chars": chars, "words": words, "ok": bool(ok)}

    def words(self, texts, bias=0.0, seed=None):
        rng = np.random.default_rng(seed)
        self.ensure_prime()
        jobs = []
        for idx, text in enumerate(texts):
            for order, run in enumerate(chunk_words(self.sanitize(text).split(), 38)):
                jobs.append({"owner": idx, "order": (order,), "words": run})
        finished = []
        for depth in range(5):
            if not jobs:
                break
            lines = self.sample_lines([" ".join(j["words"]) for j in jobs], bias, int(rng.integers(0, 2 ** 31 - 1)))
            retry = []
            for job, line in zip(jobs, lines):
                if line["ok"] or depth == 4:
                    finished.append((job, self.level(line)))
                elif len(job["words"]) > 1 and depth >= 1:
                    half = len(job["words"]) // 2
                    retry.append({**job, "order": job["order"] + (0,), "words": job["words"][:half]})
                    retry.append({**job, "order": job["order"] + (1,), "words": job["words"][half:]})
                elif len(job["words"]) == 1 and depth >= 1:
                    retry.append({**job, "carrier": True})
                else:
                    retry.append(job)
            jobs = [self.carried(j) if j.get("carrier") else j for j in retry]
        out = [[] for _ in texts]
        for job, words in sorted(finished, key=lambda item: (item[0]["owner"], item[0]["order"])):
            if job.get("carrier"):
                words = words[1:2]
            out[job["owner"]].extend(words)
        return out

    def carried(self, job):
        if job["words"][0].startswith("so ") and job["words"][0].endswith(" is"):
            return job
        return {**job, "words": [CARRIER.format(job["words"][0])]}

    def level(self, line):
        words = line["words"]
        cores = [w["core"] for w in words if w["strokes"]]
        if not cores or len(line["text"]) < 14:
            return words
        factor = float(np.clip(self.core_units / cores[0], 0.75, 1.3))
        for w in words:
            w["strokes"] = [s * factor for s in w["strokes"]]
            w["width"] *= factor
            w["gap"] *= factor
        return words

    def ensure_prime(self):
        if self.prime is not None or self.meta.get("prime") is False:
            return
        self.meta["prime"] = False
        text = "the quick brown fox"
        for seed in range(11, 31):
            line = self.sample_lines([text], bias=1.0, seed=seed, use_prime=False)[0]
            core = core_metrics(line["strokes"])["core"] if line["ok"] else 0
            if line["ok"] and 0.75 * self.core_units < core < 1.3 * self.core_units:
                raw = np.concatenate([np.concatenate([s, np.zeros((len(s), 1))], axis=1) for s in line["strokes"]])
                ends = np.cumsum([len(s) for s in line["strokes"]]) - 1
                raw[ends, 2] = 1.0
                offsets = raw.copy()
                offsets[1:, :2] = raw[1:, :2] - raw[:-1, :2]
                offsets[0, :2] = 0
                offsets[:, :2] = (offsets[:, :2] - self.mu[:2]) / self.std[:2]
                self.prime = (torch.tensor(offsets, dtype=torch.float32), self.charset.encode(text + " "))
                return

    def line(self, text, bias=0.0, seed=None):
        words = self.words([text], bias, seed)[0]
        strokes, x = [], 0.0
        for i, w in enumerate(words):
            if i > 0:
                x += float(np.clip(w["gap"] if w["gap"] > 0 else 0.9 * self.core_units, 0.35 * self.core_units, 1.6 * self.core_units))
            strokes += [np.stack([s[:, 0] + x, s[:, 1]], axis=1) for s in w["strokes"]]
            x += w["width"]
        return {"text": text, "strokes": strokes, "ok": all(w["strokes"] for w in words) and len(words) == len(self.sanitize(text).split())}

    def piece(self, text, bias=0.0, seed=None):
        clean = self.sanitize(text)
        tokens = clean.split()
        if not tokens:
            return []
        if len(clean) >= 6:
            return self.words([clean], bias, seed)[0]
        rng = np.random.default_rng(seed)
        self.ensure_prime()
        out = []
        for token in tokens:
            best = None
            for _ in range(4):
                line = self.sample_lines([CARRIER.format(token)], bias, int(rng.integers(0, 2 ** 31 - 1)))[0]
                if len(line["words"]) == 3 and line["words"][1]["strokes"]:
                    best = line["words"][1]
                    if line["ok"]:
                        break
            if best:
                out.append(best)
        return out


def chunk_words(words, limit):
    runs, current, size = [], [], 0
    for w in words:
        if current and size + len(w) + 1 > limit:
            runs.append(current)
            current, size = [], 0
        current.append(w)
        size += len(w) + 1
    if current:
        runs.append(current)
    if len(runs) > 1 and len(runs[-1]) == 1 and len(runs[-2]) > 2:
        runs[-1].insert(0, runs[-2].pop())
    return runs


def split_words(strokes, chars, text):
    word_of_char, w = [], 0
    for i, ch in enumerate(text + " "):
        if ch == " ":
            word_of_char.append(min(w, max(len(text.split()) - 1, 0)))
            if i > 0 and text[i - 1:i] != " ":
                w += 1
        else:
            word_of_char.append(w)
    count = len(text.split())
    if count == 0:
        return []
    owner = []
    for labels in chars:
        ids = [word_of_char[min(int(c), len(word_of_char) - 1)] for c in labels]
        owner.append(int(np.bincount(ids, minlength=count).argmax()))
    for i in range(1, len(owner)):
        owner[i] = max(owner[i], owner[i - 1]) if owner[i] < owner[i - 1] - 1 else owner[i]
    extents = {}
    for o, s in zip(owner, strokes):
        if len(s) > 3:
            lo, hi = extents.get(o, (np.inf, -np.inf))
            extents[o] = (min(lo, s[:, 0].min()), max(hi, s[:, 0].max()))
    for i, s in enumerate(strokes):
        o = owner[i]
        if len(s) <= 3 and o - 1 in extents and o in extents:
            cx = s[:, 0].mean()
            if cx < extents[o][0] and cx <= extents[o - 1][1] + 0.5 * (extents[o][0] - extents[o - 1][1]):
                owner[i] = o - 1
    metrics = core_metrics(strokes)
    origin = min(s[:, 0].min() for s in strokes)
    names = text.split()
    words, prev_right = [], None
    for wi in range(count):
        mine = [s for o, s in zip(owner, strokes) if o == wi]
        if not mine:
            words.append({"text": names[wi], "strokes": [], "width": 0.0, "gap": 0.0, "core": metrics["core"]})
            continue
        left = min(s[:, 0].min() for s in mine)
        right = max(s[:, 0].max() for s in mine)
        shifted = [np.stack([s[:, 0] - left, s[:, 1] - metrics["baseline"] - metrics["slope"] * (s[:, 0] - origin)], axis=1) for s in mine]
        gap = (left - prev_right) if prev_right is not None else 0.0
        words.append({"text": names[wi], "strokes": shifted, "width": right - left, "gap": float(gap), "core": metrics["core"]})
        prev_right = right
    return words
