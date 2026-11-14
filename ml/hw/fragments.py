import numpy as np

from .strokes import densify

WIDE = "mw"
NARROW = "iljt.,;:'!"
MEDIUM = "frsc"
STEP_MM = 0.2


def prior_width(ch):
    if ch.isupper():
        return 1.5
    if ch in WIDE:
        return 1.6
    if ch in NARROW:
        return 0.55
    if ch in MEDIUM:
        return 0.8
    return 1.0


class Sample:
    def __init__(self, text, strokes, xheight):
        self.text = text
        extents = [float(np.ptp(s, axis=0).max()) for s in strokes]
        body = [s for s, e in zip(strokes, extents) if e >= 0.45 * xheight]
        self.marks = [s for s, e in zip(strokes, extents) if e < 0.45 * xheight]
        if not body:
            body, self.marks = strokes, []
        body = [b[::-1] if b[-1, 0] < b[0, 0] else b for b in body]
        body.sort(key=lambda b: b[:, 0].min())
        dense = [d for d in densify(body, STEP_MM) if len(d)]
        self.points = np.concatenate(dense)
        self.starts = np.cumsum([0] + [len(d) for d in dense])[:-1]
        self.lift = np.zeros(len(self.points) + 1, dtype=bool)
        self.lift[self.starts] = True
        self.lift[len(self.points)] = True
        self.cuts = None

    def align(self, widths, xheight, valley=0.35):
        pts, n = self.points, len(self.text)
        count = len(pts)
        if n == 1 or count < 2 * n:
            self.cuts = None if count < 2 * n and n > 1 else [0, count]
            return
        x = np.concatenate([pts[:, 0], [pts[-1, 0]]])
        height = np.concatenate([np.clip(-pts[:, 1] / xheight, 0, 1.5), [0.0]])
        penalty = np.where(self.lift, 0.0, valley * height)
        expected = np.array([widths.get(ch.lower(), prior_width(ch)) * (1.5 if ch.isupper() else 1.0) for ch in self.text])
        scale = max(x[count] - x[0], 0.5 * xheight) / expected.sum()
        expected = expected * scale
        advance = x[None, :] - x[:, None]
        allowed = np.arange(count + 1)[:, None] < np.arange(count + 1)[None, :]
        best = np.full((n + 1, count + 1), np.inf)
        back = np.zeros((n + 1, count + 1), dtype=np.int64)
        best[0, 0] = 0.0
        edge = np.concatenate([penalty[:count], [0.0]])
        for i in range(1, n + 1):
            cost = best[i - 1][:, None] + ((advance - expected[i - 1]) / expected[i - 1]) ** 2
            cost = np.where(allowed, cost, np.inf)
            back[i] = cost.argmin(axis=0)
            best[i] = cost.min(axis=0) + edge
        cuts, p = [count], count
        for i in range(n, 0, -1):
            p = int(back[i, p])
            cuts.append(p)
        self.cuts = cuts[::-1]

    def piece(self, a, b):
        lo, hi = self.cuts[a], self.cuts[b]
        strokes, start = [], lo
        for edge in list(self.starts[(self.starts > lo) & (self.starts < hi)]) + [hi]:
            if edge > start:
                strokes.append(self.points[start:edge].copy())
            start = int(edge)
        x0, x1 = self.points[lo:hi, 0].min(), self.points[lo:hi, 0].max()
        first, last = a == 0, b == len(self.text)
        marks = [m.copy() for m in self.marks
                 if (x0 if not first else -np.inf) - 0.15 <= m[:, 0].mean() <= (x1 if not last else np.inf) + 0.15]
        return {"strokes": strokes, "marks": marks, "joined_in": not self.lift[lo], "joined_out": not self.lift[hi]}


class Fragments:
    def __init__(self, real):
        heights = [-np.percentile(np.concatenate([s[:, 1] for s in v]), 35) for variants in real.values() for v in variants]
        self.xheight = float(np.clip(np.median(heights), 0.8, 4.0)) if heights else 1.8
        self.samples = []
        for text, variants in real.items():
            if not text.replace("'", "").isalpha() or not text.isascii():
                continue
            for strokes in variants:
                self.samples.append(Sample(text, strokes, self.xheight))
        widths = {}
        for _ in range(3):
            for sample in self.samples:
                sample.align(widths, self.xheight)
            seen = {}
            for sample in self.samples:
                if sample.cuts and len(sample.text) > 1:
                    x = np.concatenate([sample.points[:, 0], [sample.points[-1, 0]]])
                    total = x[sample.cuts[-1]] - x[sample.cuts[0]]
                    unit = total / max(sum(prior_width(c) for c in sample.text), 1e-6)
                    for i, ch in enumerate(sample.text):
                        seen.setdefault(ch.lower(), []).append((x[sample.cuts[i + 1]] - x[sample.cuts[i]]) / max(unit, 1e-6))
            widths = {ch: float(np.clip(0.5 * np.median(v) + 0.5 * prior_width(ch), 0.3, 2.4)) for ch, v in seen.items() if len(v) >= 3}
        self.widths = widths
        self.index = {}
        for si, sample in enumerate(self.samples):
            if not sample.cuts:
                continue
            n = len(sample.text)
            for a in range(n):
                for b in range(a + 1, n + 1):
                    self.index.setdefault(sample.text[a:b], []).append((si, a, b))

    def letters(self):
        return {key for key in self.index if len(key) == 1}

    def cover(self, word, rng):
        n = len(word)
        best = [np.inf] * (n + 1)
        choice = [None] * (n + 1)
        best[0] = 0.0
        for i in range(n):
            if not np.isfinite(best[i]):
                continue
            for j in range(i + 1, n + 1):
                sources = self.index.get(word[i:j])
                if not sources:
                    continue
                fit = [s for s in sources if (s[1] == 0) == (i == 0) and (s[2] == len(self.samples[s[0]].text)) == (j == n)]
                cost = 1.0 + (0.9 if j - i == 1 else 0.0) + (0.0 if fit else 0.45)
                if best[i] + cost < best[j]:
                    best[j] = best[i] + cost
                    pool = fit or sources
                    choice[j] = (i, pool[int(rng.integers(len(pool)))])
        if not np.isfinite(best[n]):
            return None
        units, j = [], n
        while j > 0:
            i, source = choice[j]
            units.append(source)
            j = i
        return units[::-1]

    def compose(self, word, rng):
        units = self.cover(word, rng)
        if units is None:
            return None
        strokes, marks, cursor, joined = [], [], 0.0, False
        for si, a, b in units:
            piece = self.samples[si].piece(a, b)
            if not piece["strokes"]:
                continue
            entry = piece["strokes"][0][0]
            if strokes and joined and piece["joined_in"]:
                shift = strokes[-1][-1][0] + STEP_MM - entry[0]
            elif strokes:
                left = min(s[:, 0].min() for s in piece["strokes"])
                shift = cursor + 0.18 * self.xheight - left
            else:
                shift = -min(s[:, 0].min() for s in piece["strokes"])
            moved = [s + [shift, 0.0] for s in piece["strokes"]]
            marks += [m + [shift, 0.0] for m in piece["marks"]]
            if strokes and joined and piece["joined_in"]:
                tail, head = strokes[-1], moved[0]
                meet = (tail[-1, 1] + head[0, 1]) / 2
                k = min(6, len(tail), len(head))
                ramp = np.linspace(0, 1, k + 1)[1:]
                tail[-k:, 1] += ramp * (meet - tail[-1, 1])
                head[:k, 1] += ramp[::-1] * (meet - head[0, 1])
                strokes[-1] = np.concatenate([tail, head])
                moved = moved[1:]
            strokes += moved
            cursor = max(cursor, max(s[:, 0].max() for s in strokes[-max(1, len(moved)):]))
            joined = piece["joined_out"]
        if not strokes:
            return None
        out = strokes + marks
        left = min(s[:, 0].min() for s in out)
        return [np.stack([s[:, 0] - left, s[:, 1]], axis=1) for s in out], len(units)
