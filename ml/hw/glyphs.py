import json
import re
from pathlib import Path

import numpy as np

from . import style as style_fit
from .fragments import Fragments

XHEIGHT_LETTERS = "acemnorsuvwxz"
FLAT_LETTERS = "abcdehiklmnorstuvwxz"
MIN_VARIANT_POINTS = 1
RULE_MM = 8.0
EDGES = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_rules = {}


def chaikin(points, rounds=2):
    for _ in range(rounds):
        if len(points) < 3:
            break
        q = 0.75 * points[:-1] + 0.25 * points[1:]
        r = 0.25 * points[:-1] + 0.75 * points[1:]
        points = np.concatenate([points[:1], np.stack([q, r], axis=1).reshape(-1, 2), points[-1:]])
    return points


def photo_rules(data_dir, photo_id):
    if photo_id not in _rules:
        import cv2
        from .photo import find_rules
        image = cv2.imread(str(Path(data_dir) / "photos" / f"{photo_id}.jpg"))
        rules = find_rules(image) if image is not None else []
        width = image.shape[1] if image is not None else 1
        mids = sorted(b + k * width / 2 for b, k in rules)
        pitch = float(np.median(np.diff(mids))) if len(mids) > 3 else None
        _rules[photo_id] = (rules, pitch)
    return _rules[photo_id]


def load_real_words(data_dir):
    out = {}
    for path in sorted((Path(data_dir) / "photo_words").glob("*.json")):
        word = json.loads(path.read_text(encoding="utf-8"))
        text = word.get("text", "").strip()
        if word.get("status") != "accepted" or not text or " " in text or "box" not in word or not word.get("strokes"):
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
        strokes = []
        height = y1 - y0
        dotted = any(ch in text for ch in "ij")
        marked = any(ch in text for ch in ".,;:!?'\"")
        for raw in word["strokes"]:
            local = np.asarray(raw, dtype=np.float64)
            if np.ptp(local, axis=0).max() < 0.12 * height:
                high = local[:, 1].mean() < 0.5 * height
                if not ((dotted and high) or marked):
                    continue
            pts = local + [x0, y0]
            pts = chaikin(pts) if len(pts) > 2 else pts
            strokes.append(np.stack([pts[:, 0], pts[:, 1] - (b + k * pts[:, 0])], axis=1) * mm)
        if not strokes:
            continue
        left = min(s[:, 0].min() for s in strokes)
        out.setdefault(text, []).append([np.stack([s[:, 0] - left, s[:, 1]], axis=1) for s in strokes])
    flat = [max(s[:, 1].max() for s in v) for text, variants in out.items() if not any(ch in text for ch in "gjpqyf,;")
            for v in variants]
    sink = float(np.median(flat)) if flat else 0.0
    return {text: [[np.stack([s[:, 0], s[:, 1] - sink], axis=1) for s in v] for v in variants] for text, variants in out.items()}


def normalise(strokes, baseline):
    cleaned = []
    for stroke in strokes:
        pts = np.asarray(stroke, dtype=np.float64)
        if pts.ndim == 2 and len(pts) >= MIN_VARIANT_POINTS:
            cleaned.append(pts[:, :2])
    if not cleaned:
        return None
    left = min(s[:, 0].min() for s in cleaned)
    return [np.stack([s[:, 0] - left, s[:, 1] - baseline], axis=1).round(3).tolist() for s in cleaned]


def load_library(path):
    path = Path(path)
    if not path.exists():
        return {"glyphs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_glyph(path, char, variants, baseline, replace=True):
    library = load_library(path)
    cleaned = [v for v in (normalise(v, baseline) for v in variants) if v]
    existing = [] if replace else library["glyphs"].get(char, [])
    if cleaned or replace:
        library["glyphs"][char] = existing + cleaned
    if not library["glyphs"].get(char):
        library["glyphs"].pop(char, None)
    Path(path).write_text(json.dumps(library, ensure_ascii=False), encoding="utf-8")
    return library


def summary(library):
    return {ch: len(v) for ch, v in library["glyphs"].items()}


class GlyphWriter:
    def __init__(self, path, data_dir=None):
        self.path = Path(path)
        self.real = load_real_words(data_dir) if data_dir else {}
        self.fragments = Fragments(self.real) if self.real else None
        self.use_words = True
        self.joins = 1.0
        self.style = style_fit.load(data_dir) if data_dir else dict(style_fit.DEFAULT)
        self.keys = []
        library = load_library(path)
        self.glyphs = {ch: [[np.asarray(s, dtype=np.float64) for s in variant] for variant in variants]
                       for ch, variants in library["glyphs"].items() if variants}
        for ch, variants in self.glyphs.items():
            if len(ch) != 1 or not ch.isalnum() or len(variants) < 2:
                continue
            tall = [max(float(np.ptp(np.concatenate(v)[:, 1])), 1e-6) for v in variants]
            typical = float(np.median(tall))
            for v, h in zip(variants, tall):
                factor = float(np.clip(typical / h, 0.75, 1.3))
                for i, s in enumerate(v):
                    v[i] = s * factor
            if ch not in "gjpqyf":
                floors = [max(s[:, 1].max() for s in v) for v in variants]
                level = float(np.median(floors))
                for v, floor in zip(variants, floors):
                    for i, s in enumerate(v):
                        v[i] = s + [0.0, level - floor]
        heights = [-min(s[:, 1].min() for s in v) for ch in XHEIGHT_LETTERS for v in self.glyphs.get(ch, [])]
        self.core_units = float(np.median(heights)) if heights else 2.4
        self.size_mm = self.core_units if self.core_units <= 2.4 else 1.9
        widths = [max(s[:, 0].max() for s in v) for ch in "abcdeghknopqsuvxyz" for v in self.glyphs.get(ch, [])]
        self.em = float(np.median(widths)) if widths else self.core_units
        bottoms = [max(s[:, 1].max() for s in v) for ch in FLAT_LETTERS for v in self.glyphs.get(ch, [])]
        self.sink = float(np.median(bottoms)) if bottoms else 0.0
        self.keys = sorted((k for k in self.glyphs if len(k) > 1), key=len, reverse=True)
        leans = [style_fit.slant_of(v, 0.1) for ch in "abcdefghijklmnopqrstuvwxyz" for v in self.glyphs.get(ch, [])[:1]]
        leans = [a for a in leans if a is not None]
        self.native_slant = float(np.median(leans)) if leans else 0.0
        target = self.style.get("slant")
        self.lean = float(np.tan(np.radians(target)) - np.tan(np.radians(self.native_slant))) if target is not None and leans else 0.0

    def sanitize(self, text):
        return text

    def unsupported(self, text):
        return sorted({ch for ch in text if ch not in self.glyphs and not ch.isspace()})

    def mess(self, bias):
        return float(np.clip(1.0 - (bias - 0.15) / 1.35, 0.0, 1.0))

    def units(self, text):
        out, i = [], 0
        while i < len(text):
            key = next((k for k in self.keys if text.startswith(k, i)), text[i])
            if key in self.glyphs:
                out.append((key, self.glyphs[key], 1.0))
            i += len(key)
        return out

    def assembled(self, core, rng):
        if not self.fragments or not core:
            return None
        made = self.fragments.compose(core, rng)
        if made:
            return [(core, [made[0]], 0.35)]
        if len(core) > 1 and core[0].isupper():
            lower = self.fragments.compose(core[0].lower() + core[1:], rng) if core[0] not in self.glyphs else None
            rest = self.fragments.compose(core[1:], rng)
            if rest and core[0] in self.glyphs:
                return self.units(core[0]) + [(core[1:], [rest[0]], 0.35)]
            if lower:
                return [(core, [lower[0]], 0.35)]
        return None

    def plan(self, text, rng):
        if self.use_words and self.real:
            if text in self.real:
                return [(text, self.real[text], 0.35)]
            lead, core, tail = EDGES.match(text).groups()
            if core in self.real:
                return self.units(lead) + [(core, self.real[core], 0.35)] + self.units(tail)
            made = self.assembled(core, rng)
            if made:
                return self.units(lead) + made + self.units(tail)
        return self.units(text)

    def warp(self, strokes, rng, amount):
        if amount <= 0:
            return strokes
        span = 2.6 * self.core_units
        p1, p2, p3, p4 = rng.uniform(0, 2 * np.pi, 4)
        a1, a2 = rng.normal(0, amount, 2)
        out = []
        for s in strokes:
            x = s[:, 0] + a1 * np.sin(2 * np.pi * s[:, 1] / span + p1) + 0.5 * a2 * np.sin(4 * np.pi * s[:, 0] / span + p3)
            y = s[:, 1] + a2 * np.sin(2 * np.pi * s[:, 0] / span + p2) + 0.5 * a1 * np.sin(4 * np.pi * s[:, 1] / span + p4)
            out.append(np.stack([x, y], axis=1))
        return out

    def connect(self, strokes, before, after, fresh):
        core = self.core_units

        def ends(indices, own_start, low, high):
            found = []
            for i in indices:
                s = strokes[i]
                if s is None or len(s) < 4:
                    continue
                for at_end in (True, False):
                    if not at_end and i not in own_start:
                        continue
                    p = s[-1] if at_end else s[0]
                    if low * core <= p[1] <= high * core:
                        found.append((p[0], i, at_end))
            return found

        exits = ends(before, set(before) - fresh, -1.7, 0.4)
        entries = ends(after, set(after), -1.6, 0.35)
        if not exits or not entries:
            return None
        _, i, exit_at_end = max(exits)
        _, j, entry_at_end = min(entries)
        if i == j:
            return None
        tail = strokes[i] if exit_at_end else strokes[i][::-1]
        head = strokes[j][::-1] if entry_at_end else strokes[j]
        gap = head[0] - tail[-1]
        reach = float(np.linalg.norm(gap))
        if not (-0.2 * core <= gap[0] <= 1.7 * core) or abs(gap[1]) > 1.4 * core or reach < 1e-6:
            return None

        def heading(points):
            d = points[-1] - points[0]
            n = np.linalg.norm(d)
            d = d / n if n else np.array([1.0, 0.0])
            d = 0.55 * d + 0.45 * np.array([1.0, 0.0])
            return d / max(np.linalg.norm(d), 1e-9)

        t0, t1 = heading(tail[-4:]) * reach * 0.9, heading(head[:4]) * reach * 0.9
        u = np.linspace(0, 1, max(4, int(reach / (0.12 * core)) + 2))[1:-1, None]
        bridge = ((2 * u ** 3 - 3 * u ** 2 + 1) * tail[-1] + (u ** 3 - 2 * u ** 2 + u) * t0
                  + (-2 * u ** 3 + 3 * u ** 2) * head[0] + (u ** 3 - u ** 2) * t1)
        strokes[i] = np.concatenate([tail, bridge, head])
        return i, j

    def word(self, text, rng, mess, last):
        core = self.core_units
        vary = mess / 0.4
        strokes, x = [], 0.0
        word_lean = self.lean + rng.normal(0, np.tan(np.radians(self.style["slant_spread"])) * 0.6 * vary)
        word_size = float(np.exp(rng.normal(0, self.style["size_spread"] * 0.6 * vary)))
        drift, previous = 0.0, None
        for key, variants, wobble in self.plan(text, rng):
            sheet = wobble == 1.0
            options = [i for i in range(len(variants)) if i != last.get(key)] or [0]
            pick = int(rng.choice(options))
            last[key] = pick
            sx = (word_size if sheet else 1.0) * (1 + wobble * rng.normal(0, 0.03 * vary))
            sy = (word_size if sheet else 1.0) * (1 + wobble * rng.normal(0, 0.03 * vary))
            shear = (word_lean if sheet else 0.0) + wobble * rng.normal(0, 0.025 * vary)
            drift = 0.6 * drift + rng.normal(0, 0.035 * core * vary)
            dy = drift - (self.sink if sheet else 0.0)
            shaped = self.warp(variants[pick], rng, 0.03 * core * vary) if sheet else variants[pick]
            join = None
            if previous and sheet and previous["sheet"] and key[0].islower() and previous["key"][-1].islower():
                if rng.random() < self.joins * style_fit.join_chance(self.style, previous["key"][-1], key[0]):
                    join = previous
            if strokes:
                x += self.em * (0.07 if join else max(0.05, 0.13 + rng.normal(0, 0.03 * vary)))
            left = min(s[:, 0].min() for s in shaped)
            first = len(strokes)
            for s in shaped:
                strokes.append(np.stack([(s[:, 0] - left) * sx - s[:, 1] * shear + x, s[:, 1] * sy + dy], axis=1))
            mine = list(range(first, len(strokes)))
            fresh = set()
            if join:
                linked = self.connect(strokes, join["indices"], mine, join["fresh"])
                if linked:
                    kept, gone = linked
                    strokes[gone] = None
                    mine = [kept if i == gone else i for i in mine]
                    fresh = {kept}
            x = max(x, max(strokes[i][:, 0].max() for i in mine))
            previous = {"key": key, "sheet": sheet, "indices": mine, "fresh": fresh}
        strokes = [s for s in strokes if s is not None]
        if not strokes:
            return None
        left = min(s[:, 0].min() for s in strokes)
        strokes = [np.stack([s[:, 0] - left, s[:, 1]], axis=1) for s in strokes]
        return {"text": text, "strokes": strokes, "width": max(s[:, 0].max() for s in strokes), "gap": 0.0, "core": core}

    def run(self, text, rng, mess, last):
        words = []
        for token in text.split():
            word = self.word(token, rng, mess, last)
            if word is None:
                continue
            if words:
                spread = min(self.style["gap_spread"], 0.4) * mess / 0.4
                word["gap"] = self.core_units * max(0.9, self.style["gap"] * float(np.exp(rng.normal(0, spread))))
            words.append(word)
        return words

    def words(self, texts, bias=0.0, seed=None):
        rng = np.random.default_rng(seed)
        mess, last = self.mess(bias), {}
        return [self.run(text, rng, mess, last) for text in texts]

    def piece(self, text, bias=0.0, seed=None):
        return self.words([text], bias, seed)[0]

    def line(self, text, bias=0.0, seed=None):
        strokes, x = [], 0.0
        for i, w in enumerate(self.piece(text, bias, seed)):
            x += w["gap"] if i else 0.0
            strokes += [np.stack([s[:, 0] + x, s[:, 1]], axis=1) for s in w["strokes"]]
            x += w["width"]
        return {"text": text, "strokes": strokes, "ok": bool(strokes) and not self.unsupported(text)}

