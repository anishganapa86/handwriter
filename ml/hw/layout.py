import math

import numpy as np

LETTER = {"width": 215.9, "height": 279.4, "margin_left": 22.0, "margin_right": 18.0, "margin_top": 24.0,
          "margin_bottom": 20.0, "pitch": 8.0}
DEFAULT_SLIDERS = {"neatness": 0.6, "size": None, "slant": 0.0, "spacing": 1.0, "jitter": 0.3}


def sliders_with_defaults(sliders):
    out = dict(DEFAULT_SLIDERS)
    out.update({k: v for k, v in (sliders or {}).items() if v is not None})
    return out


def bias_for(sliders):
    return 0.15 + 1.35 * float(np.clip(sliders["neatness"], 0, 1))


def split_tokens(text):
    tokens, depth, buf = [], 0, ""
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == " " and depth == 0:
            if buf:
                tokens.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens


def read_group(tok, i):
    if i >= len(tok) or tok[i] != "{":
        return tok[i:i + 1], i + 1
    depth, j = 0, i
    while j < len(tok):
        if tok[j] == "{":
            depth += 1
        elif tok[j] == "}":
            depth -= 1
            if depth == 0:
                return tok[i + 1:j], j + 1
        j += 1
    return tok[i + 1:], len(tok)


def parse_atom(tok):
    parts, buf, i = [], "", 0
    while i < len(tok):
        if tok.startswith("\\frac{", i):
            if buf:
                parts.append(("text", buf))
                buf = ""
            num, i = read_group(tok, i + 5)
            den, i = read_group(tok, i)
            parts.append(("frac", num, den))
        elif tok[i] in "^_" and i + 1 < len(tok):
            if buf:
                parts.append(("text", buf))
                buf = ""
            group, j = read_group(tok, i + 1)
            parts.append(("sup" if tok[i] == "^" else "sub", group))
            i = j
        else:
            buf += tok[i]
            i += 1
    if buf:
        parts.append(("text", buf))
    return parts


def is_math(tok):
    return "\\frac{" in tok or any(c in tok[:-1] for c in "^_") and any(p[0] != "text" for p in parse_atom(tok))


def transform(strokes, scale, dx, dy):
    return [np.stack([s[:, 0] * scale + dx, s[:, 1] * scale + dy], axis=1) for s in strokes]


def run_of(words, core, scale=1.0):
    strokes, x = [], 0.0
    for i, w in enumerate(words):
        if i > 0:
            x += float(max(w["gap"], 0.35 * core)) * scale
        strokes += transform(w["strokes"], scale, x, 0.0)
        x += w["width"] * scale
    return strokes, x


def compose_atom(writer, tok, bias, rng):
    core = writer.core_units
    strokes, x = [], 0.0
    seed = lambda: int(rng.integers(0, 2 ** 31 - 1))
    for part in parse_atom(tok):
        kind = part[0]
        if kind == "text":
            sub, w = run_of(writer.piece(part[1], bias, seed()), core)
            strokes += transform(sub, 1.0, x, 0.0)
            x += w + 0.1 * core
        elif kind in ("sup", "sub"):
            sub, w = run_of(writer.piece(part[1], bias, seed()), core, 0.62)
            strokes += transform(sub, 1.0, x, -0.85 * core if kind == "sup" else 0.35 * core)
            x += w + 0.15 * core
        else:
            num, wn = run_of(writer.piece(part[1], bias, seed()), core, 0.72)
            den, wd = run_of(writer.piece(part[2], bias, seed()), core, 0.72)
            width = max(wn, wd) + 0.5 * core
            bar_y = -0.55 * core
            strokes += transform(num, 1.0, x + (width - wn) / 2, bar_y - 0.42 * core)
            strokes += transform(den, 1.0, x + (width - wd) / 2, bar_y + 1.35 * core)
            t = np.linspace(0, 1, 14)
            wobble = rng.normal(0, 0.02 * core, len(t)).cumsum() * 0.3 + rng.normal(0, 0.06 * core) * t
            strokes.append(np.stack([x + t * width, bar_y + wobble], axis=1))
            x += width + 0.2 * core
    return {"text": tok, "strokes": strokes, "width": x, "gap": 0.0, "core": core}


def generate_words(writer, text, sliders, seed=None):
    rng = np.random.default_rng(seed)
    bias = bias_for(sliders)
    paragraphs = [split_tokens(p) for p in text.split("\n")]
    plain_runs, slots = [], []
    for pi, tokens in enumerate(paragraphs):
        run = []
        for tok in tokens + [None]:
            if tok is not None and not is_math(tok):
                run.append(tok)
                continue
            if run:
                plain_runs.append(" ".join(run))
                slots.append((pi, "plain", len(plain_runs) - 1))
                run = []
            if tok is not None:
                slots.append((pi, "math", tok))
    generated = writer.words(plain_runs, bias, int(rng.integers(0, 2 ** 31 - 1)))
    out = [[] for _ in paragraphs]
    for pi, kind, ref in slots:
        if kind == "plain":
            out[pi].extend(generated[ref])
        else:
            out[pi].append(compose_atom(writer, ref, bias, rng))
    return out


def place(paragraphs, writer, width_mm, sliders, seed=None, pitch=None, max_lines=None):
    rng = np.random.default_rng(seed)
    size = float(sliders["size"] or writer.size_mm)
    scale = size / writer.core_units
    pitch = pitch or max(LETTER["pitch"], size * 3.0)
    shear = math.tan(math.radians(float(sliders["slant"])))
    jitter = float(sliders["jitter"])
    spacing = float(sliders["spacing"])
    gaps = [w["gap"] for p in paragraphs for w in p if w["gap"] > 0]
    default_gap = float(np.median(gaps)) if gaps else 0.9 * writer.core_units
    habit = getattr(writer, "style", None)
    wobble = (0.6 * habit["baseline"] * size * jitter / 0.3) if habit else 0.07 * size * jitter
    tilt = (0.35 * habit["line_slope"] * jitter / 0.3) if habit else 0.004 * jitter
    indent = (0.4 * habit["margin"] * jitter / 0.3) if habit else 0.0
    widest = 3.4 if habit else 1.6
    lines, strokes_out = 0, []
    for words in paragraphs:
        x = start = abs(rng.normal(0, indent)) if indent else 0.0
        slope = rng.normal(0, tilt)
        offset = rng.normal(0, 0.12 * size * jitter)
        for w in words:
            if not w["strokes"]:
                continue
            gap = w["gap"] if w["gap"] > 0 else default_gap
            unit = getattr(writer, "em", writer.core_units)
            gap = float(np.clip(gap, 0.4 * unit, widest * unit)) * scale * spacing
            width = w["width"] * scale
            if x > start and x + gap + width > width_mm:
                lines += 1
                x = start = abs(rng.normal(0, indent)) if indent else 0.0
                slope = rng.normal(0, tilt)
                offset = rng.normal(0, 0.12 * size * jitter)
            elif x > start:
                x += gap
            dy = offset + slope * x + rng.normal(0, wobble)
            for s in w["strokes"]:
                sx = s[:, 0] * scale - s[:, 1] * scale * shear + x
                sy = s[:, 1] * scale + dy + lines * pitch
                strokes_out.append({"line": lines, "points": np.stack([sx, sy], axis=1)})
            x += width
        lines += 1
    return {"strokes": strokes_out, "lines": lines, "pitch": pitch, "size": size}


def layout_page(writer, text, sliders=None, seed=None, page=None, labels=None):
    sliders = sliders_with_defaults(sliders)
    page = {**LETTER, **(page or {})}
    width = page["width"] - page["margin_left"] - page["margin_right"]
    placed = place(generate_words(writer, text, sliders, seed), writer, width, sliders, seed, page["pitch"])
    per_page = max(1, int((page["height"] - page["margin_top"] - page["margin_bottom"]) / placed["pitch"]))
    pages = [[] for _ in range(max(1, math.ceil(placed["lines"] / per_page)))]
    for s in placed["strokes"]:
        pi = s["line"] // per_page
        pts = s["points"].copy()
        pts[:, 0] += page["margin_left"]
        pts[:, 1] += page["margin_top"] + placed["pitch"] - (pi * per_page) * placed["pitch"]
        pages[pi].append(pts)
    for label in labels or []:
        opts = {**sliders, "size": label.get("size") or sliders["size"], "jitter": 0.0}
        sub = place(generate_words(writer, label["text"], opts, seed), writer, 1e9, opts, seed)
        angle = math.radians(label.get("angle", 0.0))
        ca, sa = math.cos(angle), math.sin(angle)
        for s in sub["strokes"]:
            p = s["points"]
            pages[min(label.get("page", 0), len(pages) - 1)].append(
                np.stack([label["x"] + p[:, 0] * ca - p[:, 1] * sa, label["y"] + p[:, 0] * sa + p[:, 1] * ca], axis=1))
    return {"page": page, "pages": [[p.round(3).tolist() for p in pg] for pg in pages],
            "unsupported": writer.unsupported(strip_markup(text))}


def strip_markup(text):
    return text.replace("\\frac", "").replace("{", "").replace("}", "").replace("^", "").replace("_", "")


def fit_box(writer, text, box, sliders=None, seed=None, valign="bottom"):
    sliders = sliders_with_defaults(sliders)
    words = generate_words(writer, text, sliders, seed)
    size = float(sliders["size"] or writer.size_mm)
    pad = min(1.8 if valign == "middle" else 1.0, box["w"] * 0.05)
    while True:
        opts = {**sliders, "size": size}
        pitch = size * 2.9
        placed = place(words, writer, box["w"] - 2 * pad, opts, seed, pitch)
        pts = np.concatenate([s["points"] for s in placed["strokes"]]) if placed["strokes"] else np.zeros((1, 2))
        height = (placed["lines"] - 1) * pitch + size * 2.4
        if (height <= box["h"] and pts[:, 0].max() <= box["w"] - 2 * pad) or size <= 1.1:
            break
        size *= 0.92
    top = box["y"] + (box["h"] - height) / 2 if valign == "middle" else box["y"] + box["h"] - height - 0.1 * size
    base = max(top, box["y"]) + size * 1.75
    return {"size": size, "strokes": [np.stack([s["points"][:, 0] + box["x"] + pad, s["points"][:, 1] + base], axis=1).round(3).tolist()
                                       for s in placed["strokes"]]}
