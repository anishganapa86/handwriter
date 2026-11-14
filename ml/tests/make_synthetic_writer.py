import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw.generate import Writer

ROOT = Path(__file__).resolve().parents[2]


def main(out_dir, limit=None):
    script = json.loads((ROOT / "app" / "src" / "script.json").read_text(encoding="utf-8"))["lines"]
    if limit:
        script = script[: int(limit)]
    writer = Writer()
    mm = 2.4 / writer.core_units
    lines = []
    for n, entry in enumerate(script):
        line = writer.line(entry["text"], bias=0.9, seed=100 + n)
        t, strokes = 0.0, []
        for s in line["strokes"]:
            pts = []
            for p in s:
                pts.append([round(float(p[0] * mm - 0.22 * p[1] * mm) + 12, 3), round(float(p[1] * mm) + 20, 3), round(t, 1), 0.5, 0, 0])
                t += 8.0
            t += 150.0
            strokes.append({"points": pts})
        lines.append({"id": entry["id"], "promptId": entry["id"], "text": entry["text"], "holdout": entry["holdout"], "strokes": strokes})
        print(n, line["ok"], entry["text"][:40])
    out = Path(out_dir) / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    session = {"schema": "hw-capture/1", "sessionId": "synthetic", "createdAt": "2026-01-01T00:00:00Z",
               "device": {"pointerType": "synthetic", "pxPerMm": 1}, "lines": lines}
    (out / "synthetic.json").write_text(json.dumps(session), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
