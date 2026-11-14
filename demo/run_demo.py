import base64
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "hardware"))

from hw import plotter
from hw.glyphs import GlyphWriter
from hw.layout import layout_page

TEXT = """Notes for Thursday

The market by the canal opens at seven, so I will leave the house a little before six and walk the long way round past the old mill.

Things to bring: a jar for the honey, two cloth bags, exact change and the list Maya gave me. She wants 3 lemons, 12 eggs and a loaf of the dark rye bread.

A quick sum for the paving job. The yard is a square of side x, so its area is x^2. We only pave \\frac{3}{4} of it, which leaves \\frac{1}{4} for the herb bed. If x = 6 then 36 - 27 = 9 square metres stay green.

Remember to ring Jon about the ladder, pay the window cleaner, and ask whether the choir still meets on the first Friday of every month.

On the way back, stop at the library before it shuts at 5 and return the two books on bridges. The fine is 40 cents a day, so 3 days late costs 1.20 in total.

For the weekend: fix the gate latch, plant the beans in rows 30 cm apart, and write to Aunt Zelda to thank her for the quilt.

None of these sentences were in the pages the letters came from. Every word here is put together from single letters and joined pairs, then spaced, slanted and wobbled the way the measured habits say."""

FALU_PENCIL = dict(holder_front=50, holder_left=43, tip_depth=17, z_draw=15.5, z_lift=20, draw_speed=1200)
INK = (70, 62, 58)


def to_px(points, k, dx=0.0, dy=0.0):
    return np.round((np.asarray(points) + [dx, dy]) * k * 16).astype(np.int32)


def draw(img, strokes, k, colour=INK, width=2, dx=0.0, dy=0.0):
    for s in strokes:
        s = np.asarray(s, dtype=np.float64)
        if len(s) == 1:
            cv2.circle(img, tuple(int(v) for v in to_px(s, k, dx, dy)[0]), int(width * 9), colour, -1, cv2.LINE_AA, 4)
        else:
            cv2.polylines(img, [to_px(s, k, dx, dy)], False, colour, width, cv2.LINE_AA, 4)


def paper(page, k):
    img = np.full((int(page["height"] * k), int(page["width"] * k), 3), (243, 250, 252), np.uint8)
    y = page["margin_top"] + page["pitch"]
    while y < page["height"] - 8:
        cv2.line(img, (0, int(y * k)), (img.shape[1], int(y * k)), (232, 214, 196), 1, cv2.LINE_AA)
        y += page["pitch"]
    return img


def save(name, img):
    cv2.imwrite(str(OUT / name), img)
    return name


def animate(strokes, page, name, k=2.4, frames=70):
    canvas = paper(page, k)
    total = sum(len(s) for s in strokes)
    step, used, shots = total / frames, 0, []
    for s in strokes:
        draw(canvas, [s], k, width=1)
        used += len(s)
        while len(shots) < used / step:
            shots.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE, colors=32))
    shots += [shots[-1]] * 12
    shots[0].save(OUT / name, save_all=True, append_images=shots[1:], duration=70, loop=0, optimize=True)
    return name


def read_back(gcode, pen):
    x = y = None
    down, started, strokes, current = False, False, [], []
    for line in gcode.splitlines():
        if line.startswith("; PEN_FITTED"):
            started = True
        code = line.split(";")[0].split()
        if not started or not code or code[0] not in ("G0", "G1", "G4"):
            continue
        args = {a[0]: float(a[1:]) for a in code[1:]}
        if code[0] == "G4":
            continue
        if "Z" in args:
            now_down = code[0] == "G1"
            if now_down and not down:
                current = [[x + pen.offset_x, y + pen.offset_y]]
            if down and not now_down and current:
                strokes.append(np.array(current))
                current = []
            down = now_down
            continue
        x, y = args.get("X", x), args.get("Y", y)
        if down:
            current.append([x + pen.offset_x, y + pen.offset_y])
    return strokes


def bed_view(drawn, pen, page, left, turned, k=3.0):
    pad_front = max(page["height"] - pen.back_stop_y, 0) + 6
    h = int((plotter.BED + pad_front + 6) * k)
    img = np.full((h, int((plotter.BED + 12) * k), 3), 236, np.uint8)
    at = lambda px, py: (int((px + 6) * k), int((plotter.BED + 6 - py) * k))
    cv2.rectangle(img, at(0, plotter.BED), at(plotter.BED, 0), (205, 205, 205), -1)
    cv2.rectangle(img, at(0, plotter.BED), at(plotter.BED, 0), (120, 120, 120), 1)
    cv2.rectangle(img, at(left, pen.back_stop_y), at(left + page["width"], pen.back_stop_y - page["height"]), (243, 250, 252), -1)
    cv2.rectangle(img, at(left, pen.back_stop_y), at(left + page["width"], pen.back_stop_y - page["height"]), (150, 150, 150), 1)
    x0, y0, x1, y1 = pen.reach()
    cv2.rectangle(img, at(x0, y1), at(x1, y0), (60, 140, 60), 1)
    flipped = [np.stack([s[:, 0] + 6, plotter.BED + 6 - s[:, 1]], axis=1) for s in drawn]
    last = None
    for s in flipped:
        if last is not None:
            cv2.line(img, tuple(int(v * k) for v in last), tuple(int(v * k) for v in s[0]), (215, 190, 150), 1, cv2.LINE_AA)
        last = s[-1]
    draw(img, flipped, k, width=1)
    label = "part turned upside down" if turned else "paper the right way up"
    cv2.putText(img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "green box = where the pen may go", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 120, 60), 1, cv2.LINE_AA)
    return img


def to_page(drawn, page, left, back, turned):
    if turned:
        return [np.stack([left + page["width"] - s[:, 0], page["height"] - (back - s[:, 1])], axis=1) for s in drawn]
    return [np.stack([s[:, 0] - left, back - s[:, 1]], axis=1) for s in drawn]


def worst_gap(original, rebuilt):
    cloud = np.concatenate(rebuilt)
    worst = 0.0
    for s in original:
        for p in s[:: max(1, len(s) // 6)]:
            worst = max(worst, float(np.sqrt(((cloud - p) ** 2).sum(1).min())))
    return worst


def embed(name):
    kind = {"png": "image/png", "gif": "image/gif"}[name.rsplit(".", 1)[1]]
    return f"data:{kind};base64," + base64.b64encode((OUT / name).read_bytes()).decode()


def page_html(report, images):
    steps = [
        ("1. Type anything", f"<pre>{report['text']}</pre><p>{report['characters']} characters, none of it from the pages the letters were traced from. "
         "<code>x^2</code> and <code>\\frac{3}{4}</code> are layout markup.</p>", None),
        ("2. It is written in the handwriting", f"<p>{report['strokes']} pen strokes on a letter page, built from {report['library_units']} letters and joined groups "
         f"({report['library_samples']} samples) and the habits measured from real pages. Characters with no sample: {report['unsupported'] or 'none'}.</p>", images["page"]),
        ("3. Stroke by stroke", "<p>The same strokes in the order the pen will make them.</p>", images["gif"]),
        ("4. It becomes a file for the Bambu Lab P1S", "<p>" + " ".join(
            f"Part {p['part']}: {p['strokes']} strokes, about {p['minutes']} min, {p['gcode_lines']} lines of G-code, safety faults: {len(p['faults'])}." for p in report["parts"])
         + f" {report['reach']}</p>", images["bed"]),
        ("5. Read back from the G-code", f"<p>The picture is redrawn only from the printer commands. Furthest any point lands from where it was meant to: "
         f"{report['readback_worst_mm']:.2f} mm. The pen goes down {report['readback_strokes']} times for {report['strokes']} strokes, because strokes that meet are drawn without lifting.</p>", images["readback"]),
        ("6. The pencil holder", f"<p>The pencil guide from this project in Falu's head mount with his {report['holder']['spring_module']} spring module. "
         f"Overlap with the mount's clip rings: {report['holder']['guide_in_mount_mm3']} mm&sup3; (snug). Pencil against guide and spring: "
         f"{report['holder']['pencil_in_guide_mm3']} and {report['holder']['pencil_in_spring_mm3']} mm&sup3; (free).</p>", images["holder"]),
    ]
    body = "".join(f"<section><h2>{t}</h2>{text}{f'<img src={chr(34)}{img}{chr(34)}>' if img else ''}</section>" for t, text, img in steps)
    credits = ("<section><h2>Credits</h2><ul>"
               "<li>Head mount and spring module: <b>Falu Mendy Vergara</b>, Universal Modular Tool System, makerworld.com/en/models/2029113</li>"
               "<li>P1S setup guide for that system: rebelthor/bambu-lab-pen-plotter (MIT)</li>"
               "<li>P1S plotting approach and printer profile: gtls64/p1s-pen-plotter (MIT)</li>"
               "<li>Base handwriting network: X-rayLaser/pytorch-handwriting-synthesis-toolkit (MIT); photo tracing: Google InkSight (Apache-2.0)</li>"
               "</ul></section>")
    limits = ("<section><h2>What this demo does not show</h2><p>Nothing here has run on a real printer: the files pass a simulated safety check only, "
              "the pen offsets are uncalibrated (0, 0) and the holder has not been printed.</p></section>")
    style = ("body{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px 16px;color:#14213d;background:#fafbfd}"
             "h1{margin:0 0 4px}h2{margin:32px 0 8px;font-size:20px}img{max-width:100%;border:1px solid #d9dee7;border-radius:6px;background:#fff}"
             "pre{white-space:pre-wrap;background:#f3f5f9;padding:12px;border-radius:8px;font-size:13px}code{background:#f3f5f9;padding:1px 4px;border-radius:4px}")
    return (f"<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Handwriter demo</title><style>{style}</style></head><body><h1>Handwriter, end to end</h1>"
            f"<p>Made by <code>python demo/run_demo.py</code> on {report['made']}.</p>{body}{limits}{credits}</body></html>")


def main():
    OUT.mkdir(exist_ok=True)
    for stale in OUT.glob("*.gcode.3mf"):
        stale.unlink()
    data = ROOT / "data"
    writer = GlyphWriter(data / "glyphs.json", data)
    writer.use_words = False
    writer.joins = 1.0
    result = layout_page(writer, TEXT, {"neatness": 0.6, "jitter": 0.3}, seed=7)
    page = result["page"]
    strokes = [np.asarray(s, dtype=np.float64) for s in result["pages"][0]]

    sheet = paper(page, 6.0)
    draw(sheet, strokes, 6.0)
    images = {"page": save("1_page.png", sheet), "gif": animate(strokes, page, "2_writing.gif")}

    pen = plotter.pen_from({**asdict(plotter.Pen()), **FALU_PENCIL})
    left = plotter.paper_left(pen, page["width"])
    parts, problems = plotter.place_page(strokes, page["width"], page["height"], pen)
    if problems:
        raise SystemExit("\n".join(problems))
    dry = plotter.pen_from({**asdict(pen), "z_draw": pen.z_draw + 5, "z_lift": min(pen.z_lift + 5, pen.set_z() - 1)})
    views, rebuilt, summary = [], [], []
    for i, part in enumerate(parts, 1):
        job, issues = plotter.build_gcode(part["strokes"], pen, f"demo part {i}", dry_run=True)
        if job is None:
            raise SystemExit("\n".join(issues))
        faults, _ = plotter.simulate(job["gcode"], dry)
        name = f"demo-page-dryrun-part{i}of{len(parts)}.gcode.3mf"
        (OUT / name).write_bytes(plotter.package(job, part["strokes"], f"demo part {i}"))
        drawn = read_back(job["gcode"], pen)
        views.append(bed_view(drawn, pen, page, left, part["turned"]))
        rebuilt += to_page(drawn, page, left, pen.back_stop_y, part["turned"])
        summary.append({"part": i, "file": name, "turned": part["turned"], "strokes": job["strokes"], "minutes": int(round(job["seconds"] / 60)),
                        "gcode_lines": len(job["gcode"].splitlines()), "faults": faults})
    height = max(v.shape[0] for v in views)
    images["bed"] = save("3_on_the_plate.png", np.hstack([cv2.copyMakeBorder(v, 0, height - v.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(236, 236, 236)) for v in views]))

    back = paper(page, 6.0)
    draw(back, rebuilt, 6.0, colour=(150, 70, 30))
    images["readback"] = save("4_read_back_from_gcode.png", back)

    import preview
    holder = preview.assembly(ROOT / "hardware" / "falu")
    Path(OUT / "5_pencil_holder.png").write_bytes((ROOT / "hardware" / "out" / "pencil_holder_assembly.png").read_bytes())
    images["holder"] = "5_pencil_holder.png"

    x0, y0, x1, y1 = pen.reach()
    report = {
        "made": time.strftime("%Y-%m-%d %H:%M"),
        "text": TEXT, "characters": len(TEXT), "strokes": len(strokes), "unsupported": result["unsupported"],
        "library_units": len(writer.glyphs), "library_samples": sum(len(v) for v in writer.glyphs.values()),
        "pen": asdict(pen), "parts": summary,
        "reach": f"With Falu's clearances the pen reaches {x1 - x0:.0f} mm across and {y1 - y0:.0f} mm front to back, so a letter page takes {len(parts)} pass{'es' if len(parts) > 1 else ''}.",
        "readback_strokes": len(rebuilt), "readback_worst_mm": worst_gap(strokes, rebuilt), "holder": holder,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (OUT / "index.html").write_text(page_html(report, {k: embed(v) for k, v in images.items()}), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("text", "pen")}, indent=1))
    failed = [p for p in summary if p["faults"]] or report["unsupported"] or report["readback_worst_mm"] > 0.2
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
