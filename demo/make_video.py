import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import printer_scene as ps
import run_demo as demo
from hw import plotter
from hw.glyphs import GlyphWriter
from hw.layout import layout_page

W, H, FPS = 1280, 720, 30
BG = (253, 251, 250)
DARK = (61, 33, 20)
MUTED = (138, 116, 104)
ACCENT = (199, 86, 36)
INK = (70, 62, 58)
FONTS = {"regular": "segoeui.ttf", "bold": "segoeuib.ttf", "mono": "consola.ttf"}


def font(kind, size):
    for folder in ("C:/Windows/Fonts/", "/System/Library/Fonts/Supplemental/", "/usr/share/fonts/truetype/dejavu/"):
        for name in (FONTS[kind], "Arial.ttf", "DejaVuSans.ttf"):
            if Path(folder + name).exists():
                return ImageFont.truetype(folder + name, size)
    return ImageFont.load_default()


def wrap(text, face, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            if face.getlength(trial) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def write(img, xy, text, size=22, kind="regular", colour=DARK, width=None, gap=1.35):
    face = font(kind, size)
    pil = Image.fromarray(img)
    pen = ImageDraw.Draw(pil)
    x, y = xy
    for line in wrap(text, face, width) if width else text.split("\n"):
        pen.text((x, y), line, font=face, fill=colour)
        y += int(size * gap)
    img[:] = np.asarray(pil)
    return y


def blank():
    return np.full((H, W, 3), BG, np.uint8)


def sub16(points):
    return np.round(np.asarray(points) * 16).astype(np.int32)


def trace(canvas, strokes, frames, colour=INK, width=1, travel=None):
    total = sum(max(len(s) - 1, 1) for s in strokes)
    per = total / max(frames, 1)
    budget, last = per, None
    for s in strokes:
        if travel is not None and last is not None:
            cv2.line(canvas, tuple(sub16(last)), tuple(sub16(s[0])), travel, 1, cv2.LINE_AA, 4)
        if len(s) == 1:
            cv2.circle(canvas, tuple(sub16(s[0])), 20, colour, -1, cv2.LINE_AA, 4)
            budget -= 1
            if budget <= 0:
                yield s[0]
                budget += per
        pos = 0
        while pos < len(s) - 1:
            n = int(min(max(budget, 1), len(s) - 1 - pos))
            cv2.polylines(canvas, [sub16(s[pos:pos + n + 1])], False, colour, width, cv2.LINE_AA, 4)
            pos += n
            budget -= n
            if budget <= 0:
                yield s[pos]
                budget += per
        last = s[-1]
    yield last


class Encoder:
    def __init__(self, path):
        self.count = 0
        try:
            import imageio_ffmpeg
            self.cv = None
            self.proc = subprocess.Popen(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        except ImportError:
            self.proc = None
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def add(self, frame, seconds=None):
        for _ in range(1 if seconds is None else int(seconds * FPS)):
            if self.proc:
                self.proc.stdin.write(frame.tobytes())
            else:
                self.cv.write(frame)
            self.count += 1

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()
        return self.count / FPS


def title_card(out):
    f = blank()
    write(f, (90, 210), "Handwriter", 84, "bold")
    write(f, (94, 330), "Type anything. It is written in your own handwriting, then turned into a job for a Bambu Lab P1S holding a pencil.", 30, width=1050)
    write(f, (94, 470), "Every frame is drawn by demo/make_video.py from the real pipeline. The printer is a simplified 3D model moved by the real plotter file. Nothing has run on a real printer yet.", 22, colour=MUTED, width=1050)
    out.add(f, 3.5)


def typing_and_writing(out, strokes, page):
    k = 690 / page["height"]
    px, py = 720, 15
    sheet = demo.paper(page, k)
    base = blank()
    write(base, (40, 28), "1. Type anything", 30, "bold")
    face = font("mono", 15)
    lines = wrap(demo.TEXT, face, 600)
    base[py:py + sheet.shape[0], px:px + sheet.shape[1]] = sheet
    cv2.rectangle(base, (px, py), (px + sheet.shape[1], py + sheet.shape[0]), (200, 200, 200), 1)
    total = sum(len(line) for line in lines)
    frames = int(5 * FPS)
    typed = base
    for i in range(frames + 1):
        left = int(total * i / frames)
        f = base.copy()
        pil = Image.fromarray(f)
        pen = ImageDraw.Draw(pil)
        y = 86
        for line in lines:
            shown = line[:left]
            left = max(left - len(line), 0)
            pen.text((40, y), shown, font=face, fill=DARK)
            y += 21
        typed = np.asarray(pil).copy()
        out.add(typed)
    write(typed, (40, 640), "2. It is written with your letters, joins and measured habits", 22, "bold", ACCENT, width=620)
    out.add(typed, 0.7)
    canvas = typed.copy()
    scaled = [np.asarray(s) * k + [px, py] for s in strokes]
    for _ in trace(canvas, scaled, int(10 * FPS)):
        out.add(canvas)
    write(canvas, (40, 682), f"{len(strokes)} pen strokes. None of these sentences were in the source pages.", 18, colour=MUTED)
    out.add(canvas, 2.0)


def caption(frame, text, speed=None):
    band = frame[640:704, 24:1256].astype(np.float32)
    frame[640:704, 24:1256] = (band * 0.14 + 255 * 0.86).astype(np.uint8)
    write(frame, (44, 656), text, 22, "bold", width=950)
    write(frame, (1010, 646), "3D simulation from the G-code", 15, colour=MUTED)
    if speed:
        write(frame, (1010, 668), f"{speed:.0f}x speed" if speed >= 1.5 else "real time", 20, "bold", ACCENT)
    return frame


def state_at(rows, t):
    i = int(np.clip(np.searchsorted(rows[:, 0], t), 1, len(rows) - 1))
    t0, t1 = rows[i - 1, 0], rows[i, 0]
    f = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0, 1))
    return rows[i - 1, 1:] + (rows[i, 1:] - rows[i - 1, 1:]) * f


def printer_sequence(out, parts, jobs, pen, page):
    p = ps.Printer(pen, page, (W, H))
    s = p.state
    cx, cy = plotter.clip_xy(pen)
    fit_tip = (cx + pen.offset_x, cy + pen.offset_y)

    def hold(seconds, view, text, door=False, moves=None):
        n = max(1, int(seconds * FPS))
        for i in range(n):
            f = ps.ease((i + 1) / n)
            for key, (a, b) in (moves or {}).items():
                s[key] = a + (b - a) * f
            p.show("door", door)
            p.look(*ps.shot(view, p, (i + 1) / n, fit_tip if view == "fit" else None))
            out.add(caption(p.frame(), text))

    def plot(job, turned, plan):
        rows, ink = ps.timeline(job["gcode"], pen)
        done, follow = 0, None
        for seconds, t0, t1, view, text in plan:
            t0, t1 = t0 * rows[-1, 0] if t0 <= 1 else t0, t1 * rows[-1, 0] if t1 <= 1 else t1
            n = int(seconds * FPS)
            for i in range(n):
                now = t0 + (t1 - t0) * (i + 1) / n
                k = int(np.searchsorted(ink[:, 0], now, side="right"))
                if k > done:
                    p.ink(ink[done:k], turned)
                    p.refresh_paper()
                    done = k
                s["x"], s["y"], s["z"] = state_at(rows, now)
                tip = np.array([s["x"] + pen.offset_x, s["y"] + pen.offset_y])
                follow = tip if follow is None or view != "close" else follow + (tip - follow) * 0.08
                p.show("door", view == "outside")
                p.look(*ps.shot(view, p, (i + 1) / n, follow if view == "close" else None))
                out.add(caption(p.frame(), text, (t1 - t0) / seconds))
        if done < len(ink):
            p.ink(ink[done:], turned)
            p.refresh_paper()

    park = plotter.park_xy(pen)
    s.update(x=park[0], y=park[1], z=pen.set_z())
    hold(4.0, "outside", "Simplified model of the Bambu Lab P1S. Top glass off, paper taped down, heaters off.", door=True)
    hold(1.4, "fit", "The printer homes with nothing fitted, comes to the front and pauses.", moves={"x": (park[0], cx), "y": (park[1], cy)})
    p.show("tool", True)
    hold(1.8, "fit", "Clip on Falu's head mount with the pencil guide and spring module.")
    p.show("pencil", True)
    p.show("block", True)
    hold(1.4, "fit", "Rest the pencil tip on a 10 mm block and tighten it.", moves={"pencil_up": (70.0, 0.0)})
    hold(0.8, "fit", "Rest the pencil tip on a 10 mm block and tighten it.")
    p.show("block", False)
    s["fitted"] = True
    hold(1.2, "fit", "Take the block away and press Resume.", moves={"z": (pen.set_z(), pen.z_lift)})
    first = "Pass 1 of 2: the bed drops to lift the pencil between strokes." if len(parts) > 1 else "The bed drops to lift the pencil between strokes."
    plot(jobs[0], parts[0]["turned"], [
        (8.0, 0.0, 26.0, "close", first),
        (8.0, 26.0, 0.55, "inside", f"Inside, door open for the camera: {jobs[0]['strokes']} strokes, about {int(round(jobs[0]['seconds'] / 60))} minutes in real life."),
        (7.0, 0.55, 1.0, "outside", "From outside, through the door glass."),
    ])
    if len(parts) > 1:
        hold(1.0, "top", "Pass 1 done. The bed lowers and the head parks at the back.", moves={"x": (s["x"], park[0]), "y": (s["y"], park[1]), "z": (s["z"], pen.z_lift + plotter.END_LIFT)})
        hold(0.5, "top", "Turn the paper upside down against the same marks.", moves={"paper_lift": (0.0, 14.0)})
        hold(1.6, "top", "Turn the paper upside down against the same marks.", moves={"turn": (0.0, 180.0)})
        hold(0.5, "top", "Turn the paper upside down against the same marks.", moves={"paper_lift": (14.0, 0.0)})
        plot(jobs[1], parts[1]["turned"], [
            (5.0, 0.0, 18.0, "close", "Pass 2 of 2 writes the rest of the page from the other end."),
            (5.0, 18.0, 0.5, "side", f"Pass 2: {jobs[1]['strokes']} strokes, about {int(round(jobs[1]['seconds'] / 60))} minutes in real life."),
            (5.0, 0.5, 1.0, "top", "From above, through the open top."),
        ])
    hold(1.2, "top", "Done: the bed lowers and the head parks.", moves={"x": (s["x"], park[0]), "y": (s["y"], park[1]), "z": (s["z"], pen.z_lift + plotter.END_LIFT)})
    hold(3.5, "outside", "The finished page. This is a simulation: nothing has run on the real printer yet.", door=True)
    p.pl.close()


def read_back_scene(out, strokes, rebuilt, page, worst):
    k = 590 / page["height"]
    f = blank()
    write(f, (40, 24), "4. Proof the files say what the page says", 30, "bold")
    for x, ink, colour, label in ((150, strokes, INK, "The page as written"), (700, rebuilt, (150, 70, 30), "Redrawn from the printer commands alone")):
        sheet = demo.paper(page, k)
        demo.draw(sheet, ink, k, colour=colour, width=1)
        f[80:80 + sheet.shape[0], x:x + sheet.shape[1]] = sheet
        cv2.rectangle(f, (x, 80), (x + sheet.shape[1], 80 + sheet.shape[0]), (200, 200, 200), 1)
        write(f, (x, 678), label, 20, "bold", colour if colour != INK else DARK)
    write(f, (1180, 300), f"{worst:.2f}", 30, "bold", ACCENT)
    write(f, (1180, 340), "mm worst\ndrift", 17, colour=MUTED)
    out.add(f, 4.5)


def holder_scene(out):
    f = blank()
    write(f, (40, 24), "5. The pencil holder", 30, "bold")
    picture = cv2.imread(str(demo.ROOT / "hardware" / "out" / "pencil_holder_assembly.png"))
    k = 1200 / picture.shape[1]
    picture = cv2.resize(picture, (1200, int(picture.shape[0] * k)), interpolation=cv2.INTER_AREA)
    f[90:90 + picture.shape[0], 40:1240] = picture
    y = write(f, (40, 110 + picture.shape[0]), "Head mount and spring module: Falu Mendy Vergara's Universal Modular Tool System (MakerWorld model 2029113).", 22, "bold", width=1200)
    y = write(f, (40, y + 6), "This project adds a pencil guide that clips into his mount: same outside as his guide, bore sized to the pencil. Checked on screen against his files; not printed yet.", 20, colour=MUTED, width=1200)
    out.add(f, 5.5)


def end_card(out):
    f = blank()
    y = write(f, (90, 70), "Where it stands", 44, "bold")
    for line in ("Works: exact typed text in the handwriting, plotter files that pass the safety simulation, pencil guide that fits Falu's mount on screen.",
                 "Not done: nothing has run on the real P1S, the holder is not printed, pen offsets are not calibrated.",
                 "Not good enough yet: some words are hard to read, and the blind test against real writing has not been run."):
        y = write(f, (94, y + 16), line, 24, width=1090)
    y = write(f, (90, y + 40), "Credits", 30, "bold")
    for line in ("Pencil holder hardware: Falu Mendy Vergara, Universal Modular Tool System (makerworld.com/en/models/2029113)",
                 "P1S setup guide: rebelthor/bambu-lab-pen-plotter (MIT)",
                 "P1S plotting approach and profile: gtls64/p1s-pen-plotter (MIT)",
                 "Base handwriting network: X-rayLaser/pytorch-handwriting-synthesis-toolkit (MIT)",
                 "Photo tracing: Google InkSight (Apache-2.0)"):
        y = write(f, (94, y + 8), line, 20, colour=MUTED, width=1090)
    out.add(f, 6.0)


def main():
    demo.OUT.mkdir(exist_ok=True)
    data = demo.ROOT / "data"
    writer = GlyphWriter(data / "glyphs.json", data)
    writer.use_words = False
    writer.joins = 1.0
    result = layout_page(writer, demo.TEXT, {"neatness": 0.6, "jitter": 0.3}, seed=7)
    page = result["page"]
    strokes = [np.asarray(s, dtype=np.float64) for s in result["pages"][0]]
    pen = plotter.pen_from({**asdict(plotter.Pen()), **demo.FALU_PENCIL, **demo.ESTIMATED_PLACING})
    left = plotter.paper_left(pen, page["width"])
    parts, problems = plotter.place_page(strokes, page["width"], page["height"], pen)
    if problems:
        raise SystemExit("\n".join(problems))
    jobs, rebuilt = [], []
    for i, part in enumerate(parts, 1):
        job, issues = plotter.build_gcode(part["strokes"], pen, f"demo part {i}")
        if job is None:
            raise SystemExit("\n".join(issues))
        job["faults"] = plotter.simulate(job["gcode"], pen)[0]
        jobs.append(job)
        rebuilt += demo.to_page(demo.read_back(job["gcode"], pen), page, left, pen.back_stop_y, part["turned"])
    path = demo.OUT / "demo.mp4"
    out = Encoder(path)
    title_card(out)
    typing_and_writing(out, strokes, page)
    printer_sequence(out, parts, jobs, pen, page)
    read_back_scene(out, strokes, rebuilt, page, demo.worst_gap(strokes, rebuilt))
    holder_scene(out)
    end_card(out)
    seconds = out.close()
    print(f"{path} {seconds:.0f} s {path.stat().st_size / 1e6:.1f} MB h264={out.proc is not None}")


if __name__ == "__main__":
    main()
