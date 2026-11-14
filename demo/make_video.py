import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    write(f, (94, 470), "Every frame of this video is drawn by demo/make_video.py from the real pipeline. Nothing has run on a real printer yet.", 22, colour=MUTED, width=1050)
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
    for _ in trace(canvas, scaled, int(15 * FPS)):
        out.add(canvas)
    write(canvas, (40, 682), f"{len(strokes)} pen strokes. None of these sentences were in the source pages.", 18, colour=MUTED)
    out.add(canvas, 2.0)


def plate(out, parts, jobs, pen, page, left):
    k = 2.3
    ox, oy = 30, 14
    at = lambda p: np.stack([ox + (np.asarray(p)[..., 0] + 6) * k, oy + (plotter.BED + 6 - np.asarray(p)[..., 1]) * k], axis=-1)
    back = pen.back_stop_y
    corner = lambda x, y: tuple(int(v) for v in at([x, y]))
    done_before = []
    for i, (part, job) in enumerate(zip(parts, jobs), 1):
        base = blank()
        cv2.rectangle(base, corner(0, plotter.BED), corner(plotter.BED, 0), (205, 205, 205), -1)
        cv2.rectangle(base, corner(0, plotter.BED), corner(plotter.BED, 0), (120, 120, 120), 1)
        cv2.rectangle(base, corner(left, back), corner(left + page["width"], back - page["height"]), (252, 250, 243), -1)
        cv2.rectangle(base, corner(left, back), corner(left + page["width"], back - page["height"]), (150, 150, 150), 1)
        x0, y0, x1, y1 = pen.reach()
        cv2.rectangle(base, corner(x0, y1), corner(x1, y0), (60, 140, 60), 1)
        for s in done_before:
            turned = np.stack([2 * left + page["width"] - s[:, 0], 2 * back - page["height"] - s[:, 1]], axis=1)
            cv2.polylines(base, [sub16(at(turned))], False, (170, 165, 160), 1, cv2.LINE_AA, 4)
        tx = 760
        y = write(base, (tx, 28), "3. On the P1S plate", 30, "bold")
        y = write(base, (tx, y + 8), f"Pass {i} of {len(parts)}: " + ("paper turned upside down against the same back stop" if part["turned"] else "paper the right way up"), 22, "bold", ACCENT, width=480)
        facts = [f"{job['strokes']} strokes, about {int(round(job['seconds'] / 60))} minutes",
                 f"{len(job['gcode'].splitlines()):,} lines of G-code in a .gcode.3mf for Bambu Studio",
                 "Heaters off, nothing extruded",
                 "Homes first, then pauses for the holder and pencil",
                 "Pencil lifts between strokes; dots get a short dwell",
                 f"Safety simulation: {'passed' if not job['faults'] else job['faults'][0]}"]
        for fact in facts:
            y = write(base, (tx, y + 10), "•  " + fact, 20, width=470)
        write(base, (tx, 560), "Grey square: the 256 mm plate. A letter page is longer, so it overhangs the front. Green box: where the pencil may go with this holder. Blue lines: moves with the pencil lifted.", 17, colour=MUTED, width=470)
        write(base, (tx, 668), "Simulated from the G-code. Pen offsets are not calibrated yet.", 17, colour=MUTED, width=470)
        drawn = [at(s) for s in demo.read_back(job["gcode"], pen)]
        canvas = base.copy()
        seconds = 6 + 6 * len(drawn) / max(sum(len(demo.read_back(j["gcode"], pen)) for j in jobs), 1)
        for tip in trace(canvas, drawn, int(seconds * FPS), travel=(235, 200, 160)):
            f = canvas.copy()
            cv2.circle(f, tuple(sub16(tip)), 90, (40, 40, 220), -1, cv2.LINE_AA, 4)
            out.add(f)
        out.add(canvas, 1.2)
        done_before += [np.asarray(s) for s in part["strokes"]]
        if i < len(parts):
            card = blank()
            write(card, (90, 270), "Turn the paper upside down", 60, "bold")
            write(card, (94, 370), "Same back stop, same tape marks. The second file writes the rest of the page from the other end.", 28, width=1050)
            out.add(card, 2.5)


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
    pen = plotter.pen_from({**asdict(plotter.Pen()), **demo.FALU_PENCIL})
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
    plate(out, parts, jobs, pen, page, left)
    read_back_scene(out, strokes, rebuilt, page, demo.worst_gap(strokes, rebuilt))
    holder_scene(out)
    end_card(out)
    seconds = out.close()
    print(f"{path} {seconds:.0f} s {path.stat().st_size / 1e6:.1f} MB h264={out.proc is not None}")


if __name__ == "__main__":
    main()
