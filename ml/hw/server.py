import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataclasses import asdict

from . import copypage, photo, plotter, template, worksheet
from .dataset import load_samples, split
from .generate import Writer
from .glyphs import GlyphWriter, load_library, save_glyph, summary
from .layout import LETTER, fit_box, layout_page, sliders_with_defaults
from . import style as style_fit
from .strokes import bounds, capture_line_to_strokes, clean, core_metrics, write_iam_xml
from .train import champion_path, registry

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("HW_DATA", ROOT / "data"))
DIST = ROOT / "app" / "dist"
RETRAIN_EVERY = 50
GLYPHS = DATA / "glyphs.json"
PLOTTER = DATA / "plotter.json"
(DATA / "plots").mkdir(parents=True, exist_ok=True)
for name in ("sessions", "photo_words", "photo_crops", "photos", "worksheets", "reports", "checkpoints", "copies"):
    (DATA / name).mkdir(parents=True, exist_ok=True)

app = FastAPI()
state = {"writer": None, "writer_path": None, "job": None, "photo_jobs": {}, "glyphs": None, "glyphs_stamp": None}
lock = threading.Lock()


def glyph_writer():
    stamp = GLYPHS.stat().st_mtime if GLYPHS.exists() else None
    with lock:
        if state["glyphs"] is None or state["glyphs_stamp"] != stamp:
            accepted = sum(1 for p in (DATA / "photo_words").glob("*.json") if '"status": "accepted"' in p.read_text(encoding="utf-8"))
            known = style_fit.load(DATA)
            if not (DATA / "style.json").exists() or abs(known.get("accepted", 0) - accepted) >= 25:
                fitted = style_fit.fit(DATA)
                fitted["accepted"] = accepted
                (DATA / "style.json").write_text(json.dumps(fitted, indent=1), encoding="utf-8")
            state["glyphs"] = GlyphWriter(GLYPHS, DATA)
            state["glyphs_stamp"] = stamp
        return state["glyphs"]


def writer(engine="model"):
    if engine == "glyphs":
        found = glyph_writer()
        if not found.glyphs:
            raise HTTPException(409, "no glyphs captured yet")
        return found
    path = champion_path(DATA)
    with lock:
        if state["writer_path"] != path:
            state["writer"] = Writer(path)
            state["writer_path"] = path
        return state["writer"]


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value or ""):
        raise HTTPException(400, "bad id")
    return value


def training_running():
    job = state["job"]
    return job is not None and job.poll() is None


def status():
    root, champion, versions = registry(DATA)
    samples = load_samples(DATA)
    train, holdout = split(samples)
    state_path = root / "state.json"
    last = json.loads(state_path.read_text()) if state_path.exists() else {"lines_at_last_train": 0}
    log = DATA / "train.log"
    blind_path = DATA / "blind.json"
    trials = json.loads(blind_path.read_text()) if blind_path.exists() else []
    correct = sum(t["correct"] for t in trials)
    return {"lines": len(samples), "train": len(train), "holdout": len(holdout),
            "photo": sum(s["source"] == "photo" for s in samples),
            "new_lines": len(samples) - last["lines_at_last_train"], "retrain_every": RETRAIN_EVERY,
            "champion": champion, "versions": versions, "training": training_running(),
            "log": log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.exists() else "",
            "reports": sorted(p.name for p in (DATA / "reports").glob("*.html")),
            "glyphs": summary(load_library(GLYPHS)),
            "habits": {k: v for k, v in style_fit.load(DATA).items() if not isinstance(v, dict)},
            "real_words": len({s["text"] for s in samples if s["source"] == "photo" and " " not in s["text"]}),
            "blind": {"trials": len(trials), "correct": correct,
                      "passed": len(trials) >= 20 and correct / len(trials) <= 0.65}}


def start_training():
    if training_running():
        return False
    log = open(DATA / "train.log", "w", encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    state["job"] = subprocess.Popen([sys.executable, "-m", "hw.train", "--data", str(DATA)], cwd=str(ROOT / "ml"),
                                    stdout=log, stderr=subprocess.STDOUT, env=env)
    return True


def maybe_retrain():
    info = status()
    if info["new_lines"] >= RETRAIN_EVERY and info["holdout"] > 0 and info["lines"] - info["photo"] >= 20 and not info["training"]:
        return start_training()
    return False


class GenerateRequest(BaseModel):
    text: str
    sliders: dict = {}
    seed: int | None = None
    labels: list[dict] = []
    page: dict = {}
    engine: str = "model"


class FillRequest(BaseModel):
    boxes: list[dict]
    sliders: dict = {}
    seed: int | None = None
    engine: str = "model"


class PlotRequest(BaseModel):
    paths: list
    page: dict
    title: str = "page"
    dry_run: bool = True


class DiagramRequest(BaseModel):
    id: str
    boxes: list[dict]


class GlyphUpdate(BaseModel):
    char: str
    variants: list
    baseline: float


class WordUpdate(BaseModel):
    status: str
    text: str | None = None
    strokes: list | None = None


class BlindResult(BaseModel):
    correct: bool
    text: str


@app.get("/api/status")
def get_status():
    return status()


@app.post("/api/sessions")
def save_session(session: dict, final: int = 0):
    sid = safe_id(session.get("sessionId"))
    (DATA / "sessions" / f"{sid}.json").write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    xml_dir = DATA / "sessions" / f"{sid}_xml"
    xml_dir.mkdir(exist_ok=True)
    saved = 0
    for line in session.get("lines", []):
        strokes = clean(capture_line_to_strokes(line))
        if strokes:
            write_iam_xml(strokes, line["text"], xml_dir / f"{safe_id(str(line['id']))}.xml", str(line["id"]))
            saved += 1
    return {"saved": saved, "retraining": maybe_retrain() if final else False}


@app.post("/api/train")
def train_now():
    return {"started": start_training()}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    w = writer(req.engine)
    w.use_words = bool(req.sliders.get("real_words", False))
    w.joins = float(req.sliders.get("joins", 1.0))
    out = layout_page(w, req.text, req.sliders, req.seed, req.page or None, req.labels)
    out["checkpoint"] = "glyph library" if req.engine == "glyphs" else w.path.name
    out["default_size"] = w.size_mm
    return out


@app.get("/api/glyphs")
def get_glyphs():
    return load_library(GLYPHS)


@app.post("/api/glyphs")
def post_glyph(update: GlyphUpdate):
    if len(update.char) != 1 or update.char.isspace():
        raise HTTPException(400, "one visible character at a time")
    with lock:
        library = save_glyph(GLYPHS, update.char, update.variants, update.baseline)
    return {"glyphs": summary(library)}


@app.get("/api/template.pdf")
def template_pdf():
    path = DATA / "templates" / "letter_sheets.pdf"
    template.make_pdf(path)
    return FileResponse(path, media_type="application/pdf", filename="letter_sheets.pdf")


@app.post("/api/template/import")
async def template_import(file: UploadFile = File(...)):
    data = await file.read()
    first, pages, is_pdf = copypage.load_image(data, file.filename or "")
    if first is None:
        raise HTTPException(400, "not an image or PDF")
    images = [first] + [copypage.load_image(data, file.filename or "", n)[0] for n in range(1, pages)]
    results = []
    with lock:
        library = load_library(GLYPHS)
        for image in images:
            found = template.read_page(image)
            if found is None:
                results.append({"page": None})
                continue
            library["glyphs"].update(found["glyphs"])
            results.append({"page": found["page"], "units": len(found["glyphs"]),
                            "variants": sum(len(v) for v in found["glyphs"].values()), "empty": found["empty"]})
        GLYPHS.write_text(json.dumps(library, ensure_ascii=False), encoding="utf-8")
    return {"results": results, "glyphs": summary(library)}


@app.post("/api/worksheet")
async def upload_worksheet(file: UploadFile = File(...)):
    raw = np.frombuffer(await file.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "not an image")
    page, method = worksheet.deskew(image)
    boxes, size = worksheet.suggest_blanks(page)
    wid = f"ws_{int(time.time() * 1000)}"
    cv2.imwrite(str(DATA / "worksheets" / f"{wid}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"id": wid, "image": f"/files/worksheets/{wid}.jpg", "page": size, "boxes": boxes, "deskew": method}


@app.post("/api/worksheet/fill")
def fill_worksheet(req: FillRequest):
    w = writer(req.engine)
    w.use_words = bool(req.sliders.get("real_words", False))
    w.joins = float(req.sliders.get("joins", 1.0))
    sliders = sliders_with_defaults(req.sliders)
    seed = req.seed if req.seed is not None else random.randrange(2 ** 31)
    out = []
    for i, box in enumerate(req.boxes):
        if not str(box.get("text", "")).strip():
            out.append({"strokes": [], "size": 0})
            continue
        own = {**sliders, "size": box["size"]} if box.get("size") else sliders
        out.append(fit_box(w, box["text"], box, own, seed + i, "middle" if box.get("kind") == "box" else "bottom"))
    return {"answers": out}


def plotter_info():
    pen = plotter.load_pen(PLOTTER)
    x0, y0, x1, y1 = pen.reach()
    width, height = LETTER["width"], LETTER["height"]
    left = float(np.clip((x0 + x1) / 2 - width / 2, 0.0, plotter.BED - width))
    top_a, bottom_a = pen.back_stop_y - y1, pen.back_stop_y - y0
    top_b, bottom_b = height - pen.back_stop_y + y0, height - pen.back_stop_y + y1
    if y1 <= y0 or x1 <= x0:
        summary = "The pen cannot reach the plate with these settings."
    elif bottom_a >= height - 20:
        summary = (f"A letter page fits in one pass: the pen reaches from {max(top_a, 0):.0f} mm below the top edge "
                   f"to {min(bottom_a, height):.0f} mm, and from {x0 - left:.0f} to {x1 - left:.0f} mm across.")
    else:
        summary = (f"A letter page needs two passes. Right way up the pen reaches from {max(top_a, 0):.0f} to {bottom_a:.0f} mm "
                   f"down the page; turned upside down it reaches from {top_b:.0f} to {min(bottom_b, height):.0f} mm. "
                   f"Across, it reaches {x0 - left:.0f} to {x1 - left:.0f} mm. The paper's left edge sits {left:.0f} mm from the "
                   f"left of the plate and overhangs the front by {max(height - pen.back_stop_y, 0):.0f} mm.")
    return {"pen": asdict(pen), "reach": [x0, y0, x1, y1], "paper_left": left, "summary": summary, "problems": pen.problems()}


@app.get("/api/plotter")
def get_plotter():
    return plotter_info()


@app.post("/api/plotter")
def set_plotter(pen: dict):
    plotter.save_pen(plotter.pen_from(pen), PLOTTER)
    return plotter_info()


@app.post("/api/plot")
def make_plot(req: PlotRequest):
    pen = plotter.load_pen(PLOTTER)
    strokes = [np.asarray(p, dtype=np.float64) for p in req.paths if len(p)]
    parts, problems = plotter.place_page(strokes, float(req.page.get("width", LETTER["width"])),
                                         float(req.page.get("height", LETTER["height"])), pen)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", req.title).strip("-")[:40] or "page"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = []
    for i, part in enumerate(parts, 1):
        job, issues = plotter.build_gcode(part["strokes"], pen, req.title, req.dry_run)
        if job is None:
            problems += issues
            continue
        faults, _ = plotter.simulate(job["gcode"], pen if not req.dry_run else plotter.pen_from({**asdict(pen), "z_draw": pen.z_draw + 5,
                                                                                                    "z_lift": min(pen.z_lift + 5, pen.set_z() - 1)}))
        if faults:
            problems.append(f"The file failed its safety check and was not written: {faults[0]}")
            continue
        name = f"{stem}-{stamp}{'-dryrun' if req.dry_run else ''}-part{i}of{len(parts)}.gcode.3mf"
        (DATA / "plots" / name).write_bytes(plotter.package(job, part["strokes"], req.title))
        out.append({"name": name, "url": f"/files/plots/{name}", "turned": part["turned"], "strokes": job["strokes"],
                    "minutes": int(round(job["seconds"] / 60))})
    return {"parts": out if not problems else [], "problems": problems}


@app.get("/api/plot/calibration")
def plot_calibration():
    pen = plotter.load_pen(PLOTTER)
    job, problems = plotter.calibration(pen)
    if job is None:
        raise HTTPException(409, " ".join(problems))
    path = DATA / "plots" / "pen-calibration.gcode.3mf"
    path.write_bytes(plotter.package(job, [], "calibration"))
    return FileResponse(path, media_type="application/octet-stream", filename="pen-calibration.gcode.3mf")


@app.post("/api/copy")
async def copy_page(file: UploadFile = File(...), page: int = 0):
    found = copypage.analyse(await file.read(), file.filename or "", page)
    if found is None:
        raise HTTPException(400, "not an image or PDF")
    cid = f"cp_{int(time.time() * 1000)}"
    cv2.imwrite(str(DATA / "copies" / f"{cid}.jpg"), found["image"], [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(DATA / "copies" / f"{cid}_mask.png"), found["mask"])
    return {"id": cid, "image": f"/files/copies/{cid}.jpg", "page": found["page"], "pages": found["pages"],
            "boxes": found["boxes"], "deskew": found["method"], "diagram": copypage.diagram(found["mask"], found["boxes"])}


@app.post("/api/copy/diagram")
def copy_diagram(req: DiagramRequest):
    mask = cv2.imread(str(DATA / "copies" / f"{safe_id(req.id)}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(404, "unknown page")
    return {"diagram": copypage.diagram(mask, req.boxes)}


def process_photo(pid, image):
    job = state["photo_jobs"][pid]
    try:
        boxes = photo.segment_words(image)
        job["total"] = len(boxes)
        places = [(li, order, index) for li, line in enumerate(photo.group_lines(boxes, image)) for order, index in enumerate(line)]
        words = photo.derender(image, [boxes[index] for _, _, index in places], DATA / "photo_crops", pid)
        for word, (li, order, _) in zip(words, places):
            word.update({"photo": pid, "line": li, "order": order, "status": "pending" if word["strokes"] else "rejected",
                         "crop": f"/files/photo_crops/{word['id']}.png"})
            (DATA / "photo_words" / f"{word['id']}.json").write_text(json.dumps(word), encoding="utf-8")
        job.update({"state": "done", "words": len(words)})
    except Exception as exc:
        job.update({"state": "error", "error": str(exc)})


@app.post("/api/photos")
async def upload_photo(file: UploadFile = File(...)):
    raw = np.frombuffer(await file.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "not an image")
    scale = 2400 / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pid = f"p{int(time.time() * 1000)}"
    cv2.imwrite(str(DATA / "photos" / f"{pid}.jpg"), image)
    state["photo_jobs"][pid] = {"state": "running", "total": 0}
    threading.Thread(target=process_photo, args=(pid, image), daemon=True).start()
    return {"id": pid}


@app.get("/api/photos/jobs")
def photo_jobs():
    return state["photo_jobs"]


@app.get("/api/photos/words")
def list_words(status: str = "pending"):
    words = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((DATA / "photo_words").glob("*.json"))]
    counts = {}
    for w in words:
        counts[w["status"]] = counts.get(w["status"], 0) + 1
    return {"words": [w for w in words if w["status"] == status][:200], "counts": counts}


@app.post("/api/photos/words/{word_id}")
def update_word(word_id: str, update: WordUpdate):
    path = DATA / "photo_words" / f"{safe_id(word_id)}.json"
    if not path.exists():
        raise HTTPException(404, "unknown word")
    if update.status not in ("accepted", "rejected", "pending"):
        raise HTTPException(400, "bad status")
    word = json.loads(path.read_text(encoding="utf-8"))
    word["status"] = update.status
    if update.text is not None:
        word["text"] = update.text
    if update.strokes is not None:
        word["strokes"] = update.strokes
    path.write_text(json.dumps(word), encoding="utf-8")
    state["glyphs"] = None
    return {"retraining": maybe_retrain() if update.status == "accepted" else False}


def shifted(strokes, scale=1.0):
    x0, y0, _, _ = bounds(strokes)
    return [((s[:, :2] - [x0, y0]) * scale).round(3).tolist() for s in strokes]


@app.get("/api/blind")
def blind_trial():
    _, holdout = split(load_samples(DATA))
    if not holdout:
        raise HTTPException(409, "no held-out lines captured yet")
    sample = random.choice(holdout)
    w = writer()
    real = [s[:, :2] for s in sample["strokes"]]
    line = w.line(sample["text"], 0.15 + 1.35 * 0.6, random.randrange(2 ** 31))
    scale = core_metrics(real)["core"] / max(core_metrics(line["strokes"])["core"], 1e-6)
    pair = [{"real": True, "strokes": shifted(real)}, {"real": False, "strokes": shifted(line["strokes"], scale)}]
    random.shuffle(pair)
    return {"text": sample["text"], "options": pair, "checkpoint": w.path.name}


@app.post("/api/blind")
def blind_result(result: BlindResult):
    path = DATA / "blind.json"
    trials = json.loads(path.read_text()) if path.exists() else []
    trials.append({"correct": result.correct, "text": result.text, "checkpoint": champion_path(DATA).name, "at": time.time()})
    path.write_text(json.dumps(trials))
    return status()["blind"]


@app.delete("/api/blind")
def blind_reset():
    (DATA / "blind.json").unlink(missing_ok=True)
    return {"ok": True}


app.mount("/files", StaticFiles(directory=str(DATA)), name="files")

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
