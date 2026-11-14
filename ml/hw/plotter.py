import datetime
import hashlib
import io
import json
import math
import struct
import zipfile
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

BED = 256.0
PEN_LENGTH = 6.0
CUBE = 10.0
PEN_SET_Z = CUBE + PEN_LENGTH
FRONT_CLEAR = 5.0
EDGE = 1.0
CLIP_XY = (128.0, 20.0)
PARK_XY = (128.0, 250.0)
END_LIFT = 30.0
MAX_LINE = 48
BAMBU_VERSION = "02.08.02.61"
PRINTER_MODEL_ID = "C12"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "weights" / "bambu" / "project_settings.json"


@dataclass
class Pen:
    offset_x: float = 0.0
    offset_y: float = 0.0
    holder_front: float = 60.0
    z_draw: float = 5.0
    z_lift: float = 8.0
    draw_speed: float = 1500.0
    travel_speed: float = 6000.0
    z_speed: float = 600.0
    accel: float = 3000.0
    dot_dwell_ms: float = 80.0
    back_stop_y: float = 254.0
    calibrated: float = 0.0

    def problems(self):
        out = []
        if self.z_draw < 0.5:
            out.append("Draw height must be at least 0.5 mm so the nozzle never reaches the paper.")
        if self.z_lift < PEN_LENGTH + 1.0:
            out.append(f"Lift height must be at least {PEN_LENGTH + 1:g} mm so the pen tip clears the paper.")
        if self.z_lift <= self.z_draw:
            out.append("Lift height must be above draw height.")
        if self.z_lift > PEN_SET_Z - 1.0:
            out.append(f"Lift height must be {PEN_SET_Z - 1:g} mm or less.")
        for name in ("draw_speed", "travel_speed", "z_speed", "accel"):
            if getattr(self, name) <= 0:
                out.append(f"{name} must be greater than 0.")
        if not 0 <= self.holder_front <= 150:
            out.append("Holder reach must be between 0 and 150 mm.")
        if abs(self.offset_x) >= 100 or abs(self.offset_y) >= 100:
            out.append("Pen offsets must be smaller than 100 mm.")
        if not 150 <= self.back_stop_y <= 258:
            out.append("Paper back edge must sit between Y150 and Y258.")
        return out

    def min_nozzle_y(self):
        return self.holder_front + FRONT_CLEAR if self.holder_front > 0 else 0.0

    def reach(self):
        x0 = max(0.0, self.offset_x) + EDGE
        x1 = min(BED, BED + self.offset_x) - EDGE
        y0 = max(0.0, self.min_nozzle_y() + self.offset_y) + EDGE
        y1 = min(BED, BED + self.offset_y) - EDGE
        return x0, y0, x1, y1


def pen_from(data):
    known = {f.name for f in fields(Pen)}
    return Pen(**{k: float(v) for k, v in (data or {}).items() if k in known and v is not None})


def load_pen(path):
    path = Path(path)
    return pen_from(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else Pen()


def save_pen(pen, path):
    Path(path).write_text(json.dumps(asdict(pen), indent=1), encoding="utf-8")


def num(value):
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def thin(points, spacing=0.06):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points
    keep = [0]
    for i in range(1, len(points) - 1):
        if np.linalg.norm(points[i] - points[keep[-1]]) >= spacing:
            keep.append(i)
    keep.append(len(points) - 1)
    return points[keep]


def place_page(strokes, page_w, page_h, pen):
    x0, y0, x1, y1 = pen.reach()
    if page_w > BED:
        return [], [f"The page is {page_w:g} mm wide but the plate is {BED:g} mm."]
    left = float(np.clip((x0 + x1) / 2 - page_w / 2, 0.0, BED - page_w))
    back = pen.back_stop_y
    upright = lambda p: np.stack([left + p[:, 0], back - p[:, 1]], axis=1)
    turned = lambda p: np.stack([left + page_w - p[:, 0], back - (page_h - p[:, 1])], axis=1)
    inside = lambda q: bool((q[:, 0] >= x0).all() and (q[:, 0] <= x1).all() and (q[:, 1] >= y0).all() and (q[:, 1] <= y1).all())
    strokes = [np.asarray(s, dtype=np.float64) for s in strokes if len(s)]
    fits_up = [inside(upright(s)) for s in strokes]
    fits_turn = [inside(turned(s)) for s in strokes]
    lost = [s for s, a, b in zip(strokes, fits_up, fits_turn) if not a and not b]
    if lost:
        pts = np.concatenate(lost)
        top_limit, side = back - y1, left
        return [], [f"{len(lost)} strokes cannot be reached by the pen (page x {pts[:, 0].min():.0f}-{pts[:, 0].max():.0f} mm, "
                    f"y {pts[:, 1].min():.0f}-{pts[:, 1].max():.0f} mm). With this pen holder the pen reaches page x "
                    f"{x0 - side:.0f}-{x1 - side:.0f} mm and needs at least {top_limit:.0f} mm of blank margin at the top and bottom."]
    if all(fits_up):
        return [{"turned": False, "strokes": [upright(s) for s in strokes]}], []
    if all(fits_turn):
        return [{"turned": True, "strokes": [turned(s) for s in strokes]}], []
    tops = np.array([s[:, 1].min() for s in strokes])
    bottoms = np.array([s[:, 1].max() for s in strokes])
    must_up, must_turn = ~np.array(fits_turn), ~np.array(fits_up)
    lo = float(bottoms[must_up].max()) if must_up.any() else float(tops.min()) - 1.0
    hi = float(bottoms[must_turn].min()) if must_turn.any() else float(bottoms.max()) + 1.0
    if lo >= hi:
        return [], ["The page cannot be split into two passes with this pen holder: the two reachable bands do not overlap enough."]
    best, cut = -1.0, lo
    for y in np.concatenate([[lo], bottoms[(bottoms > lo) & (bottoms < hi)]]):
        below = tops[bottoms > y]
        gap = float(below.min() - y) if len(below) else 0.0
        score = gap - 0.002 * abs(y - (lo + hi) / 2)
        if score > best:
            best, cut = score, float(y)
    first = [upright(s) for s, bottom in zip(strokes, bottoms) if bottom <= cut]
    second = [turned(s) for s, bottom in zip(strokes, bottoms) if bottom > cut]
    return [{"turned": False, "strokes": first}, {"turned": True, "strokes": second}], []


def check_points(points, pen, label):
    points = np.asarray(points, dtype=np.float64)
    nozzle = points - [pen.offset_x, pen.offset_y]
    out = []
    if points.min() < 0 or points.max() > BED:
        out.append(f"{label}: the pen would leave the plate.")
    if nozzle[:, 0].min() < 0 or nozzle[:, 0].max() > BED or nozzle[:, 1].max() > BED:
        out.append(f"{label}: the nozzle would leave the plate.")
    if nozzle[:, 1].min() < pen.min_nozzle_y():
        out.append(f"{label}: the holder would hit the front; keep drawing at least {pen.min_nozzle_y() - nozzle[:, 1].min():.1f} mm further back.")
    return out


def clip_xy(pen):
    x, y = CLIP_XY
    return x, max(y, pen.min_nozzle_y(), -pen.offset_y)


def park_xy(pen):
    return PARK_XY[0], min(PARK_XY[1], BED - max(pen.offset_y, 0.0) - EDGE)


def pause(count):
    return ["M104 S0", "M140 S0", f"; layer num/total_layer_count: {count}", f"M73 L{count}", f"M991 S0 P{count - 1}",
            "M400", "M400 U1", "M104 S0", "M140 S0"]


def build_gcode(strokes, pen, title="page", dry_run=False, after=None):
    problems = pen.problems()
    strokes = [thin(s) for s in strokes if len(s)]
    if not strokes:
        problems.append("Nothing to draw.")
    else:
        problems += check_points(np.concatenate(strokes), pen, "Drawing")
    cx, cy = clip_xy(pen)
    problems += check_points([[cx + pen.offset_x, cy + pen.offset_y]], pen, "Pen fitting position")
    park = park_xy(pen)
    problems += check_points([[park[0] + pen.offset_x, park[1] + pen.offset_y]], pen, "Park position")
    if problems:
        return None, problems
    raise_by = 5.0 if dry_run else 0.0
    z_draw = pen.z_draw + raise_by
    z_lift = min(pen.z_lift + raise_by, PEN_SET_Z - 1.0)
    if z_lift <= z_draw:
        z_lift = z_draw + 1.0
    length = sum(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in strokes)
    travel, cursor = 0.0, np.array([cx + pen.offset_x, cy + pen.offset_y])
    for s in strokes:
        travel += float(np.linalg.norm(s[0] - cursor))
        cursor = s[-1]
    lifts = len(strokes)
    seconds = 90 + length / pen.draw_speed * 60 + travel / pen.travel_speed * 60 + lifts * 2 * (z_lift - z_draw) / pen.z_speed * 60 \
        + sum(1 for s in strokes if len(s) == 1) * pen.dot_dwell_ms / 1000
    minutes, rest = divmod(int(round(seconds)), 60)
    pauses = 1 + (1 if after else 0)
    out = ["; HEADER_BLOCK_START", f"; BambuStudio {BAMBU_VERSION}",
           f"; model printing time: {minutes}m {rest}s; total estimated time: {minutes}m {rest}s",
           f"; total layer number: {pauses}", "; total filament length [mm] : 0.00", "; total filament volume [cm^3] : 0.00",
           "; total filament weight [g] : 0.00", f"; max_z_height: {PEN_SET_Z:.2f}", "; HEADER_BLOCK_END", "",
           f"; pen plot: {title}", "; EXECUTABLE_BLOCK_START",
           "G21", "G90", "M104 S0", "M140 S0", "M73 P0", "G28", "M104 S0", "M140 S0",
           "M1002 gcode_claim_action : 0", "; MACHINE_START_GCODE_END", f"M204 S{num(pen.accel)}",
           f"G0 Z{num(PEN_SET_Z)} F{num(pen.z_speed)}", f"G0 X{num(cx)} Y{num(cy)} F{num(pen.travel_speed)}",
           *pause(1), "; PEN_FITTED", f"G0 Z{num(z_lift)} F{num(pen.z_speed)}"]
    done, last_pct, up, cursor = 0.0, 0, True, None
    for s in strokes:
        nozzle = s - [pen.offset_x, pen.offset_y]
        joined = cursor is not None and not up and float(np.linalg.norm(s[0] - cursor)) < 0.12
        if not joined:
            if not up:
                out.append(f"G0 Z{num(z_lift)} F{num(pen.z_speed)}")
            out.append(f"G0 X{num(nozzle[0, 0])} Y{num(nozzle[0, 1])} F{num(pen.travel_speed)}")
            out.append(f"G1 Z{num(z_draw)} F{num(pen.z_speed)}")
        if len(s) == 1:
            out.append(f"G4 P{num(pen.dot_dwell_ms)}")
        for k, p in enumerate(nozzle[1:]):
            out.append(f"G1 X{num(p[0])} Y{num(p[1])}" + (f" F{num(pen.draw_speed)}" if k == 0 else ""))
        up, cursor = False, s[-1]
        done += float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())
        pct = min(99, int(100 * done / max(length, 1e-6)))
        if pct != last_pct:
            out.append(f"M73 P{pct} R{math.ceil(seconds * (1 - done / max(length, 1e-6)) / 60)}")
            last_pct = pct
    out.append(f"G0 Z{num(z_lift)} F{num(pen.z_speed)}")
    if after:
        out.append(f"G0 X{num(after[0])} Y{num(after[1])} F{num(pen.travel_speed)}")
        out += pause(2)
    out += ["M104 S0", "M140 S0", f"G0 Z{num(z_lift + END_LIFT)} F{num(pen.z_speed)}",
            f"G0 X{num(park[0])} Y{num(park[1])} F{num(pen.travel_speed)}", "M73 P100 R0", "M400", "M18",
            "; EXECUTABLE_BLOCK_END", ""]
    return {"gcode": "\n".join(out), "seconds": seconds, "strokes": len(strokes), "pauses": pauses}, []


def calibration(pen):
    cx = cy = BED / 2
    circle = [[cx + 5 * math.cos(2 * math.pi * k / 72), cy + 5 * math.sin(2 * math.pi * k / 72)] for k in range(73)]
    marks = [np.array([[cx - 15, cy], [cx + 15, cy]]), np.array([[cx, cy - 15], [cx, cy + 15]]), np.array(circle)]
    extra = check_points([[cx + pen.offset_x, cy + pen.offset_y]], pen, "Calibration park")
    job, problems = build_gcode(marks, pen, "calibration", after=(cx, cy))
    if extra:
        return None, problems + extra
    return job, problems


def png(image):
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def thumbnail(strokes, size):
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    k = size / BED
    for s in strokes:
        for a, b in zip(s, s[1:]):
            n = max(2, math.ceil(math.dist(a, b) * k * 2))
            t = np.linspace(0, 1, n)
            cols = np.clip(((a[0] + (b[0] - a[0]) * t) * k).astype(int), 0, size - 1)
            rows = np.clip(((BED - (a[1] + (b[1] - a[1]) * t)) * k).astype(int), 0, size - 1)
            image[rows, cols] = (20, 20, 20)
    return png(image)


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
</Types>
"""
RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
 <Relationship Target="/Metadata/plate_1.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>
 <Relationship Target="/Metadata/plate_1.png" Id="rel-4" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>
 <Relationship Target="/Metadata/plate_1_small.png" Id="rel-5" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>
</Relationships>
"""
SETTINGS_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/Metadata/plate_1.gcode" Id="rel-1" Type="http://schemas.bambulab.com/package/2021/gcode"/>
</Relationships>
"""
MODEL_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <metadata key="filament_map_mode" value="Auto For Flush"/>
    <metadata key="filament_maps" value="1"/>
    <metadata key="filament_volume_maps" value="0"/>
    <metadata key="gcode_file" value="Metadata/plate_1.gcode"/>
    <metadata key="thumbnail_file" value="Metadata/plate_1.png"/>
    <metadata key="thumbnail_no_light_file" value="Metadata/plate_no_light_1.png"/>
    <metadata key="top_file" value="Metadata/top_1.png"/>
    <metadata key="pick_file" value="Metadata/pick_1.png"/>
    <metadata key="pattern_bbox_file" value="Metadata/plate_1.json"/>
  </plate>
</config>
"""


def package(job, strokes, title="page"):
    data = job["gcode"].encode("utf-8")
    today = datetime.date.today().isoformat()
    name = quoteattr(title)[1:-1]
    model = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">
 <metadata name="Application">BambuStudio-{BAMBU_VERSION}</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="CreationDate">{today}</metadata>
 <metadata name="ModificationDate">{today}</metadata>
 <metadata name="Title">{name}</metadata>
 <resources>
 </resources>
 <build/>
</model>
"""
    info = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="{BAMBU_VERSION}"/>
  </header>
  <plate>
    <metadata key="index" value="1"/>
    <metadata key="extruder_type" value="0"/>
    <metadata key="nozzle_volume_type" value="0"/>
    <metadata key="printer_model_id" value="{PRINTER_MODEL_ID}"/>
    <metadata key="nozzle_diameters" value="0.4"/>
    <metadata key="timelapse_type" value="0"/>
    <metadata key="prediction" value="{int(round(job['seconds']))}"/>
    <metadata key="weight" value="0.00"/>
    <metadata key="pause_count" value="{job['pauses']}"/>
    <metadata key="first_layer_time" value="0"/>
    <metadata key="outside" value="false"/>
    <metadata key="support_used" value="false"/>
    <metadata key="label_object_enabled" value="false"/>
    <metadata key="support_material_on_wipe_tower" value="false"/>
    <metadata key="enable_filament_dynamic_map" value="false"/>
    <metadata key="has_filament_switcher" value="false"/>
    <metadata key="filament_maps" value="1"/>
    <metadata key="limit_filament_maps" value="0"/>
    <filament id="1" tray_info_idx="GFL99" type="PLA" color="#161616" used_m="0.00" used_g="0.00" group_id="0" nozzle_diameter="0.40" volume_type="Standard" used_for_object="true" used_for_support="false" total_load_time="0.00" total_unload_time="0.00"/>
    <nozzle id="0" extruder_id="1" nozzle_diameter="0.4" volume_type="Standard"/>
    <layer_filament_lists>
      <layer_filament_list filament_list="0" layer_ranges="0 0" />
    </layer_filament_lists>
  </plate>
</config>
"""
    pts = np.concatenate(strokes) if strokes else np.zeros((1, 2))
    plate = json.dumps({"bbox_all": [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())],
                        "bbox_objects": [], "bed_type": "textured_plate", "filament_colors": ["#161616"], "filament_ids": [0],
                        "first_extruder": 0, "first_layer_time": 0, "is_seq_print": False, "nozzle_diameter": 0.4, "version": 2})
    big, small = thumbnail(strokes, 512), thumbnail(strokes, 128)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)
        for item in ("plate_1.png", "plate_no_light_1.png", "top_1.png", "pick_1.png"):
            z.writestr(f"Metadata/{item}", big)
        z.writestr("Metadata/plate_1_small.png", small)
        z.writestr("Metadata/plate_1.json", plate)
        z.writestr("Metadata/project_settings.config", SETTINGS_FILE.read_bytes())
        z.writestr("Metadata/plate_1.gcode", data)
        z.writestr("Metadata/plate_1.gcode.md5", hashlib.md5(data).hexdigest().upper())
        z.writestr("Metadata/_rels/model_settings.config.rels", SETTINGS_RELS)
        z.writestr("Metadata/model_settings.config", MODEL_SETTINGS)
        z.writestr("Metadata/cut_information.xml", '<?xml version="1.0" encoding="utf-8"?>\n<objects>\n</objects>\n')
        z.writestr("Metadata/slice_info.config", info)
        z.writestr("Metadata/filament_sequence.json", '{"plate_1":{"nozzle_sequence":[0],"optimal_assignment":[0],"sequence":[1]}}')
    return buffer.getvalue()


def simulate(gcode, pen):
    x = y = z = None
    fitted, faults, drawn = False, [], 0
    for line in gcode.splitlines():
        code = line.split(";")[0].strip()
        if line.startswith("; PEN_FITTED"):
            fitted = True
        if not code:
            continue
        if len(line) > MAX_LINE:
            faults.append(f"line too long: {line[:30]}")
        word = code.split()[0]
        args = {a[0]: a[1:] for a in code.split()[1:] if a}
        if word in ("M104", "M140", "M109", "M190") and float(args.get("S", "0")) != 0:
            faults.append("heater switched on")
        if "E" in args and word in ("G0", "G1"):
            faults.append("extrusion")
        if word in ("G0", "G1"):
            nx = float(args["X"]) if "X" in args else x
            ny = float(args["Y"]) if "Y" in args else y
            nz = float(args["Z"]) if "Z" in args else z
            moves_xy = ("X" in args or "Y" in args) and (nx != x or ny != y)
            if "Z" in args and ("X" in args or "Y" in args):
                faults.append("XY and Z in one move")
            if fitted and nz is not None and nz < pen.z_draw - 1e-6:
                faults.append(f"Z below draw height: {nz}")
            if moves_xy:
                if not (0 <= nx <= BED and 0 <= ny <= BED):
                    faults.append(f"nozzle off plate: {nx},{ny}")
                if not (0 <= nx + pen.offset_x <= BED and 0 <= ny + pen.offset_y <= BED):
                    faults.append(f"pen off plate: {nx + pen.offset_x},{ny + pen.offset_y}")
                if ny < pen.min_nozzle_y() - 1e-6:
                    faults.append(f"holder past the front: Y{ny}")
                if fitted and word == "G0" and z is not None and z < pen.z_lift - 1e-6:
                    faults.append("travel with the pen down")
                if word == "G1":
                    drawn += 1
            x, y, z = nx, ny, nz
    return faults, drawn
