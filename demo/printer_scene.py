import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_demo as demo
from hw import plotter

HARDWARE = demo.ROOT / "hardware"
NOZZLE_Z = 330.0
SHELL = (-66.0, 322.0, -62.0, 327.0, 0.0, 458.0)
MOUNT_NOZZLE = (31.5, 14.0)
GUIDE_BELOW_NOZZLE = 16.5
SEAT = 1.35
RING = (63.83, 49.71)
TEX = 4.0
PENCIL_LEN = 150.0


def box(x0, x1, y0, y1, z0, z1):
    return pv.Box(bounds=(x0, x1, y0, y1, z0, z1))


def rod(a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return pv.Cylinder(center=(a + b) / 2, direction=b - a, radius=r, height=float(np.linalg.norm(b - a)), resolution=24)


def from_mount(mesh):
    out = mesh.copy()
    pts = out.points.copy()
    out.points = np.stack([MOUNT_NOZZLE[0] - pts[:, 0], MOUNT_NOZZLE[1] - pts[:, 1], pts[:, 2] - SEAT - GUIDE_BELOW_NOZZLE], axis=1)
    return out


def timeline(gcode, pen):
    x, y = plotter.clip_xy(pen)
    z, feed, t = pen.set_z(), pen.travel_speed, 0.0
    rows, ink, started = [(0.0, x, y, z)], [], False
    for line in gcode.splitlines():
        if line.startswith("; PEN_FITTED"):
            started = True
        code = line.split(";")[0].split()
        if not started or not code:
            continue
        args = {a[0]: float(a[1:]) for a in code[1:] if a[1:].replace(".", "").replace("-", "").isdigit()}
        if code[0] == "G4":
            t += args.get("P", 0.0) / 1000
            ink.append((t, x + pen.offset_x, y + pen.offset_y, x + pen.offset_x, y + pen.offset_y))
            rows.append((t, x, y, z))
            continue
        if code[0] not in ("G0", "G1"):
            continue
        feed = args.get("F", feed)
        nx, ny, nz = args.get("X", x), args.get("Y", y), args.get("Z", z)
        dist = math.hypot(nx - x, ny - y) + abs(nz - z)
        if dist == 0:
            continue
        t += dist / (feed / 60)
        if code[0] == "G1" and (nx != x or ny != y):
            ink.append((t, x + pen.offset_x, y + pen.offset_y, nx + pen.offset_x, ny + pen.offset_y))
        x, y, z = nx, ny, nz
        rows.append((t, x, y, z))
    return np.array(rows), np.array(ink)


class Printer:
    def __init__(self, pen, page, size=(1280, 720)):
        self.pen, self.page = pen, page
        self.left = plotter.paper_left(pen, page["width"])
        self.pl = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
        self.pl.set_background((0.93, 0.94, 0.96), top=(0.99, 0.99, 1.0))
        self.pl.enable_depth_peeling(8)
        self.pl.enable_anti_aliasing("ssaa")
        self.pl.add_light(pv.Light(position=(128, 40, 445), focal_point=(128, 128, 150), intensity=0.55, positional=True, cone_angle=75))
        self.groups = {"head": [], "bed": [], "beam": [], "tool": [], "pencil": [], "block": [], "door": []}
        self.shell()
        self.bed()
        self.gantry()
        self.toolhead()
        self.holder()
        self.paper = None
        self.canvas = demo.paper(page, TEX)
        self.make_paper()
        self.state = {"x": 128.0, "y": 240.0, "z": pen.set_z(), "fitted": False, "pencil_up": 0.0, "turn": 0.0, "paper_lift": 0.0}
        self.show("tool", False)
        self.show("pencil", False)
        self.show("block", False)
        self.pl.camera.view_angle = 32
        self.apply()

    def add(self, group, mesh, colour, **kw):
        actor = self.pl.add_mesh(mesh, color=colour, smooth_shading=kw.pop("smooth", False), **kw)
        if group:
            self.groups[group].append(actor)
        return actor

    def show(self, group, on):
        for actor in self.groups[group]:
            actor.SetVisibility(bool(on))

    def shell(self):
        x0, x1, y0, y1, z0, z1 = SHELL
        dark, panel = (0.13, 0.13, 0.14), (0.3, 0.31, 0.33)
        self.add(None, box(x0, x1, y0, y1, z0, 14), dark)
        self.add(None, box(x0, x1, y1 - 6, y1, z0, z1), panel)
        self.add(None, box(x0, x0 + 6, y0, y1, z0, z1), panel)
        self.add(None, box(x1 - 6, x1, y0, y1, z0, z1), panel)
        for px in (x0, x1 - 16):
            for py in (y0, y1 - 16):
                self.add(None, box(px, px + 16, py, py + 16, z0, z1), dark)
        self.add(None, box(x0, x1, y0, y0 + 10, z0, 78), dark)
        self.add(None, box(x0, x1, y0, y0 + 16, z1 - 18, z1), dark)
        self.add(None, box(x0, x1, y1 - 16, y1, z1 - 18, z1), dark)
        self.add(None, box(x0, x0 + 16, y0, y1, z1 - 18, z1), dark)
        self.add(None, box(x1 - 16, x1, y0, y1, z1 - 18, z1), dark)
        self.add(None, box(x1 - 96, x1 - 22, y0 - 3, y0, z1 - 62, z1 - 24), (0.06, 0.06, 0.07))
        self.add(None, box(x1 - 90, x1 - 52, y0 - 3.6, y0 - 3, z1 - 56, z1 - 32), (0.75, 0.86, 0.9))
        self.add("door", box(x0 + 16, x1 - 16, y0 + 2, y0 + 6, 80, z1 - 20), (0.55, 0.62, 0.66), opacity=0.22)
        self.add("door", box(x0 + 16, x0 + 24, y0 - 4, y0 + 2, 230, 300), (0.75, 0.75, 0.78))
        for x in (-28, 284):
            self.add(None, rod((x, 285, 14), (x, 285, 420), 4), (0.7, 0.7, 0.72), smooth=True)
        self.add(None, rod((284, -20, 14), (284, -20, 420), 4), (0.7, 0.7, 0.72), smooth=True)

    def bed(self):
        self.add("bed", box(-6, 262, -6, 262, -11, -1.2), (0.16, 0.16, 0.17))
        self.add("bed", box(0, 256, 0, 256, -1.2, 0), (0.72, 0.58, 0.25))
        self.add("bed", box(98, 158, -12, 0, -1.2, 0), (0.72, 0.58, 0.25))
        self.add("bed", box(-36, 292, 270, 296, -30, -11), (0.2, 0.2, 0.22))
        self.add("bed", box(272, 296, -30, 296, -30, -11), (0.2, 0.2, 0.22))
        self.add("block", box(-5, 5, -5, 5, 0.3, 10.3), (0.2, 0.55, 0.9))

    def gantry(self):
        z = NOZZLE_Z + 62
        for x in (-46, 302):
            self.add(None, rod((x, -40, z), (x, 312, z), 5), (0.72, 0.72, 0.75), smooth=True)
            self.add("beam", box(x - 12, x + 12, 22, 62, z - 14, z + 14), (0.1, 0.1, 0.11))
        for dz in (-8, 12):
            self.add("beam", rod((-46, 42, z + dz), (302, 42, z + dz), 4), (0.08, 0.08, 0.09), smooth=True)

    def toolhead(self):
        self.add("head", box(-23.5, 23.5, -22, 40, 12, 86), (0.36, 0.37, 0.4))
        self.add("head", box(-20, 20, 40, 52, 30, 92), (0.22, 0.22, 0.24))
        self.add("head", box(-8, 8, -8, 8, 3, 14), (0.75, 0.25, 0.1))
        self.add("head", pv.Cone(center=(0, 0, 1.5), direction=(0, 0, -1), height=3, radius=2.2, resolution=24), (0.78, 0.62, 0.25))
        self.add("head", pv.Cylinder(center=(0, -22.4, 50), direction=(0, 1, 0), radius=17, height=1, resolution=48), (0.05, 0.05, 0.06))
        self.add("head", rod((0, 10, 86), (0, 10, 128), 2.2), (0.9, 0.9, 0.92), smooth=True)

    def holder(self):
        falu = HARDWARE / "falu"
        mount = pv.read(str(falu / "Head-Mount-V1-Series X and Series P.stl"))
        guide = pv.read(str(HARDWARE / "out" / "pencil_guide.stl")).translate((RING[0], RING[1], SEAT), inplace=False)
        spring = pv.read(str(falu / "Upper Spring System" / "Spring Module D9 5mm.stl"))
        spring = spring.translate((-24.2, 0, -10), inplace=False).rotate_x(90, inplace=False).rotate_z(105, inplace=False)
        spring = spring.translate((RING[0], RING[1], SEAT + 70 - 9.25), inplace=False)
        self.add("tool", from_mount(mount), (0.93, 0.93, 0.9), smooth=True, split_sharp_edges=True)
        self.add("tool", from_mount(guide), (0.95, 0.5, 0.12), smooth=True, split_sharp_edges=True)
        self.add("tool", from_mount(spring), (0.93, 0.93, 0.9), smooth=True, split_sharp_edges=True)
        ax, ay = self.pen.offset_x, self.pen.offset_y
        tip = -self.pen.tip_depth
        self.add("pencil", rod((ax, ay, tip), (ax, ay, tip + 4), 0.9), (0.2, 0.2, 0.22), smooth=True)
        self.add("pencil", pv.Cone(center=(ax, ay, tip + 9), direction=(0, 0, -1), height=10, radius=4.0, resolution=32), (0.78, 0.78, 0.8), smooth=True)
        self.add("pencil", rod((ax, ay, tip + 14), (ax, ay, tip + PENCIL_LEN - 14), 4.0), (0.96, 0.78, 0.12), smooth=True)
        self.add("pencil", rod((ax, ay, tip + PENCIL_LEN - 14), (ax, ay, tip + PENCIL_LEN - 8), 4.1), (0.75, 0.75, 0.78), smooth=True)
        self.add("pencil", rod((ax, ay, tip + PENCIL_LEN - 8), (ax, ay, tip + PENCIL_LEN), 3.8), (0.9, 0.45, 0.5), smooth=True)

    def make_paper(self):
        w, h, back = self.page["width"], self.page["height"], self.pen.back_stop_y
        pts = np.array([[self.left, back, 0.35], [self.left + w, back, 0.35], [self.left + w, back - h, 0.35], [self.left, back - h, 0.35]])
        quad = pv.PolyData(pts, faces=[4, 0, 1, 2, 3])
        quad.active_texture_coordinates = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        self.paper = self.pl.add_mesh(quad, texture=self.texture(), lighting=True, ambient=0.35)
        self.paper.SetOrigin(self.left + w / 2, back - h / 2, 0)

    def texture(self):
        return pv.Texture(np.ascontiguousarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)))

    def refresh_paper(self):
        self.paper.SetTexture(self.texture())

    def ink(self, segments, turned):
        w, h, back = self.page["width"], self.page["height"], self.pen.back_stop_y
        for _, xa, ya, xb, yb in segments:
            if turned:
                a = (self.left + w - xa, h - (back - ya))
                b = (self.left + w - xb, h - (back - yb))
            else:
                a = (xa - self.left, back - ya)
                b = (xb - self.left, back - yb)
            pa = tuple(int(round(v * TEX * 16)) for v in a)
            pb = tuple(int(round(v * TEX * 16)) for v in b)
            if pa == pb:
                cv2.circle(self.canvas, pa, 22, demo.INK, -1, cv2.LINE_AA, 4)
            else:
                cv2.line(self.canvas, pa, pb, demo.INK, 2, cv2.LINE_AA, 4)

    def apply(self):
        s = self.state
        bed_z = NOZZLE_Z - s["z"]
        for actor in self.groups["head"] + self.groups["tool"]:
            actor.SetPosition(s["x"], s["y"], NOZZLE_Z)
        push = max(0.0, self.pen.tip_depth - s["z"]) if s["fitted"] else 0.0
        for actor in self.groups["pencil"]:
            actor.SetPosition(s["x"], s["y"], NOZZLE_Z + push + s["pencil_up"])
        for actor in self.groups["beam"]:
            actor.SetPosition(0, s["y"], 0)
        for actor in self.groups["bed"]:
            actor.SetPosition(0, 0, bed_z)
        for actor in self.groups["block"]:
            actor.SetPosition(s["x"] + self.pen.offset_x, s["y"] + self.pen.offset_y, bed_z)
        self.paper.SetPosition(0, 0, bed_z + s["paper_lift"])
        self.paper.SetOrientation(0, 0, s["turn"])

    def look(self, position, focus, up=(0, 0, 1), angle=32):
        cam = self.pl.camera
        cam.position, cam.focal_point, cam.up, cam.view_angle = position, focus, up, angle
        self.pl.renderer.ResetCameraClippingRange()

    def frame(self):
        self.apply()
        self.pl.render()
        return cv2.cvtColor(self.pl.screenshot(return_img=True), cv2.COLOR_RGB2BGR)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def shot(name, p, t=0.0, follow=None):
    s = p.state
    bed_z = NOZZLE_Z - s["z"]
    tip = follow if follow is not None else (s["x"] + p.pen.offset_x, s["y"] + p.pen.offset_y)
    if name == "outside":
        a = math.radians(-118 + 26 * t)
        return (128 + 1080 * math.cos(a), 130 + 1080 * math.sin(a), 560), (128, 130, 235), (0, 0, 1), 30
    if name == "close":
        return (tip[0] - 80, tip[1] - 140, bed_z + 82), (tip[0] + 6, tip[1] + 6, bed_z + 16), (0, 0, 1), 34
    if name == "fit":
        return (tip[0] - 150, tip[1] - 250, bed_z + 150), (tip[0] + 25, tip[1] + 20, bed_z + 45), (0, 0, 1), 34
    if name == "inside":
        return (-20 + 40 * t, -335, 445), (128, 118, bed_z + 5), (0, 0, 1), 38
    if name == "side":
        return (-48, -30 + 30 * t, bed_z + 150), (150, 150, bed_z + 10), (0, 0, 1), 44
    return (128, 105, 1250), (128, 112, bed_z), (0, 1, 0), 22
