import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, Mesh

OUT = Path(__file__).resolve().parent / "out"
SEG = 128
MOUNT_FILE = "Head-Mount-V1-Series X and Series P.stl"
GUIDE_FILE = "Tool Guide Modules/Guide Stabilo D 8 55mm.stl"
SPRINGS = {"D9": ("Upper Spring System/Spring Module D9 5mm.stl", 9.28), "D11": ("Upper Spring System/Spring Module D11 5 mm.stl", 11.28)}
RING_AXIS = (63.83, 49.71)
OUTER = [
    (0.0, 7.95), (0.8, 8.75), (8.9, 8.75), (10.15, 7.5), (12.3, 7.5), (13.55, 8.75),
    (32.7, 8.75), (33.95, 7.5), (36.05, 7.5), (37.3, 8.75),
    (56.55, 8.75), (57.8, 7.5), (59.9, 7.5), (61.15, 8.75),
    (65.5, 8.75), (66.75, 7.5), (67.25, 7.5), (68.5, 8.75), (69.2, 8.75), (70.0, 7.95),
]
LENGTH = 70.0


@dataclass
class Guide:
    pencil_d: float = 8.0
    fit: float = 0.4
    tip_d: float = 0.0
    tip_len: float = 0.0
    chamfer: float = 0.6

    @property
    def bore(self):
        return (self.pencil_d + self.fit) / 2

    @property
    def tip_bore(self):
        return (self.tip_d + self.fit) / 2 if self.tip_d and self.tip_len else self.bore


def guide(g: Guide):
    if g.bore > 6.9:
        raise SystemExit(f"A {g.pencil_d} mm pencil leaves less than 0.6 mm of wall in the 15 mm grooves.")
    inner = [(LENGTH, g.bore + g.chamfer), (LENGTH - g.chamfer, g.bore)]
    if g.tip_bore < g.bore:
        inner += [(g.tip_len + 1.0, g.bore), (g.tip_len, g.tip_bore), (g.chamfer, g.tip_bore), (0.0, g.tip_bore + g.chamfer)]
    else:
        inner += [(g.chamfer, g.bore), (0.0, g.bore + g.chamfer)]
    poly = [(r, z) for z, r in OUTER] + [(r, z) for z, r in inner]
    return Manifold.revolve(CrossSection([poly]), SEG)


def fit_rings(pencil_d, steps=(-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)):
    out = Manifold()
    x = 0.0
    for i, s in enumerate(steps):
        r = (pencil_d + s) / 2
        R = r + 2.4
        ring = Manifold.cylinder(8, R, R, 96).translate([x, 0, 0]) - Manifold.cylinder(10, r, r, 96).translate([x, 0, -1])
        for k in range(i + 1):
            a = math.pi / 2 + (k - i / 2) * 0.32
            ring = ring + Manifold.cylinder(1.2, 0.7, 0.7, 16).translate([x + (R - 1.2) * math.cos(a), (R - 1.2) * math.sin(a), 8])
        out = out + ring
        x += 2 * R + 3
    return out, [round(pencil_d + s, 2) for s in steps]


def to_mesh(m: Manifold):
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)


def from_stl(path):
    mesh = trimesh.load(path)
    return Manifold(Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32), tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))


def falu_upright(m: Manifold):
    return m.translate([-8.75, 0, -12.5]).rotate([90, 0, 0]).rotate([0, 0, 180])


def check(part: Manifold, falu: Path, pencil_d):
    mount = from_stl(falu / MOUNT_FILE)
    theirs = falu_upright(from_stl(falu / GUIDE_FILE))
    ox, oy = RING_AXIS

    def clash(m, dz):
        return (m.translate([ox, oy, dz]) ^ mount).volume()

    shifts = np.arange(0.0, 3.01, 0.05)
    ours = [clash(part, dz) for dz in shifts]
    best = int(np.argmin(ours))
    dz = float(shifts[best])
    report = {
        "seat_shift_mm": round(dz, 2),
        "overlap_with_mount_mm3": round(ours[best], 2),
    }
    shell = Manifold.cylinder(LENGTH, 8.76, 8.76, SEG) - Manifold.cylinder(LENGTH + 2, 7.2, 7.2, SEG).translate([0, 0, -1])
    a, b = part ^ shell, theirs ^ shell
    report["outside_differs_from_falu_mm3"] = round((a - b).volume() + (b - a).volume(), 2)
    report["outside_volume_mm3"] = round(a.volume(), 1)
    report["spring_module"] = next((k for k, (_, d) in SPRINGS.items() if pencil_d + 0.3 <= d), None)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pencil", type=float, default=Guide.pencil_d)
    p.add_argument("--tip", type=float, nargs=2, metavar=("DIAMETER", "LENGTH"))
    p.add_argument("--falu", type=Path)
    a = p.parse_args()
    g = Guide(pencil_d=a.pencil)
    if a.tip:
        g.tip_d, g.tip_len = a.tip
    OUT.mkdir(exist_ok=True)
    part = guide(g)
    rings, sizes = fit_rings(g.pencil_d)
    report = {"settings": asdict(g), "parts": {}}
    for name, m in (("pencil_guide", part), ("pencil_fit_rings", rings)):
        mesh = to_mesh(m)
        mesh.export(OUT / f"{name}.stl")
        report["parts"][name] = {"watertight": bool(mesh.is_watertight), "volume_cm3": round(mesh.volume / 1000, 2),
                                 "size_mm": [round(v, 1) for v in mesh.extents], "bodies": len(mesh.split(only_watertight=False))}
    report["fit_ring_bores_mm"] = sizes
    if a.falu:
        report["falu_check"] = check(part, a.falu, g.pencil_d)
    (OUT / "pencil_holder.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
